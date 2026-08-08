"""
Wrappers around the four excitation-phase reconstruction methods explored in
notebooks 01-05 (+ 999), unified behind a single calling convention so they can
be benchmarked against each other across dose settings.

Every wrapper has the signature:

    wrapper(merged_scan, scan_mask, scan_params, **kwargs) -> dict

- merged_scan: (ny, nx, ndet, ndet) float array, checkerboard-interlaced 4D-STEM
  data. Each diffraction pattern is expected to be normalized (sums to 1) --
  i.e. a noiseless or already-renormalized-after-Poisson-noise probability
  image, matching the convention used throughout notebooks 01-03.
- scan_mask: (ny, nx) array, 0 for "off"/base-state scan positions, 1 for
  "on"/excited-state positions (checkerboard, (i+j) even <-> mask==1).
- scan_params: dict with keys 'probe_energy' (eV), 'probe_semiangle' (mrad),
  'probe_defocus' (A), 'scan_step' (A, the *coarse* / half-resolution grid
  step -- i.e. the step of a single de-interlaced channel; the native
  full-resolution merged-scan step is scan_step / 2).

Return dict always has 'phase_off', 'phase_on', 'delta_phi' (= phase_on -
phase_off wherever defined) plus method-specific extras. Any of phase_off /
phase_on may be None for methods that only recover the differential signal
directly.
"""

import numpy as np
import torch

import ptycho_wrappers
from quantem.core.datastructures import Dataset4d, Dataset4dstem
from quantem.core.utils.diffractive_imaging_utils import fit_probe_circle
from quantem.diffractive_imaging.complex_probe import (
    evaluate_probe,
    polar_coordinates,
    spatial_frequencies,
)
from quantem.diffractive_imaging.dataset_models import PtychographyDatasetRaster
from quantem.diffractive_imaging.detector_models import DetectorPixelated
from quantem.diffractive_imaging.direct_ptychography import DirectPtychography
from quantem.diffractive_imaging.object_models import ObjectMultiplexed, ObjectPixelated
from quantem.diffractive_imaging.probe_models import ProbePixelated
from quantem.diffractive_imaging.ptychography import Ptychography
from quantem.diffractive_imaging.ptycho_utils import center_crop_arr


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

def block_average_checkerboard(merged_scan, scan_mask):
    """Untangle a checkerboard-interlaced scan into off/on half-resolution
    datacubes by averaging the 2 "off" and 2 "on" samples in each 2x2 unit
    cell (notebook 01's convention)."""
    ny, nx = merged_scan.shape[:2]
    ty, tx = ny // 2, nx // 2
    off_cube = np.zeros((ty, tx, *merged_scan.shape[2:]), dtype=merged_scan.dtype)
    on_cube = np.zeros_like(off_cube)
    for i in range(ty):
        for j in range(tx):
            cell = merged_scan[2 * i : 2 * i + 2, 2 * j : 2 * j + 2]
            m = scan_mask[2 * i : 2 * i + 2, 2 * j : 2 * j + 2]
            off_cube[i, j] = cell[m == 0].mean(axis=0)
            on_cube[i, j] = cell[m == 1].mean(axis=0)
    return off_cube, on_cube


def _phase(obj_tensor):
    return obj_tensor.detach().cpu().angle().numpy()


# Uniform *physical* padding margin (Angstrom) added around the object FOV in
# every method. Methods reconstruct on different pixel grids (the coarse,
# half-resolution checkerboard-block-averaged grid for method 1, vs. the full
# native-resolution grid for methods 2/4), so an equal *pixel* padding count
# would in fact be wildly non-uniform physically (e.g. 32px on a 2x-coarser
# grid is 4x the physical margin of 16px on the native grid, which is what
# the co-reconstruction wrapper was using before this fix). Converting a
# single physical target into per-method pixel counts keeps the margin
# actually equal.
PAD_PHYSICAL_A = 12.0


def _pad_px(step_A, pad_physical_A=PAD_PHYSICAL_A):
    return int(round(pad_physical_A / step_A))


def _crop_2d(ptycho, arr_2d, pad_px):
    """Crop+derotate a single 2D real-valued array (e.g. delta_phi, native
    padded pixel grid) to the true scanned FOV, using the same
    _crop_rotate_obj_fov logic the reconstruction itself uses for its own
    object array -- so padding pixels never leak into a saved/plotted image."""
    cropped = ptycho._crop_rotate_obj_fov(arr_2d[None, ...], padding=pad_px)
    return cropped[0]


def _cropped_channel_phase(ptycho, obj_channel, pad_px):
    """Like ptycho.obj_cropped, but for a single already-indexed channel of
    an ObjectMultiplexed object (shape (num_slices, y, x)). The obj_cropped
    property assumes a channel-less object array (crops using
    self.obj_shape_crop, a 3-tuple), so it errors on the raw 4D multiplexed
    (channel, slice, y, x) tensor -- this crops one channel at a time
    instead, then applies the same mean-centering of the arbitrary global
    phase gauge that obj_cropped does for display."""
    arr = ptycho._to_numpy(obj_channel)
    cropped = ptycho._crop_rotate_obj_fov(arr, padding=pad_px)
    ph = np.angle(cropped)
    return ph - ph.mean()


def _match_shape_2d(*arrays):
    """Center-crop a set of 2D arrays down to their common shape. Two
    *independently* reconstructed Ptychography objects (as in
    reconstruct_and_subtract) each fit their own probe-circle calibration,
    so their real-space pixel sizes -- and thus obj_shape_crop, which is
    derived from dset.fov / sampling -- can differ by a pixel or two even
    when built from identically-shaped input data."""
    shape = tuple(min(a.shape[i] for a in arrays) for i in range(arrays[0].ndim))
    return [center_crop_arr(a, shape, pad_if_needed=False) for a in arrays]


# ---------------------------------------------------------------------------
# Method 1: reconstruct & subtract (notebook 01)
# ---------------------------------------------------------------------------

def reconstruct_and_subtract(
    merged_scan,
    scan_mask,
    scan_params,
    num_iters=50,
    tv_weight=1.0,
    obj_padding=None,
    obj_lr=5e-1,
    probe_lr=1e-3,
    batch_size=128,
    **kwargs,
):
    """Split the interlaced scan into two independent half-resolution
    datasets (off/on), fully reconstruct each with ordinary iterative
    ptychography, then subtract the two phase images."""
    off_cube, on_cube = block_average_checkerboard(merged_scan, scan_mask)
    step = scan_params["scan_step"]
    if obj_padding is None:
        obj_padding = _pad_px(step)

    dset_off = Dataset4dstem.from_array(
        off_cube, name="off", sampling=(step, step, 1, 1), units=("A", "A", "pixels", "pixels")
    )
    dset_on = Dataset4dstem.from_array(
        on_cube, name="on", sampling=(step, step, 1, 1), units=("A", "A", "pixels", "pixels")
    )

    common = dict(
        tv_weight=tv_weight,
        num_iters=num_iters,
        probe_energy=scan_params["probe_energy"],
        probe_semiangle=scan_params["probe_semiangle"],
        probe_defocus=scan_params["probe_defocus"],
        obj_padding=obj_padding,
        obj_lr=obj_lr,
        probe_lr=probe_lr,
        batch_size=batch_size,
    )
    ptycho_off = ptycho_wrappers.ptycho_wrapper(dset_off, **common)
    ptycho_on = ptycho_wrappers.ptycho_wrapper(dset_on, **common)

    # obj_cropped strips the object-padding border pixels (unconstrained by
    # any data, since no probe ever visits them) and independently
    # zero-means each reconstruction's arbitrary global phase gauge -- both
    # necessary before subtracting two *independently* reconstructed phases,
    # otherwise the padding border and any gauge mismatch between the two
    # separate optimizations dominate delta_phi's contrast.
    phase_off = np.angle(ptycho_off.obj_cropped[0, ...])
    phase_on = np.angle(ptycho_on.obj_cropped[0, ...])
    phase_off, phase_on = _match_shape_2d(phase_off, phase_on)

    return dict(
        phase_off=phase_off,
        phase_on=phase_on,
        delta_phi=phase_on - phase_off,
        ptycho_off=ptycho_off,
        ptycho_on=ptycho_on,
    )


# ---------------------------------------------------------------------------
# Method 2: co-reconstruction with regularization on the differential signal
# (notebooks 02/03 -- ObjectMultiplexed, joint two-channel reconstruction).
#
# Note: notebook 03 ("Joint TV") sets constraints["object"]["joint_tv"], but
# ObjectMultiplexed.get_joint_tv_loss is never actually invoked from
# apply_soft_constraints in the current codebase (dead code -- setting
# "joint_tv" has no effect on the loss). The regularizer that *is* wired up
# and actually penalizes the differential signal is "tv_channel_diff"
# (notebook 02), so that's what this wrapper uses.
# ---------------------------------------------------------------------------

def co_reconstruction_regularized(
    merged_scan,
    scan_mask,
    scan_params,
    num_iters=25,
    tv_weight=1.0,
    tv_channel_diff=10.0,
    obj_padding=None,
    obj_lr=1e-3,
    probe_lr=1e-4,
    batch_size=128,
    **kwargs,
):
    step = scan_params["scan_step"] / 2.0
    if obj_padding is None:
        obj_padding = _pad_px(step)
    dset = Dataset4dstem.from_array(
        merged_scan, name="merged", sampling=(step, step, 1, 1), units=("A", "A", "pixels", "pixels")
    )
    probe_qy0, probe_qx0, probe_R = fit_probe_circle(dset.dp_mean.array, show=False)
    dset.sampling[2] = scan_params["probe_semiangle"] / probe_R
    dset.sampling[3] = scan_params["probe_semiangle"] / probe_R
    dset.units[2:] = ["mrad", "mrad"]

    pdset = PtychographyDatasetRaster.from_dataset4dstem(dset)
    pdset.preprocess(
        com_fit_function="constant",
        plot_rotation=False,
        plot_com=False,
        probe_energy=scan_params["probe_energy"],
        force_com_rotation=0,
        force_com_transpose=False,
    )

    probe_params = {
        "energy": scan_params["probe_energy"],
        "defocus": scan_params["probe_defocus"],
        "semiangle_cutoff": scan_params["probe_semiangle"],
    }
    detector_model = DetectorPixelated()
    probe_model = ProbePixelated.from_params(num_probes=1, probe_params=probe_params)
    obj_model = ObjectMultiplexed.from_uniform(
        num_slices=1,
        slice_thicknesses=1,
        obj_type="pure_phase",
        num_channels=2,
        patches_mask=torch.tensor(scan_mask),
    )

    ptycho = Ptychography.from_models(
        dset=pdset,
        obj_model=obj_model,
        probe_model=probe_model,
        detector_model=detector_model,
        device="cuda",
    )
    ptycho.preprocess(obj_padding_px=(obj_padding, obj_padding), batch_size=batch_size)

    opt_params = {
        "object": {"type": "adam", "lr": obj_lr},
        "probe": {"type": "adam", "lr": probe_lr},
    }
    scheduler_params = {
        "object": {"type": "exp", "factor": 9e-2},
        "probe": {"type": "plateau"},
    }
    constraints = {
        "object": {
            "tv_weight_xy": tv_weight,
            "fix_potential_baseline": False,
            "identical_slices": True,
            "apply_fov_mask": False,
            "tv_channel_diff": tv_channel_diff,
        },
        "probe": {"center_probe": False, "orthogonalize_probe": True},
        "dataset": {"descan_tv_weight": 0, "descan_shifts_constant": False},
    }

    ptycho.reconstruct(
        num_iters=num_iters,
        reset=True,
        autograd=True,
        device="cuda",
        constraints=constraints,
        optimizer_params=opt_params,
        scheduler_params=scheduler_params,
        batch_size=batch_size,
        multichannel_mode=True,
    )

    recon_off = _cropped_channel_phase(ptycho, ptycho.obj_model.obj[0, ...], ptycho.obj_padding_px)[0]
    recon_on = _cropped_channel_phase(ptycho, ptycho.obj_model.obj[1, ...], ptycho.obj_padding_px)[0]

    return dict(
        phase_off=recon_off,
        phase_on=recon_on,
        delta_phi=recon_on - recon_off,
        ptycho=ptycho,
    )


# ---------------------------------------------------------------------------
# Method 3: direct (closed-form) SSB (notebook 999's calibrated fix to
# notebook 04's "Multiplexed SSB" section).
#
# Reconstructs the mean object Obar (baseband SSB) and the carrier-shifted
# differential field dO (SSB on the same data with the scan-Fourier-transform
# fftshifted by Nyquist, i.e. the checkerboard-modulation carrier), combined
# with a *coherent complex* average over bright-field detector pixels (not
# angle-then-average), then recovers the excitation phase in closed form via
#     delta_phi = 2*arctan[Im(dO / Obar)]
# No optimizer, no iteration -- this is the fast, direct baseline.
# ---------------------------------------------------------------------------

def _reconstruct_complex_ssb(ptycho, vbf_fourier, force_dc):
    qxa, qya = ptycho._return_upsampled_qgrid(None)
    kxa, kya = spatial_frequencies(
        ptycho.gpts, ptycho.sampling, rotation_angle=ptycho.rotation_angle, device=ptycho.device
    )
    k, phi = polar_coordinates(kxa, kya)
    alpha = k * ptycho.wavelength

    cmplx_probe = evaluate_probe(
        alpha,
        phi,
        ptycho.semiangle_cutoff,
        ptycho.angular_sampling,
        ptycho.wavelength,
        aberration_coefs=ptycho.aberration_coefs,
    )

    batch_idx = torch.arange(ptycho.num_bf, device=ptycho.device)
    operator = ptycho._compute_gamma_operator(
        kxa, kya, qxa, qya, ptycho.aberration_coefs, cmplx_probe, batch_idx,
        asymmetric_version=True, normalize=True,
    )

    fourier_factor = vbf_fourier[batch_idx] * operator
    if force_dc:
        fourier_factor = fourier_factor.clone()
        fourier_factor[..., 0, 0] = ptycho._dc_per_image

    corrected_stack = torch.fft.ifft2(fourier_factor)
    return corrected_stack.mean(dim=0)


def _direct_excitation_phase(ptycho):
    ptycho._preprocess()
    vbf_baseband = ptycho._vbf_fourier.clone()
    vbf_carrier = torch.fft.fftshift(ptycho._vbf_fourier, dim=(-1, -2))

    Obar = _reconstruct_complex_ssb(ptycho, vbf_baseband, force_dc=True)
    dO = _reconstruct_complex_ssb(ptycho, vbf_carrier, force_dc=False)

    delta_phi = 2.0 * torch.arctan((dO / Obar).imag)
    return delta_phi, Obar, dO


def direct_ssb(merged_scan, scan_mask, scan_params, **kwargs):
    step = scan_params["scan_step"] / 2.0
    mean_dp = merged_scan.mean(axis=(0, 1))
    _, _, probe_R = fit_probe_circle(mean_dp, show=False)
    mrad_per_px = scan_params["probe_semiangle"] / probe_R

    dataset_merged = Dataset4d.from_array(
        merged_scan,
        sampling=(step, step, mrad_per_px, mrad_per_px),
        units=("A", "A", "mrad", "mrad"),
    )

    ptycho = DirectPtychography.from_dataset4d(
        dataset_merged,
        energy=scan_params["probe_energy"],
        semiangle_cutoff=scan_params["probe_semiangle"],
        aberration_coefs={"defocus": scan_params["probe_defocus"]},
        verbose=False,
    )

    delta_phi, Obar, dO = _direct_excitation_phase(ptycho)

    return dict(
        phase_off=None,
        phase_on=None,
        delta_phi=delta_phi.detach().cpu().numpy(),
        mean_object_phase=Obar.detach().cpu().angle().numpy(),
        mean_object_amp=Obar.detach().cpu().abs().numpy(),
        ptycho=ptycho,
    )


# ---------------------------------------------------------------------------
# Method 4: linearized forward model (notebook 05's "honest" section --
# reconstructed phi_off + probe, not ground truth). Reconstructs the base
# (off) state with ordinary iterative ptychography, then fits the excitation
# phase Delta-phi as a single convex least-squares problem against the "on"
# pixels only, using the reconstruction's own forward model for psi_off.
#
# v2: the first version reconstructed phi_off from a 2x2-checkerboard-
# block-averaged half-resolution "off_cube", then compared the linear fit's
# predicted intensity against a *separately* block-averaged "on_cube" at the
# same coarse grid. That's an extra, uncontrolled approximation on top of the
# Delta-phi linearization itself: averaging two *raw* diffraction patterns
# from two physically distinct native positions is not the same as the
# diffraction pattern the probe would actually produce at their centroid
# whenever the object's phase varies over that 1-native-pixel span -- exactly
# what happens near the excitation signal we're trying to recover. That
# mismatch behaves like an extra unmodeled term in the data, which the convex
# fit dumps into delta_phi wherever the (near-singular) forward operator lets
# it, producing the large, physically implausible delta_phi seen even at
# infinite (noiseless) dose. Regularizing it away (the previous fix) treated
# the symptom, not the cause.
#
# Fix: never average diffraction patterns. Reconstruct phi_off at *native*
# resolution using ObjectMultiplexed with patches_mask=scan_mask (the same
# per-position light-mode routing the co-reconstruction wrapper already uses)
# and tv_channel_diff=0 -- this makes channel 0 (off) evolve purely from
# gradients contributed by the real, raw, individually-measured off-mode DPs
# at their true native positions (channel 1 just absorbs the on-mode data and
# is discarded; decoupling means it can't leak into channel 0). psi_off is
# then forward-simulated with channel 0's own object/probe *exactly at the
# real native on-mode positions* (forcing channel-0 patches there via
# `_get_obj_patches`, bypassing the mask's own on/off routing), and fit
# against the raw (never averaged) on-mode DPs read directly out of
# merged_scan at those same positions.
# ---------------------------------------------------------------------------

def linearized_model(
    merged_scan,
    scan_mask,
    scan_params,
    num_iters_off=25,
    tv_weight=1.0,
    obj_padding=None,
    obj_lr=1e-3,
    probe_lr=1e-4,
    batch_size=128,
    n_iters_fit=300,
    fit_lr=5e-3,
    fit_batch_size=2000,
    l2_weight=0.0,
    **kwargs,
):
    step = scan_params["scan_step"] / 2.0
    if obj_padding is None:
        obj_padding = _pad_px(step)
    ndet = merged_scan.shape[-1]

    dset = Dataset4dstem.from_array(
        merged_scan, name="merged", sampling=(step, step, 1, 1), units=("A", "A", "pixels", "pixels")
    )
    probe_qy0, probe_qx0, probe_R = fit_probe_circle(dset.dp_mean.array, show=False)
    dset.sampling[2] = scan_params["probe_semiangle"] / probe_R
    dset.sampling[3] = scan_params["probe_semiangle"] / probe_R
    dset.units[2:] = ["mrad", "mrad"]

    pdset = PtychographyDatasetRaster.from_dataset4dstem(dset)
    pdset.preprocess(
        com_fit_function="constant",
        plot_rotation=False,
        plot_com=False,
        probe_energy=scan_params["probe_energy"],
        force_com_rotation=0,
        force_com_transpose=False,
    )

    probe_params = {
        "energy": scan_params["probe_energy"],
        "defocus": scan_params["probe_defocus"],
        "semiangle_cutoff": scan_params["probe_semiangle"],
    }
    detector_model = DetectorPixelated()
    probe_model = ProbePixelated.from_params(num_probes=1, probe_params=probe_params)
    # 2 channels purely so patches_mask can route each native position's
    # gradient to the right one; channel 1 (on) is never used afterwards.
    obj_model = ObjectMultiplexed.from_uniform(
        num_slices=1,
        slice_thicknesses=1,
        obj_type="pure_phase",
        num_channels=2,
        patches_mask=torch.tensor(scan_mask),
    )

    ptycho = Ptychography.from_models(
        dset=pdset,
        obj_model=obj_model,
        probe_model=probe_model,
        detector_model=detector_model,
        device="cuda",
    )
    ptycho.preprocess(obj_padding_px=(obj_padding, obj_padding), batch_size=batch_size)

    opt_params = {
        "object": {"type": "adam", "lr": obj_lr},
        "probe": {"type": "adam", "lr": probe_lr},
    }
    scheduler_params = {
        "object": {"type": "exp", "factor": 9e-2},
        "probe": {"type": "plateau"},
    }
    constraints = {
        "object": {
            "tv_weight_xy": tv_weight,
            "fix_potential_baseline": False,
            "identical_slices": True,
            "apply_fov_mask": False,
            "tv_channel_diff": 0.0,  # decoupled: channel 0 must only see off-mode data
        },
        "probe": {"center_probe": False, "orthogonalize_probe": True},
        "dataset": {"descan_tv_weight": 0, "descan_shifts_constant": False},
    }

    ptycho.reconstruct(
        num_iters=num_iters_off,
        reset=True,
        autograd=True,
        device="cuda",
        constraints=constraints,
        optimizer_params=opt_params,
        scheduler_params=scheduler_params,
        batch_size=batch_size,
        multichannel_mode=True,
    )

    device = ptycho.device
    pad_r = ptycho.obj_padding_px
    obj_shape_r = tuple(int(x) for x in ptycho.obj_shape_full[-2:])

    on_batch = torch.nonzero(
        torch.tensor(scan_mask.reshape(-1) == 1), as_tuple=True
    )[0].to(device)

    psi_list, Psi_list, patch_list = [], [], []
    chunk = 2000
    with torch.no_grad():
        obj_off = ptycho.obj_model.obj[0, ...]  # channel 0 (off), fixed for this loop
        for start in range(0, len(on_batch), chunk):
            batch_idx = on_batch[start : start + chunk]
            patch_indices, _, positions_px_fractional, _ = ptycho.dset.forward(batch_idx, pad_r)
            # force channel-0 (off) patches regardless of this position's own
            # mask label -- we want psi_off AT the on-mode positions, not
            # channel 1's own (decoupled, off-topic) object there.
            obj_patches = ptycho.obj_model._get_obj_patches(obj_off, patch_indices)
            shifted_probes = ptycho.probe_model.forward(positions_px_fractional)
            _, overlap = ptycho.forward_operator(obj_patches, shifted_probes)
            psi = overlap[0]
            Psi = torch.fft.fftshift(torch.fft.fft2(psi, norm="ortho"), dim=(-2, -1))
            psi_list.append(psi)
            Psi_list.append(Psi)
            patch_list.append(patch_indices)

    psi_off = torch.cat(psi_list, dim=0)
    Psi_off = torch.cat(Psi_list, dim=0)
    patch_indices_on = torch.cat(patch_list, dim=0)

    # raw, single-shot on-mode DPs at their true native positions -- never averaged
    on_batch_np = on_batch.detach().cpu().numpy()
    I_on_meas = torch.from_numpy(
        merged_scan.reshape(-1, ndet, ndet)[on_batch_np].astype(np.float32)
    ).to(device)

    I_off_baseline = Psi_off.abs() ** 2
    norm_const = I_off_baseline.mean()
    I_off_n = I_off_baseline / norm_const
    I_on_n = I_on_meas / norm_const
    Psi_off_n = Psi_off / torch.sqrt(norm_const)

    def forward_delta_I(dphi_full, patch_idx, psi, Psi_n):
        dphi_patch = dphi_full.reshape(-1)[patch_idx]
        pert = psi * dphi_patch
        Pert_full = torch.fft.fftshift(torch.fft.fft2(pert, norm="ortho"), dim=(-2, -1))
        Pert_n = Pert_full / torch.sqrt(norm_const)
        return -2.0 * torch.imag(torch.conj(Psi_n) * Pert_n)

    delta_phi = torch.zeros(obj_shape_r, dtype=torch.float32, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([delta_phi], lr=fit_lr)

    # mini-batch over on-mode positions rather than always full-batch: the
    # native grid has ~2x as many on-positions as the old half-resolution
    # grid (plus a larger padded object now that padding is a uniform
    # physical margin, see PAD_PHYSICAL_A), so a full-batch fit can OOM on
    # larger scans even though it fit fine on the smaller ones this was
    # first written against.
    n_on_total = patch_indices_on.shape[0]
    mb_size = min(fit_batch_size, n_on_total)
    fit_rng = np.random.default_rng(0)

    for _ in range(n_iters_fit):
        idx = torch.from_numpy(fit_rng.choice(n_on_total, size=mb_size, replace=False)).to(device)
        optimizer.zero_grad()
        I_on_pred = I_off_n[idx] + forward_delta_I(
            delta_phi, patch_indices_on[idx], psi_off[idx], Psi_off_n[idx]
        )
        data_loss = torch.mean((I_on_pred - I_on_n[idx]) ** 2)
        loss = data_loss + l2_weight * torch.mean(delta_phi**2) if l2_weight > 0 else data_loss
        loss.backward()
        optimizer.step()

    delta_phi_crop = _crop_2d(ptycho, delta_phi.detach().cpu().numpy(), pad_r)
    phase_off_crop = _cropped_channel_phase(ptycho, ptycho.obj_model.obj[0, ...], pad_r)[0]

    return dict(
        phase_off=phase_off_crop,
        phase_on=phase_off_crop + delta_phi_crop,
        delta_phi=delta_phi_crop,
        ptycho=ptycho,
    )


METHODS = {
    "reconstruct_and_subtract": reconstruct_and_subtract,
    "co_reconstruction_regularized": co_reconstruction_regularized,
    "direct_ssb": direct_ssb,
    "linearized_model": linearized_model,
}
