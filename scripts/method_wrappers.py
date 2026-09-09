"""
Wrappers around the four excitation-phase reconstruction methods explored in
notebooks 01-05 (+ 999), unified behind a single calling convention so they can
be benchmarked against each other across dose settings.

Every wrapper has the signature:

    wrapper(merged_scan, scan_mask, scan_params, **kwargs) -> dict

- merged_scan: (ny, nx, ndet, ndet) float array, checkerboard-interlaced 4D-STEM
  data. Each diffraction pattern is expected to be normalized (sums to 1) --
  i.e. a noiseless or already-renormalized-after-Poisson-noise probability
  image, matching the convention used throughout notebooks 01-03. Methods that
  build a native-resolution Dataset4dstem internally (co_reconstruction_regularized,
  co_reconstruction_regularized_reparam, linearized_model) accept either the
  raw array or an already-built Dataset4dstem -- passing a Dataset4dstem skips
  rebuilding/recalibrating it (see _as_dset4dstem).
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
from quantem.diffractive_imaging.object_models import (
    ObjectMultiplexed,
    ObjectMultiplexReparameterized,
    ObjectPixelated,
)
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


def _dset4dstem_from_merged_scan(merged_scan: np.ndarray, scan_params: dict) -> Dataset4dstem:
    """Build a native-resolution Dataset4dstem from a raw merged_scan array,
    calibrating the detector-pixel sampling to mrad via a probe-circle fit."""
    step = scan_params["scan_step"] / 2.0
    dset = Dataset4dstem.from_array(
        merged_scan, name="merged", sampling=(step, step, 1, 1), units=("A", "A", "pixels", "pixels")
    )
    probe_qy0, probe_qx0, probe_R = fit_probe_circle(dset.dp_mean.array, show=False)
    dset.sampling[2] = scan_params["probe_semiangle"] / probe_R
    dset.sampling[3] = scan_params["probe_semiangle"] / probe_R
    dset.units[2:] = ["mrad", "mrad"]
    return dset


def _as_dset4dstem(merged_scan: np.ndarray | Dataset4dstem, scan_params: dict) -> Dataset4dstem:
    """Accept either a raw merged_scan array or an already-built
    Dataset4dstem. Arrays are converted via _dset4dstem_from_merged_scan;
    an existing Dataset4dstem is passed through untouched (its calibration
    is trusted as-is, so the builder is skipped)."""
    if isinstance(merged_scan, Dataset4dstem):
        return merged_scan
    return _dset4dstem_from_merged_scan(merged_scan, scan_params)


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
    merged_scan: np.ndarray | Dataset4dstem,
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
    dset = _as_dset4dstem(merged_scan, scan_params)
    step = dset.sampling[0]
    if obj_padding is None:
        obj_padding = _pad_px(step)

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
        # "object": {"type": "plateau"},
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
        "dataset": {"descan_tv_weight": 0, "descan_shifts_constant": False, "center_scan_positions": True},

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


def co_reconstruction_regularized_reparam(
    merged_scan: np.ndarray | Dataset4dstem,
    scan_mask,
    scan_params,
    num_iters=25,
    tv_weight_mean=1.0,
    tv_weight_excitation=10.0,
    obj_padding=None,
    obj_lr=1e-3,
    probe_lr=1e-4,
    batch_size=128,
    **kwargs,
):
    """Same as co_reconstruction_regularized, but using ObjectMultiplexReparameterized
    (mean/excitation parameterization) instead of ObjectMultiplexed. Regularization is
    TV on the mean_obj and excitation_obj channels directly (tv_weight_mean,
    tv_weight_excitation) rather than ObjectMultiplexed's tv_weight/tv_channel_diff."""
    dset = _as_dset4dstem(merged_scan, scan_params)
    step = dset.sampling[0]
    if obj_padding is None:
        obj_padding = _pad_px(step)

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
    obj_model = ObjectMultiplexReparameterized.from_uniform(
        num_slices=1,
        slice_thicknesses=1,
        obj_type="pure_phase",
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
        # "object": {"type": "plateau"},
        "probe": {"type": "plateau"},
    }
    constraints = {
        "object": {
            "tv_weight_mean_xy": tv_weight_mean,
            "tv_weight_excitation_xy": tv_weight_excitation,
            "fix_potential_baseline": False,
            "identical_slices": True,
            "apply_fov_mask": False,
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
# Method 4: linearized forward model (notebook 05). Reconstructs the base
# (off) state with iterative ptychography, restricted to the real, individually
# -measured off-mode diffraction patterns at their true native scan positions
# (never block-averaged: two DPs from physically distinct positions are only
# interchangeable with the DP at their centroid if the object's phase is flat
# over that span, which is exactly untrue right where an excitation signal is
# expected). Reconstruction uses ObjectMultiplexed with patches_mask=scan_mask
# and tv_channel_diff=0 purely to route each native position's gradient to the
# right channel -- channel 0 (off) evolves only from off-mode data; channel 1
# (on) is decoupled and discarded.
#
# The excitation phase Delta-phi is then fit as a single convex least-squares
# problem: channel 0's own object/probe forward-simulate psi_off exactly at
# the real native on-mode positions (the "missing" off-mode frames there), and
# Delta-phi is solved with full-batch LBFGS against the raw on-mode DPs read
# directly out of merged_scan -- the predicted intensity is linear in
# Delta-phi, so this is a convex quadratic with a single minimum.
# ---------------------------------------------------------------------------

def _linearized_model_reconstruct_off(
    merged_scan: np.ndarray | Dataset4dstem,
    scan_mask,
    scan_params,
    num_iters_off=100,
    tv_weight=1.0,
    obj_padding=None,
    obj_lr=1e-3,
    probe_lr=1e-4,
    batch_size=512,
):
    """Reconstruct phi_off (and the probe) from only the real, individually
    measured off-mode frames of merged_scan at their native positions, then
    forward-simulate psi_off/Psi_off at the on-mode positions (the "missing"
    off-mode frames there). Returns everything the convex Delta-phi fit below
    needs, so that fit can be re-run cheaply (e.g. to sweep regularization
    weights) without repeating this expensive iterative reconstruction.

    batch_size defaults higher here than the other wrappers in this module: any
    noise left in phi_off from an under-converged/noisy Adam trajectory (small
    batches -> noisier gradient estimates) propagates straight into Delta-phi
    with no cancellation mechanism (unlike a paired-difference reconstruction),
    so this reconstruction benefits from lower-variance updates more than most."""
    dset = _as_dset4dstem(merged_scan, scan_params)
    merged_arr = dset.array
    step = dset.sampling[0]
    if obj_padding is None:
        obj_padding = _pad_px(step)
    ndet = merged_arr.shape[-1]

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
        rng=0,  # fixed seed: batch-shuffling order otherwise differs every call, so
        # phi_off's quality (and everything downstream of it) varied a lot run-to-run
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
        merged_arr.reshape(-1, ndet, ndet)[on_batch_np].astype(np.float32)
    ).to(device)

    I_off_baseline = Psi_off.abs() ** 2
    norm_const = I_off_baseline.mean()
    I_off_n = I_off_baseline / norm_const
    I_on_n = I_on_meas / norm_const
    Psi_off_n = Psi_off / torch.sqrt(norm_const)

    return dict(
        ptycho=ptycho,
        device=device,
        pad_r=pad_r,
        obj_shape_r=obj_shape_r,
        psi_off=psi_off,
        Psi_off_n=Psi_off_n,
        patch_indices_on=patch_indices_on,
        I_off_n=I_off_n,
        I_on_n=I_on_n,
        norm_const=norm_const,
    )


def _linearized_model_fit_delta_phi(
    state,
    max_iter_fit=300,
    tv_weight=0.0,
    tv_eps=1e-4,
    poisson_weighted=True,
    track_loss=False,
):
    """Fit Delta-phi against the off-state reconstruction in `state` (from
    _linearized_model_reconstruct_off).

    Two optional, still-convex refinements on top of the plain least-squares fit:

    - `poisson_weighted`: the measured on-mode frames are Poisson counts, so their
      variance scales with their own mean count rate -- an *unweighted* L2 loss
      implicitly (and wrongly) treats a noisy, near-empty detector pixel as being
      just as informative as a bright, well-measured one. Weighting each residual by
      1/I_off (I_off_baseline is a stable, noise-free proxy for the local count level,
      since Delta-phi is a small perturbation on top of it -- using it instead of the
      noisy I_on_meas itself avoids feeding the weighting its own noise) approximates
      the Gauss-Newton expansion of the Poisson negative log-likelihood. The weights
      are fixed constants (independent of delta_phi), so this is still a convex
      quadratic form, just a better-conditioned one; weights are capped at 10x the
      mean to keep near-empty (dark-field) pixels from dominating the fit.
    - `tv_weight`: a Charbonnier (smoothed total-variation) penalty
      sqrt(|grad(delta_phi)|^2 + eps^2) on the spatial gradient. Unlike a quadratic
      (Tikhonov) gradient penalty -- which needs coefficients in the millions to
      suppress shot noise at low dose, and is numerically unstable there (LBFGS's
      curvature estimate becomes ill-conditioned across loss terms that differ by six
      orders of magnitude) -- TV's cost grows *linearly*, not quadratically, with
      gradient size, so real blob-scale edges aren't punished nearly as hard as
      Tikhonov punishes them, and useful weights stay in a numerically tame O(1)-O(10)
      range. Charbonnier's sqrt(x^2+eps^2) is a smooth, convex function of delta_phi
      (the Euclidean norm of an affine map of delta_phi), so this keeps the whole fit
      a single convex problem -- still solved by the same full-batch LBFGS.
    """
    ptycho = state["ptycho"]
    device = state["device"]
    pad_r = state["pad_r"]
    psi_off = state["psi_off"]
    Psi_off_n = state["Psi_off_n"]
    patch_indices_on = state["patch_indices_on"]
    I_off_n = state["I_off_n"]
    I_on_n = state["I_on_n"]
    norm_const = state["norm_const"]

    if poisson_weighted:
        weight = 1.0 / (I_off_n + 0.2 * I_off_n.mean())
        weight = torch.clamp(weight, max=10.0 / I_off_n.mean())
        weight = weight / weight.mean()
    else:
        weight = None

    def forward_delta_I(dphi_full, patch_idx, psi, Psi_n):
        dphi_patch = dphi_full.reshape(-1)[patch_idx]
        pert = psi * dphi_patch
        Pert_full = torch.fft.fftshift(torch.fft.fft2(pert, norm="ortho"), dim=(-2, -1))
        Pert_n = Pert_full / torch.sqrt(norm_const)
        return -2.0 * torch.imag(torch.conj(Psi_n) * Pert_n)

    # Delta-phi enters the predicted intensity linearly (forward_delta_I is
    # linear in dphi_full), so this least-squares fit is a convex quadratic
    # with a single minimum: solve it full-batch with LBFGS (quasi-Newton +
    # a strong-Wolfe line search) rather than mini-batch first-order Adam --
    # it needs far fewer outer iterations and has no stochastic-gradient noise.
    delta_phi = torch.zeros(state["obj_shape_r"], dtype=torch.float32, device=device, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [delta_phi],
        lr=1.0,
        max_iter=max_iter_fit,
        tolerance_grad=1e-10,
        tolerance_change=1e-12,
        line_search_fn="strong_wolfe",
    )
    loss_history = []

    def closure():
        optimizer.zero_grad()
        I_on_pred = I_off_n + forward_delta_I(delta_phi, patch_indices_on, psi_off, Psi_off_n)
        resid2 = (I_on_pred - I_on_n) ** 2
        loss = torch.mean(weight * resid2) if weight is not None else torch.mean(resid2)
        if tv_weight > 0:
            gx = delta_phi[:-1, 1:] - delta_phi[:-1, :-1]
            gy = delta_phi[1:, :-1] - delta_phi[:-1, :-1]
            tv = torch.sqrt(gx**2 + gy**2 + tv_eps**2)
            loss = loss + tv_weight * tv.mean()
        loss.backward()
        if track_loss:
            loss_history.append(loss.item())
        return loss

    optimizer.step(closure)

    delta_phi_crop = _crop_2d(ptycho, delta_phi.detach().cpu().numpy(), pad_r)
    phase_off_crop = _cropped_channel_phase(ptycho, ptycho.obj_model.obj[0, ...], pad_r)[0]

    return dict(
        phase_off=phase_off_crop,
        phase_on=phase_off_crop + delta_phi_crop,
        delta_phi=delta_phi_crop,
        ptycho=ptycho,
        loss_history=loss_history,
    )


def linearized_model(
    merged_scan: np.ndarray | Dataset4dstem,
    scan_mask,
    scan_params,
    num_iters_off=100,
    tv_weight=1.0,
    obj_padding=None,
    obj_lr=1e-3,
    probe_lr=1e-4,
    batch_size=512,
    max_iter_fit=300,
    dphi_tv_weight=0.0,
    dphi_tv_eps=1e-4,
    poisson_weighted=True,
    **kwargs,
):
    """tv_weight regularizes the off-state object reconstruction itself (see
    _linearized_model_reconstruct_off); dphi_tv_weight is the separate Charbonnier-TV
    weight on the convex Delta-phi fit (see _linearized_model_fit_delta_phi) -- the
    two are unrelated regularizers on two different, decoupled optimization problems,
    just given different names to avoid confusing them with each other."""
    state = _linearized_model_reconstruct_off(
        merged_scan, scan_mask, scan_params,
        num_iters_off=num_iters_off, tv_weight=tv_weight, obj_padding=obj_padding,
        obj_lr=obj_lr, probe_lr=probe_lr, batch_size=batch_size,
    )
    return _linearized_model_fit_delta_phi(
        state, max_iter_fit=max_iter_fit, tv_weight=dphi_tv_weight, tv_eps=dphi_tv_eps,
        poisson_weighted=poisson_weighted,
    )


METHODS = {
    "reconstruct_and_subtract": reconstruct_and_subtract,
    "co_reconstruction_regularized": co_reconstruction_regularized,
    "direct_ssb": direct_ssb,
    "linearized_model": linearized_model,
}
