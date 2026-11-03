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

import json
import time

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
# Aberration-free datasets for the direct methods.
#
# The 00 block runs at C10 = 100 A because the proposal's experiment does. For SSB that
# defocus is not a detail: chi = pi*C10*alpha^2/lambda = 1.85 pi at 60 kV, so the phase
# wraps almost twice across the aperture and the two sidebands interfere through
# exp(2i chi) instead of adding. SSB assumes it knows that phase; the iterative methods
# solve for it. Hence a C10 = 0 twin of the dataset, generated once and cached, for the
# 03.x notebooks to work from.
# ---------------------------------------------------------------------------

def make_scan_mask(n_rows, n_cols=None):
    """1 where the light is ON, 0 where it is OFF.

    mask[i, j] = (i + j) % 2 == 0 -> light-on. This is the convention recorded in the
    geometry files and assumed by every wrapper here; do not re-derive it elsewhere.
    """
    n_cols = n_rows if n_cols is None else n_cols
    ii, jj = np.indices((n_rows, n_cols))
    return ((ii + jj) % 2 == 0).astype(np.uint8)


def interlace_scans(cube_off, cube_on, out=None):
    """Checkerboard-interlace a light-off and a light-on scan into a single cube.

    Even-even and odd-odd positions are taken from `cube_on`, the rest from `cube_off`,
    which is exactly make_scan_mask()'s (i + j) even -> on.

    Written as four strided assignments rather than np.where so that only the output is
    allocated. np.where(mask[..., None, None], on, off) holds both inputs plus a
    full-size broadcast of the mask at once -- 40 GB for a 300^2 x 192^2 pair -- and
    builds the same answer. Inputs may be memory-mapped and are never modified.
    """
    if cube_off.shape != cube_on.shape:
        raise ValueError(f"shape mismatch: off {cube_off.shape} vs on {cube_on.shape}")
    if out is None:
        out = np.empty(cube_off.shape, dtype=np.float32)
    out[0::2, 0::2] = cube_on[0::2, 0::2]     # (i + j) even
    out[1::2, 1::2] = cube_on[1::2, 1::2]     # (i + j) even
    out[0::2, 1::2] = cube_off[0::2, 1::2]    # (i + j) odd
    out[1::2, 0::2] = cube_off[1::2, 0::2]    # (i + j) odd
    return out


# Which geometry fields an override is allowed to touch, and what has to be
# recomputed when it does. Anything not listed here is inherited unchanged from the
# 00 block so that two datasets stay physically comparable.
_DATASET_OVERRIDES = ("defocus_C10_ang", "convergence_semiangle_mrad",
                      "step_ang", "scan_extent_ang")


def _derive_meta(base_meta, overrides):
    """Apply overrides to a copy of the 00 geometry and bring every derived field
    back in line, so downstream code can read the result without knowing what changed."""
    bad = set(overrides) - set(_DATASET_OVERRIDES)
    if bad:
        raise ValueError(f"unsupported override(s) {sorted(bad)}; "
                         f"allowed: {list(_DATASET_OVERRIDES)}")
    m = json.loads(json.dumps(base_meta))
    alpha = float(overrides.get("convergence_semiangle_mrad",
                                m["beam"]["convergence_semiangle_mrad"]))
    c10 = float(overrides.get("defocus_C10_ang", m["beam"]["defocus_C10_ang"]))
    step = float(overrides.get("step_ang", m["scan"]["step_ang"]))
    extent = float(overrides.get("scan_extent_ang", m["scan"]["extent_ang"]))
    lam = m["beam"]["wavelength_ang"]
    n_scan = int(round(extent / step))

    m["beam"]["convergence_semiangle_mrad"] = alpha
    m["beam"]["defocus_C10_ang"] = c10
    m["beam"]["aberrations"] = ("none at all: C10 = 0, chi = 0 across the aperture"
                                if c10 == 0 else f"none beyond C10 = {c10} A")
    m["scan"].update(n_positions_per_side=n_scan, n_positions=n_scan ** 2,
                     step_ang=step, extent_ang=extent, span_ang=(n_scan - 1) * step,
                     axis_first_ang=-(n_scan - 1) * step / 2,
                     axis_last_ang=(n_scan - 1) * step / 2)
    m["scan"]["scan_params_for_method_wrappers"].update(
        probe_semiangle=alpha, probe_defocus=c10, scan_step=2 * step)
    bandwidth = 2 * alpha * 1e-3 / lam
    m["probe"]["bandwidth_inv_ang"] = bandwidth
    m["probe"]["alias_free_step_ang"] = 1.0 / (2 * bandwidth)
    m["probe"]["geometric_radius_alpha_C10_ang"] = abs(c10) * alpha * 1e-3
    m["scan"]["nyquist_inv_ang"] = 1.0 / (2 * step)
    m["scan"]["oversampling_vs_bandwidth"] = 1.0 / (2 * step) / bandwidth
    return m


def ensure_dataset(data_dir, base_meta, overrides=None, prefix="ds", chunk=1024,
                   verbose=True):
    """Simulate, or load from cache, the base and excited states at a given geometry.

    `overrides` may set defocus_C10_ang, convergence_semiangle_mrad, step_ang and
    scan_extent_ang; the cell, object grid, detector and energy always come from
    `base_meta`. Returns (base_path, excited_path, meta).

    The cache is keyed by `prefix` AND validated: if {prefix}_geometry.json exists but
    describes different optics or a different scan, the files are rebuilt rather than
    silently reused. That is the failure this guards against -- changing a step size
    and unknowingly reloading the old cubes.
    """
    from pathlib import Path
    data_dir = Path(data_dir)
    base_p = data_dir / f"{prefix}_base_noiseless.npy"
    exc_p = data_dir / f"{prefix}_excited_noiseless.npy"
    geo_p = data_dir / f"{prefix}_geometry.json"
    meta = _derive_meta(base_meta, overrides or {})

    def _key(m):
        return (m["beam"]["defocus_C10_ang"], m["beam"]["convergence_semiangle_mrad"],
                m["scan"]["step_ang"], m["scan"]["n_positions_per_side"],
                m["detector"]["n_pixels_binned"])

    if base_p.exists() and exc_p.exists() and geo_p.exists():
        with open(geo_p) as handle:
            cached = json.load(handle)
        if _key(cached) == _key(meta):
            if verbose:
                print(f"  cached: {base_p.name}, {exc_p.name}  "
                      f"(C10 {meta['beam']['defocus_C10_ang']} A, "
                      f"{meta['beam']['convergence_semiangle_mrad']} mrad, "
                      f"{meta['scan']['n_positions_per_side']}^2 at "
                      f"{meta['scan']['step_ang']} A)")
            return base_p, exc_p, meta
        if verbose:
            print(f"  cache MISMATCH for {prefix}: stored {_key(cached)} != "
                  f"requested {_key(meta)} -- rebuilding")

    import abtem
    from abtem.core.energy import energy2sigma
    from abtem.potentials.iam import PotentialArray
    from abtem.scan import CustomScan
    try:
        import cupy  # noqa: F401
        abtem.config.set({"device": "gpu"})
    except Exception:
        abtem.config.set({"device": "cpu"})

    cell = meta["cell"]["extent_ang"]; n_obj = meta["cell"]["n_pixels"]
    dx = meta["cell"]["dx_ang"]; bin_ = meta["detector"]["bin"]
    n_det = meta["detector"]["n_pixels_binned"]
    n_nat = meta["detector"]["n_pixels_native"]
    crop = meta["detector"]["collection_half_angle_mrad"]
    step = meta["scan"]["step_ang"]; n_scan = meta["scan"]["n_positions_per_side"]
    slice_t = meta["simulation"]["slice_thickness_ang"]
    truth = np.load(data_dir / "ideal_truth.npz")

    probe = abtem.Probe(energy=meta["beam"]["accelerating_voltage_eV"],
                        semiangle_cutoff=meta["beam"]["convergence_semiangle_mrad"],
                        defocus=meta["beam"]["defocus_C10_ang"],
                        extent=cell, gpts=n_obj)
    det = abtem.PixelatedDetector(max_angle=crop)
    sig = float(energy2sigma(meta["beam"]["accelerating_voltage_eV"]))
    axis = (np.arange(n_scan) - (n_scan - 1) / 2) * step
    ga, gb = np.meshgrid(axis, axis, indexing="ij")
    pos = np.stack([ga.ravel(), gb.ravel()], 1) + (n_obj // 2) * dx

    def crop_bin(a):
        st = (a.shape[-1] - n_nat) // 2
        if st > 0:
            a = a[..., st:st + n_nat, st:st + n_nat]
        return a.reshape(*a.shape[:-2], n_det, bin_, n_det, bin_).sum(axis=(-3, -1))

    for name, path in (("phase_base", base_p), ("phase_excited", exc_p)):
        pot = PotentialArray(array=(truth[name] / (sig * slice_t))[None].astype(np.float32),
                             slice_thickness=slice_t, sampling=(dx, dx))
        out = np.empty((len(pos), n_det, n_det), np.float32)
        t0 = time.time()
        for lo in range(0, len(pos), chunk):
            hi = min(lo + chunk, len(pos))
            mm = probe.scan(pot, scan=CustomScan(pos[lo:hi]), detectors=det).compute()
            arr = crop_bin(mm.array)
            out[lo:hi] = (arr.get() if hasattr(arr, "get") else np.asarray(arr)).astype(np.float32)
            del mm, arr
        out = out.reshape(n_scan, n_scan, n_det, n_det)
        out /= out.sum(axis=(-2, -1), keepdims=True)
        np.save(path, out)
        if verbose:
            print(f"  wrote {path.name} ({path.stat().st_size/1e9:.2f} GB, "
                  f"{time.time()-t0:.0f} s)")
        del out

    with open(geo_p, "w") as handle:
        json.dump(meta, handle, indent=2)
    if verbose:
        print(f"  wrote {geo_p.name}")
    return base_p, exc_p, meta


def ensure_focus0_dataset(data_dir, meta, prefix="focus0", chunk=1024, verbose=True):
    """The zero-defocus twin of the 00 block, at the 00 block's own scan settings."""
    return ensure_dataset(data_dir, meta, {"defocus_C10_ang": 0.0}, prefix=prefix,
                          chunk=chunk, verbose=verbose)


# ---------------------------------------------------------------------------
# Shot noise.
#
# The generator is deliberately NOT seeded by default. A fixed seed means every run
# of the pipeline sees the same single noise realisation, and a metric read off one
# realisation can look consistently good when it was only ever sampled once. Leaving
# it unseeded means each run is an independent draw, so a result that survives a few
# runs has actually survived something.
#
# `seed` is still accepted for the cases that need to pin a draw -- reproducing a
# specific figure, or giving two methods byte-identical input for a controlled
# comparison -- but nothing in the pipeline passes it.
# ---------------------------------------------------------------------------

_PIN_CACHE = {}


def _pinned_like(shape):
    """A reusable page-locked host buffer. Pinning 5.9 GB costs ~1.7 s, so it is
    allocated once per shape and reused by every later call."""
    key = tuple(shape)
    buf = _PIN_CACHE.get(key)
    if buf is None:
        buf = torch.empty(key, dtype=torch.float32).pin_memory()
        _PIN_CACHE[key] = buf
    return buf


def apply_poisson_dose(cube, dose_e_per_ang2, step_ang, seed=None, device="auto",
                       return_torch=False, rows_per_chunk=16):
    """Normalise each pattern to sum 1, then Poisson-sample at the given dose.

    `dose_e_per_ang2=None` returns the noiseless cube. `seed=None` (the default)
    draws from OS entropy, so repeated runs are independent realisations.

    The draw itself is trivial -- on the GPU a full 1.47e9-sample cube takes 0.02 s
    -- so essentially all the cost is moving data. Measured on one 5.9 GB cube:

        CPU (np.random.Generator.poisson)                21.6 s
        GPU, pageable host memory                         3.0 s
        GPU, page-locked output buffer (the default)      0.7 s
        GPU, result left on the device (return_torch)     0.02 s

    NumPy's poisson also returns int64, i.e. 11.8 GB for counts that never exceed a
    few hundred; the GPU path stays in float32 throughout and never allocates it.

    device: "auto" uses CUDA when present, else CPU. Pass "cpu" to force the
        reference implementation.
    return_torch: leave the result on the device. The consumers here (DirectPtychography,
        torch.fft) upload it again anyway, so the round trip is usually pure waste.
    rows_per_chunk: scan rows resident on the GPU at once; 16 keeps the peak near
        1.5 GB regardless of how large the cube is.
    """
    use_gpu = torch.cuda.is_available() if device == "auto" else (device != "cpu")

    if not use_gpu:
        out = np.asarray(cube, dtype=np.float32).copy()
        out /= out.sum(axis=(-2, -1), keepdims=True)
        if dose_e_per_ang2 is None:
            return out
        rng = np.random.default_rng(seed)
        out = rng.poisson(out * dose_e_per_ang2 * step_ang ** 2).astype(np.float32)
        total = out.sum(axis=(-2, -1), keepdims=True)
        out /= np.where(total > 0, total, 1.0)
        return out

    gen = None
    if seed is not None:
        gen = torch.Generator(device="cuda").manual_seed(int(seed))

    src = cube if isinstance(cube, torch.Tensor) else torch.from_numpy(
        np.ascontiguousarray(cube, dtype=np.float32))
    n_rows = src.shape[0]
    lam_scale = None if dose_e_per_ang2 is None else dose_e_per_ang2 * step_ang ** 2

    dst = (torch.empty(tuple(src.shape), dtype=torch.float32, device="cuda")
           if return_torch else _pinned_like(src.shape))

    for lo in range(0, n_rows, rows_per_chunk):
        hi = min(lo + rows_per_chunk, n_rows)
        g = src[lo:hi].to("cuda", non_blocking=True).float()
        g /= g.sum(dim=(-2, -1), keepdim=True)
        if lam_scale is not None:
            g = torch.poisson(g * lam_scale, generator=gen)
            total = g.sum(dim=(-2, -1), keepdim=True)
            g /= torch.where(total > 0, total, torch.ones_like(total))
        dst[lo:hi].copy_(g, non_blocking=True)
    torch.cuda.synchronize()

    if return_torch:
        return dst
    # the pinned buffer is reused by the next call, so hand back an owned copy
    return dst.numpy().copy()



# ---------------------------------------------------------------------------
# Plain ptychography: ONE dataset in, ONE phase image out.
#
# There is deliberately no "reconstruct and subtract" wrapper. That method is not
# a reconstruction algorithm, it is an experiment design: take a light-off scan,
# take a light-on scan, reconstruct each on its own and difference the results.
# The two scans are sequential and therefore land on grids that are not registered
# to one another, which is the entire point of studying it. Hiding that inside a
# wrapper that silently de-interlaced a single merged cube misrepresented it as a
# single measurement; notebook 01 now builds the two scans explicitly and calls
# this function twice.
# ---------------------------------------------------------------------------

def ptychography(
    cube,
    scan_params,
    num_iters=60,
    tv_weight=0.0,
    obj_padding=None,
    obj_lr=1e-3,
    probe_lr=1e-4,
    batch_size=128,
    name="scan",
    **kwargs,
):
    """Ordinary iterative ptychography on a single 4D-STEM datacube.

    `scan_params["scan_step"]` is the REAL step of this scan, in Angstrom -- there is
    no interlacing here and so no factor of two, unlike the two-state wrappers below.
    """
    step = scan_params["scan_step"]
    if obj_padding is None:
        obj_padding = _pad_px(step)

    dset = Dataset4dstem.from_array(
        cube, name=name, sampling=(step, step, 1, 1),
        units=("A", "A", "pixels", "pixels"),
    )
    ptycho = ptycho_wrappers.ptycho_wrapper(
        dset,
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
    # obj_cropped drops the padding border (no probe ever visits it) and zero-means
    # the arbitrary global phase gauge, both of which matter before any comparison
    # between two independently reconstructed phases.
    phase = np.angle(ptycho.obj_cropped[0, ...])
    return dict(phase=phase, ptycho=ptycho,
                pixel_size_ang=float(np.atleast_1d(ptycho.sampling)[0]))


def reconstruct_and_subtract(
    cube_off,
    cube_on,
    scan_params,
    num_iters=60,
    tv_weight=0.0,
    obj_padding=None,
    obj_lr=1e-3,
    probe_lr=1e-4,
    batch_size=128,
    **kwargs,
):
    """Reconstruct two SEQUENTIAL scans independently, then subtract the phases.

    Takes the two datacubes, not one interlaced cube. An earlier version of this
    function de-interlaced a single checkerboard scan into two half-resolution
    channels, which misrepresented the method: the two halves of one interlaced
    scan share a grid by construction, whereas two sequential scans do not, and
    the mismatch between them is the whole reason this method is worth studying.
    Notebook 01 builds the two scans at equal total dose and passes them here.

    `scan_params["scan_step"]` is the real step of each scan -- no interlacing, so
    no factor of two.
    """
    common = dict(num_iters=num_iters, tv_weight=tv_weight, obj_padding=obj_padding,
                  obj_lr=obj_lr, probe_lr=probe_lr, batch_size=batch_size)
    res_off = ptychography(cube_off, scan_params, name="off", **common)

    # The two reconstructions do not fit on one GPU at full scan density. Each
    # Ptychography holds its patch-index table, which is
    # num_positions x roi_shape^2 int32 -- 4.9 GB at 33k positions and a 192^2 ROI --
    # on top of its own copy of the data, so the second preprocess() ran out of
    # memory on a 32 GB card while the first was still resident. The finished
    # reconstruction is parked on the CPU; it is still returned intact, just not
    # taking up VRAM it no longer needs.
    res_off["ptycho"].to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    res_on = ptychography(cube_on, scan_params, name="on", **common)

    # Two independent optimizations each fit their own probe-circle calibration, so
    # their real-space pixel sizes -- and thus obj_shape_crop -- can differ by a
    # pixel or two even from identically shaped input. Match shapes, then gauge.
    phase_off, phase_on = _match_shape_2d(res_off["phase"], res_on["phase"])
    phase_off = phase_off - phase_off.mean()
    phase_on = phase_on - phase_on.mean()

    return dict(
        phase_off=phase_off,
        phase_on=phase_on,
        delta_phi=phase_on - phase_off,
        pixel_size_ang=res_off["pixel_size_ang"],
        ptycho_off=res_off["ptycho"],
        ptycho_on=res_on["ptycho"],
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
    channel_diff_l2_grad=0.0,
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
            # squared-L2 on the differential gradient; TV is edge-preserving and the
            # excitation is a smooth bump, so this is the better-matched prior
            "channel_diff_l2_grad": channel_diff_l2_grad,
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
        pixel_size_ang=float(np.atleast_1d(ptycho.sampling)[0]),
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
        pixel_size_ang=float(np.atleast_1d(ptycho.sampling)[0]),
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

def _reconstruct_complex_ssb(ptycho, vbf_fourier, force_dc, batch_size=None):
    """Mean over bright-field pixels of the SSB-deconvolved, inverse-transformed stack.

    `batch_size` is the number of bright-field pixels held on the device at once. The
    whole stack in one go is (num_bf, nQ, nQ) complex64 three times over -- operator,
    fourier_factor and the ifft -- which at a 300^2 scan with a 30 mrad aperture is
    ~9.5k bright-field pixels and ~27 GB of VRAM. Accumulating the sum in batches and
    dividing once at the end gives the identical answer at flat memory. None picks a
    batch that keeps each tensor near 1.5 GB.
    """
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

    if batch_size is None:
        batch_size = max(1, int(2.0e8 // max(qxa.numel(), 1)))

    total = torch.zeros(tuple(qxa.shape), dtype=torch.complex64, device=ptycho.device)
    for lo in range(0, ptycho.num_bf, batch_size):
        batch_idx = torch.arange(lo, min(lo + batch_size, ptycho.num_bf),
                                 device=ptycho.device)
        operator = ptycho._compute_gamma_operator(
            kxa, kya, qxa, qya, ptycho.aberration_coefs, cmplx_probe, batch_idx,
            asymmetric_version=True, normalize=True,
        )
        fourier_factor = vbf_fourier[batch_idx] * operator
        if force_dc:
            fourier_factor[..., 0, 0] = ptycho._dc_per_image
        total += torch.fft.ifft2(fourier_factor).sum(dim=0)
        del operator, fourier_factor

    return total / ptycho.num_bf


def _direct_excitation_phase(ptycho, batch_size=None):
    ptycho._preprocess()
    vbf_baseband = ptycho._vbf_fourier.clone()
    vbf_carrier = torch.fft.fftshift(ptycho._vbf_fourier, dim=(-1, -2))

    Obar = _reconstruct_complex_ssb(ptycho, vbf_baseband, force_dc=True,
                                    batch_size=batch_size)
    dO = _reconstruct_complex_ssb(ptycho, vbf_carrier, force_dc=False,
                                  batch_size=batch_size)

    delta_phi = 2.0 * torch.arctan((dO / Obar).imag)
    return delta_phi, Obar, dO


def classical_ssb(cube, scan_params, **kwargs):
    """Ordinary full-object SSB: quanTEM's own deconvolution, no demodulation.

    This is the textbook reconstruction -- one object phase out of one scan. It is
    quanTEM's `DirectPtychography.reconstruct(deconvolution_kernel="ssb")`, not a
    reimplementation. Use it on a single-state cube (base or excited); on an
    interlaced cube it returns the mean of the two states, which is why the
    excitation needs direct_ssb instead.
    """
    step = scan_params["scan_step"] / 2.0
    _, _, probe_R = fit_probe_circle(cube.mean(axis=(0, 1)), show=False)
    mrad_per_px = scan_params["probe_semiangle"] / probe_R
    dset = Dataset4d.from_array(
        cube, sampling=(step, step, mrad_per_px, mrad_per_px),
        units=("A", "A", "mrad", "mrad"))
    ptycho = DirectPtychography.from_dataset4d(
        dset, energy=scan_params["probe_energy"],
        semiangle_cutoff=scan_params["probe_semiangle"],
        aberration_coefs={"defocus": scan_params["probe_defocus"]},
        verbose=False, **kwargs
    ).reconstruct(deconvolution_kernel="ssb")
    obj = ptycho.obj
    obj = obj.detach().cpu().numpy() if hasattr(obj, "detach") else np.asarray(obj)
    phase = np.angle(obj) if np.iscomplexobj(obj) else obj
    return dict(phase=phase, ptycho=ptycho,
                pixel_size_ang=float(np.atleast_1d(ptycho.scan_sampling)[0]))


def direct_ssb(merged_scan, scan_mask, scan_params, batch_size=None, **kwargs):
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

    delta_phi, Obar, dO = _direct_excitation_phase(ptycho, batch_size=batch_size)

    return dict(
        phase_off=None,
        phase_on=None,
        delta_phi=delta_phi.detach().cpu().numpy(),
        # DirectPtychography carries TWO samplings: `sampling` is the detector-side
        # real-space pixel (lambda/2theta_max), while the SSB object is reconstructed
        # on the SCAN grid. Taking `sampling` here put the field of view 14% wrong.
        pixel_size_ang=float(np.atleast_1d(ptycho.scan_sampling)[0]),
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

    # These three tensors are the memory ceiling of the whole method: at 33k scan
    # positions, half of them on-mode, each is 16,562 x 192^2, i.e. 4.9 GB for a
    # complex64 and 2.4 GB for an int32. Accumulating them in lists and calling
    # torch.cat doubles that at the moment of the concatenation -- the list and the
    # result are both resident -- which is what ran a 32 GB card out of memory.
    # Preallocating and filling in place removes the duplicate entirely.
    n_on = int(len(on_batch))
    psi_off = Psi_off = patch_indices_on = None
    written = 0
    chunk = 2000
    with torch.no_grad():
        obj_off = ptycho.obj_model.obj[0, ...]  # channel 0 (off), fixed for this loop
        for start in range(0, n_on, chunk):
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
            if psi_off is None:
                psi_off = torch.empty((n_on, *psi.shape[1:]), dtype=psi.dtype, device=device)
                Psi_off = torch.empty((n_on, *Psi.shape[1:]), dtype=Psi.dtype, device=device)
                patch_indices_on = torch.empty(
                    (n_on, *patch_indices.shape[1:]), dtype=patch_indices.dtype, device=device
                )
            k = psi.shape[0]
            psi_off[written : written + k] = psi
            Psi_off[written : written + k] = Psi
            patch_indices_on[written : written + k] = patch_indices
            written += k
            del psi, Psi, patch_indices, obj_patches, shifted_probes, overlap
    assert written == n_on, f"filled {written} of {n_on} on-mode positions"

    # raw, single-shot on-mode DPs at their true native positions -- never averaged
    on_batch_np = on_batch.detach().cpu().numpy()
    I_on_meas = torch.from_numpy(
        merged_arr.reshape(-1, ndet, ndet)[on_batch_np].astype(np.float32)
    ).to(device)

    # Normalise in place. Written as `a = b / c` each of these allocates a second
    # copy of a multi-GB tensor while the first is still alive, which on this problem
    # is another ~10 GB of peak for no reason.
    I_off_n = Psi_off.abs() ** 2
    norm_const = I_off_n.mean()
    I_off_n.div_(norm_const)
    I_on_n = I_on_meas.div_(norm_const)
    Psi_off_n = Psi_off.div_(torch.sqrt(norm_const))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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
        pixel_size_ang=float(np.atleast_1d(ptycho.sampling)[0]),
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
    "ptychography": ptychography,
    "reconstruct_and_subtract": reconstruct_and_subtract,
    "co_reconstruction_regularized": co_reconstruction_regularized,
    "direct_ssb": direct_ssb,
    "linearized_model": linearized_model,
}


# ---------------------------------------------------------------------------
# Method 3b: streaming, Gamma-matched SSB.
#
# Why this exists alongside direct_ssb: at the 300^2 / 0.10 A geometry a single
# datacube is 13.3 GB, and the path through DirectPtychography holds four copies of
# it at once (the memory-mapped pages, the interlaced output, apply_poisson_dose's
# page-locked staging buffer and its output copy) before quanTEM adds a shifted
# tensor and a (num_bf, nQ, nQ) corrected stack. Measured peak: 62 GB of RSS just to
# produce one dosed cube, which OOM-kills a 122 GB machine once the reconstruction
# starts. Nothing here ever holds a whole cube: the virtual bright-field stack is
# built in scan-row blocks straight off the memory-mapped noiseless files, and the
# stack itself is num_bf * nQ^2 floats -- 3.4 GB rather than 13.3.
#
# The estimator is also the corrected one. quanTEM normalises Gamma to unit modulus
# and takes a flat mean over all bright-field pixels, which makes the transfer
# function f(Q) = (trotter area)/(BF disc area): a band-pass that peaks at 0.80 near
# Q = 0.55 1/A and falls to 0.07 at 0.03, i.e. it suppresses exactly the band a smooth
# excitation lives in. `matched=True` solves G = i*phi~*Gamma in least squares instead.
# Set matched=False to reproduce quanTEM's SSB bit for bit.
#
# Notebook 03.4 derives all of this and carries a readable inline copy; this is the
# same algorithm wired for memory. Keep the two in step.
# ---------------------------------------------------------------------------


def mean_diffraction_pattern(source, rows=16):
    """Mean diffraction pattern of a (ny, nx, kx, ky) cube, read in scan-row blocks.

    `source` may be memory-mapped; it is never fully materialised.
    """
    acc = np.zeros(source.shape[-2:], dtype=np.float64)
    for lo in range(0, source.shape[0], rows):
        hi = min(lo + rows, source.shape[0])
        blk = np.asarray(source[lo:hi], dtype=np.float32)
        blk = blk / blk.sum(axis=(-2, -1), keepdims=True)
        acc += blk.sum(axis=(0, 1))
        del blk
    return acc / (source.shape[0] * source.shape[1])


def bright_field_geometry(mean_dp, semiangle_mrad, wavelength_ang, threshold=0.5):
    """Bright-field mask, disc centre and the detector k-grid, measured from mean_dp.

    Returns (bf_mask, (c_i, c_j), mrad_per_px, dk, k_cut, KX, KY) with KX/KY in 1/A.
    The disc radius comes from the mask's area, and the radius IS the semiangle by
    definition, which is what sets the scale.
    """
    bf_mask = mean_dp > threshold * mean_dp.max()
    num_bf = int(bf_mask.sum())
    r_px = np.sqrt(num_bf / np.pi)
    idx_i, idx_j = np.indices(mean_dp.shape)
    w = np.where(bf_mask, mean_dp, 0.0)
    c_i = float((idx_i * w).sum() / w.sum())
    c_j = float((idx_j * w).sum() / w.sum())
    mrad_per_px = semiangle_mrad / r_px
    dk = mrad_per_px * 1e-3 / wavelength_ang
    k_cut = semiangle_mrad * 1e-3 / wavelength_ang
    KX, KY = np.meshgrid((np.arange(mean_dp.shape[0]) - c_i) * dk,
                         (np.arange(mean_dp.shape[1]) - c_j) * dk, indexing="ij")
    return bf_mask, (c_i, c_j), mrad_per_px, dk, k_cut, KX, KY


def build_vbf_stack(src_off, src_on, bf_mask, dose_e_per_ang2, step_ang, rows=16,
                    seed=None):
    """(ny, nx, num_bf) float32 virtual bright-field stack, built in scan-row blocks.

    src_on=None gives a single-state stack from src_off alone; otherwise the two are
    checkerboard-interlaced with this module's convention ((i + j) even -> ON) and the
    dose is applied to the interlaced result. Each scan position is an independent
    Poisson draw, so that is identical to dosing each state first.

    Peak memory is one row block plus the output: ~5 GB at 300^2 x 192^2 with rows=16,
    against 62 GB for the materialise-then-reconstruct path.
    """
    n_rows, n_cols = src_off.shape[:2]
    num_bf = int(bf_mask.sum())
    out = np.empty((n_rows, n_cols, num_bf), dtype=np.float32)
    cols = np.arange(n_cols)
    for lo in range(0, n_rows, rows):
        hi = min(lo + rows, n_rows)
        # np.array, not np.asarray: a float32 memory-mapped slice comes back as a
        # read-only view and the interlacing below writes into this block.
        blk = np.array(src_off[lo:hi], dtype=np.float32)
        if src_on is not None:
            on_blk = np.asarray(src_on[lo:hi], dtype=np.float32)
            for r in range(lo, hi):
                blk[r - lo, (r + cols) % 2 == 0] = on_blk[r - lo, (r + cols) % 2 == 0]
            del on_blk
        blk = apply_poisson_dose(blk, dose_e_per_ang2, step_ang,
                                 seed=None if seed is None else int(seed) + lo)
        out[lo:hi] = blk[:, :, bf_mask]
        del blk
    return out


def ssb_accumulate(vbf, bf_kx, bf_ky, scan_step_ang, k_cut, dk, chunk=256,
                   device=None, carrier=True):
    """The three sums over bright-field pixels that SSB needs, on the GPU.

    Returns a dict of (nQ, nQ) tensors: `num_struct` and `flat_struct` for the
    structural channel, the same with `_carrier` for the demodulated one, and `den`
    = sum_k |Gamma|^2, which doubles as the transfer function.

    Gamma(k, Q) = A(k) [ A(k - Q) - A(k + Q) ] for chi = 0, so it is real: +1 on one
    overlap crescent, -1 on the other, exactly 0 in the triple overlap and outside.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    n_scan = vbf.shape[0]
    num_bf = vbf.shape[-1]

    G = torch.fft.fft2(torch.as_tensor(vbf, device=device).permute(2, 0, 1).contiguous())
    G /= G[..., 0, 0].real.clamp(min=1e-20).view(-1, 1, 1) / (n_scan * n_scan)
    G[..., 0, 0] = 0

    f = np.fft.fftfreq(n_scan, d=scan_step_ang)
    QX = torch.as_tensor(np.broadcast_to(f[:, None], (n_scan, n_scan)).copy(),
                         dtype=torch.float32, device=device)
    QY = torch.as_tensor(np.broadcast_to(f[None, :], (n_scan, n_scan)).copy(),
                         dtype=torch.float32, device=device)
    kx_all = torch.as_tensor(bf_kx, dtype=torch.float32, device=device)
    ky_all = torch.as_tensor(bf_ky, dtype=torch.float32, device=device)

    def aperture(kx, ky):
        return torch.clip((k_cut - torch.sqrt(kx * kx + ky * ky)) / dk + 0.5, 0.0, 1.0)

    z = lambda: torch.zeros((n_scan, n_scan), dtype=torch.complex64, device=device)
    acc = dict(num_struct=z(), flat_struct=z(),
               num_carrier=z() if carrier else None,
               flat_carrier=z() if carrier else None,
               den=torch.zeros((n_scan, n_scan), dtype=torch.float32, device=device))

    for lo in range(0, num_bf, chunk):
        hi = min(lo + chunk, num_bf)
        kx = kx_all[lo:hi].view(-1, 1, 1)
        ky = ky_all[lo:hi].view(-1, 1, 1)
        gam = aperture(kx, ky) * (aperture(kx - QX, ky - QY) - aperture(kx + QX, ky + QY))
        flat = gam / gam.abs().clamp(min=1e-8)
        g_s = G[lo:hi]
        acc["num_struct"] += (gam * g_s).sum(0)
        acc["flat_struct"] += (flat * g_s).sum(0)
        acc["den"] += gam.square().sum(0)
        if carrier:
            g_c = torch.roll(g_s, (n_scan // 2, n_scan // 2), dims=(-2, -1))
            acc["num_carrier"] += (gam * g_c).sum(0)
            acc["flat_carrier"] += (flat * g_c).sum(0)
            del g_c
        del gam, flat, g_s
    del G
    if device != "cpu":
        torch.cuda.empty_cache()
    acc["num_bf"] = num_bf
    acc["n_scan"] = n_scan
    acc["QR"] = torch.sqrt(QX * QX + QY * QY)
    return acc


def ssb_invert(acc, matched=True, lam_rel=1e-5, q_max=None, channel="struct"):
    """Divide, band-limit, restore the DC, inverse transform. Returns a complex object.

    matched=False reproduces quanTEM's estimator: the phase-only kernel averaged flat
    over every bright-field pixel. `channel` is "struct" or "carrier".
    """
    n_scan = acc["n_scan"]
    key = ("num_" if matched else "flat_") + channel
    if matched:
        out = acc[key] / (acc["den"] + lam_rel * float(acc["den"].max()))
    else:
        out = acc[key] / acc["num_bf"]
    if q_max is not None:
        out = out * (acc["QR"] <= q_max).to(torch.complex64)
    out = out.clone()
    # Gamma(k, 0) = 0, so Q = 0 is never measured: the object's mean transmission is
    # pinned to 1, and the mean of delta phi is simply not recoverable at any dose.
    out[0, 0] = float(n_scan * n_scan) if channel == "struct" else 0.0
    return torch.fft.ifft2(out)


def excitation_phase(acc, matched=True, lam_rel=1e-5, q_max=None, readout="arctan",
                     floor_rel=0.1):
    """delta phi and the mean object, from one accumulation."""
    Obar = ssb_invert(acc, matched, lam_rel, q_max, channel="struct")
    dO = ssb_invert(acc, matched, lam_rel, q_max, channel="carrier")
    if readout == "arctan":
        d = 2.0 * torch.arctan((dO / Obar).imag)
    elif readout == "stable":
        a2 = Obar.abs().square()
        d = 2.0 * (dO * Obar.conj()).imag / a2.clamp(min=floor_rel * float(a2.median()))
    elif readout == "naive":
        d = 2.0 * dO.imag
    else:
        raise ValueError(f"unknown readout {readout!r}")
    return d.detach().cpu().numpy(), Obar.detach().cpu().numpy()


def build_dosed_cube(src_off, src_on, dose_e_per_ang2, step_ang, out=None, rows=16,
                     seed=None):
    """A full (ny, nx, kx, ky) dosed cube, assembled in scan-row blocks.

    For the analyses that need whole diffraction patterns rather than just the
    bright-field disc, so build_vbf_stack's trick does not apply. Pass `out` to reuse
    one preallocated buffer across calls: that is the difference between one 13.3 GB
    array resident and three.

    src_on=None gives the single-state cube. The dose must be applied to whole
    patterns -- apply_poisson_dose normalises each pattern over the full detector --
    which is why this cannot be streamed over the detector axis instead.
    """
    n_rows, n_cols = src_off.shape[:2]
    if out is None:
        out = np.empty(src_off.shape, dtype=np.float32)
    cols = np.arange(n_cols)
    for lo in range(0, n_rows, rows):
        hi = min(lo + rows, n_rows)
        blk = np.array(src_off[lo:hi], dtype=np.float32)
        if src_on is not None:
            on_blk = np.asarray(src_on[lo:hi], dtype=np.float32)
            for r in range(lo, hi):
                blk[r - lo, (r + cols) % 2 == 0] = on_blk[r - lo, (r + cols) % 2 == 0]
            del on_blk
        out[lo:hi] = apply_poisson_dose(blk, dose_e_per_ang2, step_ang,
                                        seed=None if seed is None else int(seed) + lo)
        del blk
    return out
