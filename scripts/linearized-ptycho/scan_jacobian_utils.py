"""Shared machinery for notebooks 02 and 02.5.

Nothing here draws a figure or decides anything physical.  It holds the two things
both notebooks need identically - the experiment (object, probe, scan, exit waves)
and the spectral reductions applied to a set of modes - so that the two notebooks
cannot drift apart in their setup, and so that the notebooks themselves contain the
figures and the argument rather than the bookkeeping.

The one plotting helper, ``draw_atoms_on``, is here because it is used by several
figures in both notebooks; every figure-specific plotting call lives next to its
figure in the notebook.
"""

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy import ndimage
from scipy.spatial import cKDTree

import linearized_jacobian as lj

__all__ = [
    "ExperimentConfig", "build_experiment", "SpectralGrid", "top_k_eigenpairs",
    "relative_gap", "degeneracy_counts", "half_power_band", "draw_atoms_on",
    "pick_device", "spectrum_cache_path",
]


def spectrum_cache_path(cache_dir, experiment, config, solver_settings,
                        prefix="dense_full_field_spectrum"):
    """Where a spectrum for exactly these inputs is stored.

    The key covers everything the result depends on - the base phase, the scan, the
    probe, the detector and the solver settings - so changing any of them misses the
    cache and rebuilds rather than silently returning a stale spectrum.
    """
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(experiment.phase_base, dtype=np.float32).tobytes())
    digest.update(np.ascontiguousarray(experiment.scan_positions_flat_xy_ang,
                                       dtype=np.float64).tobytes())
    digest.update(repr((
        config.acc_voltage_ev, config.probe_semiangle_mrad, config.probe_defocus_ang,
        config.slice_thickness_ang, tuple(experiment.object_sampling_ang),
        experiment.detector_shape,
    ) + tuple(solver_settings)).encode())
    return Path(cache_dir) / f"{prefix}_{digest.hexdigest()[:16]}.npz"


def pick_device(min_free_gb, verbose=True):
    """``'cuda'`` if that much device memory is actually free right now, else ``'cpu'``.

    This box shares its GPU with whatever else is running on it, so "is CUDA available"
    is the wrong question - a card with 300 MB free is available and useless.  Ask for
    what the next step needs and fall back rather than failing half way through.
    """
    if not torch.cuda.is_available():
        if verbose:
            print("device: cpu   (no CUDA device)")
        return "cpu"
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    free_gb, total_gb = free_bytes / 1e9, total_bytes / 1e9
    if free_gb >= min_free_gb:
        if verbose:
            print(f"device: cuda  ({free_gb:.1f} GB free of {total_gb:.1f} GB,"
                  f" {min_free_gb:.1f} GB asked for)")
        return "cuda"
    if verbose:
        print(f"device: cpu   (only {free_gb:.1f} GB free of {total_gb:.1f} GB on the GPU,"
              f" {min_free_gb:.1f} GB asked for)")
    return "cpu"


# --------------------------------------------------------------------------------
# the experiment
# --------------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """Everything that defines the operator, and nothing that defines a figure."""

    phase_cache_path: str
    phase_cache_key_base: str = "base"

    acc_voltage_ev: float = 60e3
    probe_semiangle_mrad: float = 30.0
    probe_defocus_ang: float = 0.0
    slice_thickness_ang: float = 1.0          # bookkeeping slice for a projected phase object

    hbn_lattice_constant_ang: float = 2.504   # annotation and the atom check only
    atom_peak_footprint_px: int = 11          # below the B-N bond, so neighbours stay separate
    atom_peak_threshold_fraction: float = 0.15

    scan_step_size_ang: float = 0.5           # requested; GridScan rounds it to fit the extent
    scan_region_start: tuple = (7.0, 7.0)
    scan_region_end: tuple = (17.0, 17.0)
    scan_endpoint: bool = False               # see the note in the notebook's config cell

    hbn_reflections: tuple = ((1, 0), (1, 1), (2, 0), (2, 1), (3, 0))
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")


def build_experiment(config):
    """The base object, the probe, the scan and the waves the Jacobian is built from.

    Returns a namespace.  Every quantity a notebook plots or prints is an attribute of
    it; nothing is returned through globals.  The exit and camera waves are complex128
    so that an accumulation over scan positions is not the limiting factor - the inputs
    are float32, so this does not make the model more accurate, only the sum.

    One transform convention is used for both factors of ``A_R``.  The centred
    ``fftshift(fft2(ifftshift(.)))`` differs from the corner-origin ``fftshift(fft2(.))``
    by a phase ramp across the detector, and that ramp cancels inside ``A_R`` only if
    the same convention is used for ``Psi_0`` and for ``F{psi_0 h}``.  Mixing the two
    silently produces a different operator, so it is defined once, here.
    """
    import pickle

    with open(config.phase_cache_path, "rb") as handle:
        phase_cache = pickle.load(handle)

    phase_base = np.asarray(phase_cache[config.phase_cache_key_base], dtype=np.float32)
    object_x_ang = np.asarray(phase_cache["x"], dtype=np.float32); object_x_ang -= object_x_ang[0]
    object_y_ang = np.asarray(phase_cache["y"], dtype=np.float32); object_y_ang -= object_y_ang[0]
    object_sampling_ang = (float(object_y_ang[1] - object_y_ang[0]),
                           float(object_x_ang[1] - object_x_ang[0]))
    object_base = np.exp(1j * phase_base).astype(np.complex64)

    potential = lj.make_phase_potential(
        phase_base, config.acc_voltage_ev, config.slice_thickness_ang, object_sampling_ang
    )
    probe = lj.make_probe(config.acc_voltage_ev, config.probe_semiangle_mrad,
                          config.probe_defocus_ang, potential)

    n_object_y, n_object_x = phase_base.shape
    n_object_pixels = n_object_y * n_object_x
    field_of_view_ang = tuple(float(v) for v in potential.extent)
    detector_shape = lj.valid_detector_shape(probe)
    n_detector_y, n_detector_x = detector_shape
    n_detector_pixels = n_detector_y * n_detector_x
    detector_sampling_inv_ang = (1.0 / field_of_view_ang[1], 1.0 / field_of_view_ang[0])
    electron_wavelength_ang = float(probe.wavelength)

    scan_positions_xy_ang = lj.grid_scan_positions_ang(
        potential, config.scan_step_size_ang, endpoint=config.scan_endpoint,
        start=config.scan_region_start, end=config.scan_region_end,
    )
    scan_shape = scan_positions_xy_ang.shape[:2]
    scan_positions_flat_xy_ang = scan_positions_xy_ang.reshape(-1, 2).astype(np.float64)
    n_scan_positions = len(scan_positions_flat_xy_ang)
    realised_scan_step_ang = float(np.diff(np.unique(scan_positions_flat_xy_ang[:, 0]))[0])

    # Two positions that coincide modulo the periodic cell are one measurement counted twice.
    scan_positions_wrapped_ang = np.round(
        np.mod(np.round(scan_positions_flat_xy_ang / np.array(field_of_view_ang), 6), 1.0)
        * np.array(field_of_view_ang), 4
    )
    n_distinct_scan_positions = len(np.unique(scan_positions_wrapped_ang, axis=0))

    device = config.device
    probe_by_scan = torch.as_tensor(
        np.asarray(probe.build(scan=scan_positions_flat_xy_ang.tolist()).compute().array
                   ).astype(np.complex64),
        device=device,
    )
    object_base_torch = torch.as_tensor(object_base, device=device)
    exit_wave_by_scan = (probe_by_scan * object_base_torch).to(torch.complex128)

    detector_row_start = (n_object_y - n_detector_y) // 2
    detector_col_start = (n_object_x - n_detector_x) // 2

    def centered_dft_torch(array):
        return torch.fft.fftshift(
            torch.fft.fft2(torch.fft.ifftshift(array, dim=(-2, -1))), dim=(-2, -1)
        )

    def crop_to_detector(array):
        return array[..., detector_row_start:detector_row_start + n_detector_y,
                     detector_col_start:detector_col_start + n_detector_x]

    camera_wave_by_scan = crop_to_detector(centered_dft_torch(exit_wave_by_scan))

    # --- atomic columns, detected from the projected phase ------------------------
    # The phase cache stores only the image, no coordinates.  A column is a local
    # maximum of the projected potential, so a maximum filter over a footprint
    # narrower than the B-N bond separates neighbours.  The median spacing is checked
    # against the hBN bond length by the caller: if it is not the bond length, the
    # peaks are not the lattice and the overlay would be decoration, not information.
    local_maximum = ndimage.maximum_filter(phase_base, size=config.atom_peak_footprint_px,
                                           mode="wrap")
    peak_threshold = phase_base.min() + config.atom_peak_threshold_fraction * np.ptp(phase_base)
    atom_rows, atom_cols = np.nonzero((phase_base == local_maximum) & (phase_base > peak_threshold))
    atom_x_ang = object_x_ang[atom_cols]
    atom_y_ang = object_y_ang[atom_rows]
    atom_xy_ang = np.c_[atom_x_ang, atom_y_ang]
    atom_nn_spacing_ang = float(np.median(cKDTree(atom_xy_ang).query(atom_xy_ang, k=2)[0][:, 1]))

    # --- probe width, measured rather than taken from the Airy formula -------------
    probe_centre_index = int(np.argmin(np.linalg.norm(
        scan_positions_flat_xy_ang - np.array(field_of_view_ang) / 2.0, axis=1
    )))
    probe_centre_intensity = (probe_by_scan[probe_centre_index].abs() ** 2).cpu().numpy()
    coverage = (probe_by_scan.abs() ** 2).sum(0).cpu().numpy()

    peak_row, peak_col = np.unravel_index(np.argmax(probe_centre_intensity),
                                          probe_centre_intensity.shape)
    row_index, col_index = np.mgrid[0:n_object_y, 0:n_object_x]
    radius_from_peak_ang = np.hypot((row_index - peak_row) * object_sampling_ang[0],
                                    (col_index - peak_col) * object_sampling_ang[1])
    radial_bin_width_ang = object_sampling_ang[0] / 2.0
    radial_bin_index = (radius_from_peak_ang / radial_bin_width_ang).astype(int).ravel()
    radial_bin_count = np.bincount(radial_bin_index)
    radial_bin_sum = np.bincount(radial_bin_index, weights=probe_centre_intensity.ravel())
    occupied = radial_bin_count > 0
    radial_profile = radial_bin_sum[occupied] / radial_bin_count[occupied]
    radial_bin_centre_ang = (
        (np.arange(len(radial_bin_count)) + 0.5) * radial_bin_width_ang)[occupied]
    half_maximum = radial_profile.max() / 2.0
    first_below = int(np.argmax(radial_profile < half_maximum))
    probe_fwhm_ang = 2.0 * float(np.interp(
        half_maximum,
        [radial_profile[first_below], radial_profile[first_below - 1]],
        [radial_bin_centre_ang[first_below], radial_bin_centre_ang[first_below - 1]],
    ))
    airy_fwhm_ang = 0.514 * electron_wavelength_ang / (config.probe_semiangle_mrad * 1e-3)

    scan_box_ang = [float(scan_positions_flat_xy_ang[:, 0].min() + object_x_ang[0]),
                    float(scan_positions_flat_xy_ang[:, 0].max() + object_x_ang[0]),
                    float(scan_positions_flat_xy_ang[:, 1].min() + object_y_ang[0]),
                    float(scan_positions_flat_xy_ang[:, 1].max() + object_y_ang[0])]

    # --- band limits ---------------------------------------------------------------
    aperture_radius_inv_ang = config.probe_semiangle_mrad * 1e-3 / electron_wavelength_ang
    detector_edge_inv_ang = (n_detector_y // 2) * detector_sampling_inv_ang[0]
    object_nyquist_inv_ang = 1.0 / (2.0 * object_sampling_ang[0])
    visibility_cutoff_inv_ang = detector_edge_inv_ang + aperture_radius_inv_ang

    # For a 2-D hexagonal lattice |G_hk| = (2/(sqrt(3) a)) sqrt(h^2 + hk + k^2): a property
    # of the specimen, not of the microscope.
    lattice_reflections_inv_ang = [
        (f"({h}{k})",
         2.0 / (np.sqrt(3.0) * config.hbn_lattice_constant_ang) * np.sqrt(h * h + h * k + k * k))
        for h, k in config.hbn_reflections
    ]

    return SimpleNamespace(
        config=config, device=device,
        phase_base=phase_base, object_base=object_base,
        object_x_ang=object_x_ang, object_y_ang=object_y_ang,
        object_sampling_ang=object_sampling_ang,
        object_extent_ang=[float(object_x_ang[0]), float(object_x_ang[-1]),
                           float(object_y_ang[0]), float(object_y_ang[-1])],
        n_object_y=n_object_y, n_object_x=n_object_x, n_object_pixels=n_object_pixels,
        potential=potential, probe=probe, field_of_view_ang=field_of_view_ang,
        detector_shape=detector_shape, n_detector_y=n_detector_y, n_detector_x=n_detector_x,
        n_detector_pixels=n_detector_pixels,
        detector_sampling_inv_ang=detector_sampling_inv_ang,
        detector_row_start=detector_row_start, detector_col_start=detector_col_start,
        electron_wavelength_ang=electron_wavelength_ang,
        scan_positions_flat_xy_ang=scan_positions_flat_xy_ang, scan_shape=scan_shape,
        n_scan_positions=n_scan_positions, n_distinct_scan_positions=n_distinct_scan_positions,
        realised_scan_step_ang=realised_scan_step_ang, scan_box_ang=scan_box_ang,
        probe_by_scan=probe_by_scan, exit_wave_by_scan=exit_wave_by_scan,
        camera_wave_by_scan=camera_wave_by_scan,
        centered_dft_torch=centered_dft_torch, crop_to_detector=crop_to_detector,
        atom_x_ang=atom_x_ang, atom_y_ang=atom_y_ang,
        atom_nn_spacing_ang=atom_nn_spacing_ang,
        hbn_bond_length_ang=config.hbn_lattice_constant_ang / np.sqrt(3.0),
        coverage=coverage, probe_centre_intensity=probe_centre_intensity,
        probe_peak_row=int(peak_row), probe_peak_col=int(peak_col),
        probe_fwhm_ang=probe_fwhm_ang, airy_fwhm_ang=airy_fwhm_ang,
        aperture_radius_inv_ang=aperture_radius_inv_ang,
        detector_edge_inv_ang=detector_edge_inv_ang,
        object_nyquist_inv_ang=object_nyquist_inv_ang,
        visibility_cutoff_inv_ang=visibility_cutoff_inv_ang,
        lattice_reflections_inv_ang=lattice_reflections_inv_ang,
        stacked_shape=(n_scan_positions * n_detector_pixels, n_object_pixels),
    )


def draw_atoms_on(ax, experiment, markersize=2.0, alpha=0.45, color="#808080"):
    """Overlay the detected atomic columns, faintly.

    Drawn after the image and with the axis limits restored, so adding them never
    changes what a panel shows.
    """
    x_limits, y_limits = ax.get_xlim(), ax.get_ylim()
    ax.plot(experiment.atom_x_ang, experiment.atom_y_ang, linestyle="none", marker="o",
            markersize=markersize, markerfacecolor=color, markeredgecolor="none",
            alpha=alpha, zorder=5)
    ax.set_xlim(x_limits); ax.set_ylim(y_limits)


# --------------------------------------------------------------------------------
# spectral reductions of a set of modes
# --------------------------------------------------------------------------------

class SpectralGrid:
    """The object-space frequency grid, and the reductions of a mode set onto it.

    A mode set may live on the whole object grid or on a window inside it; a window
    mode is embedded at ``origin`` and is zero outside, which is lossless in real
    space.  Every reduction here therefore lands on one shared frequency grid and is
    comparable across domains.
    """

    def __init__(self, experiment, mode_chunk=128):
        self.n_y = experiment.n_object_y
        self.n_x = experiment.n_object_x
        self.mode_chunk = mode_chunk

        self.axis_inv_ang = np.fft.fftshift(
            np.fft.fftfreq(self.n_y, d=experiment.object_sampling_ang[0]))
        self.radius_inv_ang = np.hypot(
            *np.meshgrid(self.axis_inv_ang, self.axis_inv_ang, indexing="ij"))
        self.extent = [self.axis_inv_ang[0], self.axis_inv_ang[-1],
                       self.axis_inv_ang[0], self.axis_inv_ang[-1]]

        bin_width = experiment.detector_sampling_inv_ang[0]
        edges = np.arange(0.0, experiment.object_nyquist_inv_ang + bin_width, bin_width)
        self.bin_edges_inv_ang = edges
        self.bin_centres_inv_ang = 0.5 * (edges[1:] + edges[:-1])
        index = np.digitize(self.radius_inv_ang.ravel(), edges) - 1
        self._index = index
        self._valid = (index >= 0) & (index < len(self.bin_centres_inv_ang))

    def embed(self, modes, origin):
        """Place window modes into the full object grid.  Lossless: zero outside."""
        if modes.shape[-2:] == (self.n_y, self.n_x):
            return modes
        out = np.zeros((len(modes), self.n_y, self.n_x), dtype=modes.dtype)
        row0, col0 = origin
        height, width = modes.shape[-2:]
        out[:, row0:row0 + height, col0:col0 + width] = modes
        return out

    def mode_mean_frequency(self, modes, origin=(0, 0)):
        """qbar_i = sum_q |v_i(q)|^2 |q| / sum_q |v_i(q)|^2, for every mode."""
        out = np.empty(len(modes), dtype=np.float64)
        for start in range(0, len(modes), self.mode_chunk):
            block = self.embed(modes[start:start + self.mode_chunk], origin)
            power = np.abs(np.fft.fftshift(np.fft.fft2(block), axes=(-2, -1))) ** 2
            power /= power.sum(axis=(-2, -1), keepdims=True)
            out[start:start + self.mode_chunk] = (power * self.radius_inv_ang).sum(axis=(-2, -1))
        return out

    def sensitivity_density(self, modes, sigma, origin=(0, 0)):
        """D(q) = sum_i sigma_i^2 |v_i(q)|^2 = ||A f_q||^2 for a unit plane wave f_q.

        The orthonormal transform is used so that <v_i, f_q> = v_i(q) and Parseval
        holds, which is what makes D the response to a unit-amplitude plane wave
        rather than a quantity scaled by the grid size.  Every term is positive, so
        truncating the mode set is a one-sided error, and sum_q D(q) = ||A||_F^2.
        """
        density = np.zeros((self.n_y, self.n_x), dtype=np.float64)
        for start in range(0, len(modes), self.mode_chunk):
            stop = min(start + self.mode_chunk, len(modes))
            block = self.embed(modes[start:stop], origin)
            magnitude = np.abs(np.fft.fftshift(np.fft.fft2(block, norm="ortho"), axes=(-2, -1)))
            weight = sigma[start:stop, None, None].astype(np.float64)
            density += ((weight ** 2) * magnitude ** 2).sum(axis=0)
        return density

    def mode_sum_amplitude(self, modes, origin=(0, 0), indices=None, sigma=None):
        """Unweighted S(q) = sum_i |v_i(q)|, or sigma-weighted if ``sigma`` is given.

        Used only by notebook 02.5, where the point is what the weighting changes.
        """
        indices = np.arange(len(modes)) if indices is None else np.asarray(indices)
        total = np.zeros((self.n_y, self.n_x), dtype=np.float64)
        for start in range(0, len(indices), self.mode_chunk):
            picked = indices[start:start + self.mode_chunk]
            block = self.embed(modes[picked], origin)
            magnitude = np.abs(np.fft.fftshift(np.fft.fft2(block, norm="ortho"), axes=(-2, -1)))
            if sigma is not None:
                magnitude = magnitude * (sigma[picked, None, None].astype(np.float64) / sigma[0])
            total += magnitude.sum(axis=0)
        return total

    def angular_integral(self, spectrum_2d):
        """Sum of a 2-D spectrum over each |q| annulus."""
        return np.bincount(self._index[self._valid],
                           weights=spectrum_2d.ravel()[self._valid],
                           minlength=len(self.bin_centres_inv_ang))


# --------------------------------------------------------------------------------
# small numerical utilities
# --------------------------------------------------------------------------------

def top_k_eigenpairs(gram, n_modes, n_oversample, n_iterations, seed):
    """Top eigenpairs of an explicit symmetric matrix by subspace iteration.

    Returns the modes largest-first, together with the relative drift of the Ritz
    values over the final iteration - the honest statement of which end has converged.
    """
    n = gram.shape[0]
    sketch = n_modes + n_oversample
    generator = torch.Generator(device=gram.device).manual_seed(seed)
    basis = torch.linalg.qr(torch.randn(n, sketch, device=gram.device,
                                        dtype=gram.dtype, generator=generator))[0]

    def ritz(basis):
        projection = basis.T @ (gram @ basis)
        projection = 0.5 * (projection + projection.T)
        return torch.linalg.eigh(projection)

    previous = None
    for iteration in range(n_iterations):
        basis = torch.linalg.qr(gram @ basis)[0]
        if iteration == n_iterations - 2:
            previous = ritz(basis)[0].flip(0)[:n_modes].clamp(min=0).sqrt()

    values, vectors = ritz(basis)
    sigma = values.flip(0)[:n_modes].clamp(min=0).sqrt()
    modes = (basis @ vectors.flip(1))[:, :n_modes].T
    drift = float(((sigma - previous).abs() / sigma[0]).max()) if previous is not None else np.nan
    return sigma, modes, drift


def relative_gap(sigma):
    """(sigma_i - sigma_{i+1}) / sigma_1, the local non-degeneracy of each mode."""
    gap = np.empty_like(sigma)
    gap[:-1] = (sigma[:-1] - sigma[1:]) / sigma[0]
    gap[-1] = np.nan
    return gap


def degeneracy_counts(sigma, tolerances=(0.001, 0.01, 0.05, 0.1)):
    """How many modes sit within a given relative distance of sigma_1."""
    return {tol: int((sigma > (1.0 - tol) * sigma[0]).sum()) for tol in tolerances}


def half_power_band(profile, bin_centres):
    """(peak position, low edge, high edge) of the range where a profile >= half its peak."""
    above = np.flatnonzero(profile >= 0.5 * profile.max())
    return (float(bin_centres[int(np.argmax(profile))]),
            float(bin_centres[above[0]]), float(bin_centres[above[-1]]))
