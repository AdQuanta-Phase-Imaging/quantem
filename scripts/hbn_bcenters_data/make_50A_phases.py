"""Wide-FOV (50 x 50 A) hBN blue-centre phase pair with a single, isolated excitation.

Source: bcenters_base_excited_phases_wide_fov.pkl, the +/-12 A pair built in
  dft_decomp/.../scripts/"21 - Phase for periodic structures.ipynb", cells 22-24,
  from the trans-defect hBN 6x6x2 spin-polarised GPAW calculation.

No DFT is re-run.  That notebook builds the phase by solving the periodic Poisson
equation on the DFT supercell and then tiling it over the imaging plane with
CrystalDefect.tile_unit_cell_image_fast, so the saved field is exactly periodic on
the 6x6 supercell lattice.  A larger field of view is therefore a lattice
translation of what is already on disk, not new physics -- verified below to
1.5e-4 of peak for the differential phase.

The base phase is extended periodically, which is what a real lattice does.  The
excitation is not: it is tiled and then multiplied by a raised cosine centred on the
dipole, so exactly one blue-red blob survives.
"""
import pickle
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates

# --- configuration -----------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent
SOURCE = DATA_DIR / "bcenters_base_excited_phases_wide_fov.pkl"
TARGET = DATA_DIR / "bcenters_base_excited_phases_50A_fov.pkl"

ALL_AXES_RANGE_ANG = 25.0      # ImagingPlane convention: x = linspace(-R, +R, n)
N_PIX_PER_AXIS = 521           # -> 0.09615 A/px, matching the source's 0.096
INTERPOLATION_ORDER = 3        # the source notebook tiled with order=1

TAPER_INNER_ANG = 6.0          # full weight inside this radius
TAPER_OUTER_ANG = 9.0          # zero beyond it; half-weight at 7.5 A, the measured minimum

A_HBN, N_CELLS = 2.506, 6      # hbn_defects.py: a = 2.506, unit_cell_counts = (6, 6, 2)
U = N_CELLS * np.array([A_HBN, 0.0])
V = N_CELLS * np.array([A_HBN / 2.0, A_HBN * np.sqrt(3) / 2.0])

# --- the source field --------------------------------------------------------
with open(SOURCE, "rb") as handle:
    cache = pickle.load(handle)
src_base = np.asarray(cache["base"], dtype=np.float64)
src_excited = np.asarray(cache["excited"], dtype=np.float64)
src_x = np.asarray(cache["x"], dtype=np.float64)
src_y = np.asarray(cache["y"], dtype=np.float64)
src_delta = src_excited - src_base
src_dx, src_dy = src_x[1] - src_x[0], src_y[1] - src_y[0]


def sample_source(image, px, py):
    """Bicubic sample of a source image. image[i, j] <-> src_x[i], src_y[j]."""
    coords = np.vstack([((px - src_x[0]) / src_dx).ravel(),
                        ((py - src_y[0]) / src_dy).ravel()])
    return map_coordinates(image, coords, order=INTERPOLATION_ORDER,
                           mode="nearest").reshape(np.shape(px))


def wrap_into_source(px, py):
    """Translate (px, py) by whole lattice vectors into the centred fundamental cell.

    With fractional coordinates reduced to [-1/2, 1/2), the cell spans at most
    |x| <= (|u_x| + |v_x|)/2 = 11.28 A and |y| <= |v_y|/2 = 6.51 A, so every wrapped
    point lands inside the +/-12 A source window with room for the interpolator.
    """
    lattice = np.column_stack((U, V))                     # columns are u and v
    frac = np.linalg.solve(lattice, np.vstack([px.ravel(), py.ravel()]))
    frac -= np.round(frac)
    wrapped = lattice @ frac
    return wrapped[0].reshape(px.shape), wrapped[1].reshape(px.shape)


# --- the enlarged plane ------------------------------------------------------
x = np.linspace(-ALL_AXES_RANGE_ANG, ALL_AXES_RANGE_ANG, N_PIX_PER_AXIS)
y = x.copy()
XX, YY = np.meshgrid(x, y, indexing="ij")                 # ImagingPlane convention
wrap_x, wrap_y = wrap_into_source(XX, YY)
print("wrapped coordinate span : |x| <= %.3f A, |y| <= %.3f A   (source half-width 12 A)"
      % (np.abs(wrap_x).max(), np.abs(wrap_y).max()))

phase_base = sample_source(src_base, wrap_x, wrap_y)
delta_tiled = sample_source(src_delta, wrap_x, wrap_y)

# --- isolate one blob --------------------------------------------------------
# Centre the taper on the midpoint of the dipole, measured from the source rather
# than assumed, so the two lobes are treated symmetrically.
SRC_XX, SRC_YY = np.meshgrid(src_x, src_y, indexing="ij")
i_neg = np.unravel_index(np.argmin(src_delta), src_delta.shape)
i_pos = np.unravel_index(np.argmax(src_delta), src_delta.shape)
lobe_negative = np.array([src_x[i_neg[0]], src_y[i_neg[1]]])
lobe_positive = np.array([src_x[i_pos[0]], src_y[i_pos[1]]])
blob_centre = 0.5 * (lobe_negative + lobe_positive)

radius = np.hypot(XX - blob_centre[0], YY - blob_centre[1])
taper = np.clip((radius - TAPER_INNER_ANG) / (TAPER_OUTER_ANG - TAPER_INNER_ANG), 0.0, 1.0)
mask = 0.5 * (1.0 + np.cos(np.pi * taper))                # 1 inside, 0 outside, C1 at both ends

delta_phase = delta_tiled * mask
phase_excited = phase_base + delta_phase

# --- checks ------------------------------------------------------------------
print()
print("base   : %.6f .. %.6f rad   (source %.6f .. %.6f)"
      % (phase_base.min(), phase_base.max(), src_base.min(), src_base.max()))
print("delta  : %+.6f .. %+.6f rad  (source %+.6f .. %+.6f)"
      % (delta_phase.min(), delta_phase.max(), src_delta.min(), src_delta.max()))
print("dipole : negative lobe (%+.2f, %+.2f), positive (%+.2f, %+.2f), centre (%+.2f, %+.2f) A"
      % (*lobe_negative, *lobe_positive, *blob_centre))

# the enlarged base must still reproduce the source inside the original window
inside = (np.abs(x) <= 11.0)
check_x, check_y = np.meshgrid(x[inside], y[inside], indexing="ij")
for name, big, small in (("base", phase_base, src_base), ("delta", delta_tiled, src_delta)):
    direct = sample_source(small, check_x, check_y)
    got = big[np.ix_(inside, inside)]
    err = np.abs(got - direct).max() / np.abs(small).max()
    print("round-trip against the source over |x|,|y| <= 11 A : %-6s %.2e of peak" % (name, err))

# exactly one blob should survive: no |delta| above a modest threshold outside the taper
outside = radius > TAPER_OUTER_ANG
print("max |delta| beyond the taper            : %.3e rad  (must be 0)"
      % np.abs(delta_phase[outside]).max())
kept = np.abs(delta_phase).sum() / np.abs(delta_tiled[radius < 1e9]).sum()
print("fraction of tiled |delta| mass retained : %.1f%%" % (100 * kept))
cut_level = np.abs(delta_tiled[(radius > TAPER_INNER_ANG) & (radius < TAPER_OUTER_ANG)]).max()
print("largest |delta| the taper acts on       : %.5f rad = %.1f%% of peak"
      % (cut_level, 100 * cut_level / np.abs(src_delta).max()))

# --- write -------------------------------------------------------------------
out = {
    "base": phase_base.astype(np.float32),
    "excited": phase_excited.astype(np.float32),
    "x": x.astype(np.float32),
    "y": y.astype(np.float32),
    "delta_phase": delta_phase.astype(np.float32),
    "excitation_mask": mask.astype(np.float32),
    "blob_centre_ang": blob_centre,
    "lattice_vectors_ang": np.stack([U, V]),
    "provenance": (
        "Periodic lattice extension of %s onto all_axes_range=%g, n_pix_per_axis=%d. "
        "The base phase is tiled on the 6x6 hBN supercell lattice "
        "(u = %s, v = %s, a = %.3f A). The excitation is tiled and then multiplied by a "
        "raised-cosine taper centred on the dipole midpoint %s, full weight inside %g A "
        "and zero beyond %g A, so a single blue-red blob survives. "
        "excited = base + delta_phase. Bicubic resampling, one pass. "
        "Source built in dft_decomp/DFT-informed-phase_decomp/quanTEM/scripts/"
        "'21 - Phase for periodic structures.ipynb' cells 22-24 from the trans-defect "
        "hBN 6x6x2 spinpol GPAW calculation; no DFT was re-run."
        % (SOURCE.name, ALL_AXES_RANGE_ANG, N_PIX_PER_AXIS,
           np.round(U, 4).tolist(), np.round(V, 4).tolist(), A_HBN,
           np.round(blob_centre, 3).tolist(), TAPER_INNER_ANG, TAPER_OUTER_ANG)
    ),
}
with open(TARGET, "wb") as handle:
    pickle.dump(out, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("\nwrote %s  (%.2f MB)" % (TARGET, TARGET.stat().st_size / 1e6))
print("shape %s, x in [%.1f, %.1f] A, dx = %.5f A"
      % (phase_base.shape, x[0], x[-1], x[1] - x[0]))
