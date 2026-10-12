"""Gram matrix of the stacked ptychographic Jacobian, without the GEMM.

The dense route forms ``A_R`` (det_y*det_x, n_window) at every scan position and takes
``A_R.T @ A_R``.  That is ``2 N^2 D`` flops per position and it dominates everything else
in the notebook.  The sum over detector pixels can be done in closed form instead.

Derivation
----------
With ``psi`` the exit wave on the object grid, ``Psi`` the cropped camera wave, and
``F[k,p] = exp(-2i pi (k-c)(p-c)/n)`` the centred DFT (symmetric in k and p),

    A[k,p] = -2 Im( conj(Psi_k) F[k,p] psi_p ).

Using ``Im(a) Im(b) = (Re(a conj(b)) - Re(a b)) / 2`` the detector sum collapses:

    G[p,q] = sum_k A[k,p] A[k,q]
           = 2 Re[ psi_p conj(psi_q) T1(p-q) ] - 2 Re[ psi_p psi_q T2(p+q) ]

because ``F[k,p] conj(F[k,q])`` depends only on ``p-q`` and ``F[k,p] F[k,q]`` only on
``p+q``.  The two lookup tables are single FFTs of detector-plane quantities, zero-padded
from the detector back onto the object grid (zero-padding is the adjoint of the crop):

    T1 = centred_dft( zeropad( |Psi|^2 ) )
    T2 = centred_dft( zeropad( conj(Psi)^2 ) )

Both are indexed modulo ``n``, so the whole Gram contribution of one scan position costs
two FFTs plus two outer products: about ``4 N^2`` flops instead of ``2 N^2 D``.

The result is exact -- this is an identity, not an approximation.  It is also, in float32,
*more* accurate than the GEMM, because the sum over the ``D`` detector pixels is performed
by the FFT rather than by accumulating ``D`` rounded products.

Accuracy note
-------------
Forming any Gram squares the condition number, so the smallest singular values a float64
Gram resolves sit near ``1e-8 * sigma_1`` and a float32 one near ``1e-5 * sigma_1``.  That
limit is a property of the Gram, not of the route taken to it, and it applies here too.

Usage
-----
    python fft_structured_gram.py verify
    python fft_structured_gram.py benchmark --window 64
    python fft_structured_gram.py benchmark --window 0        # 0 = the full object grid
    python fft_structured_gram.py run --window 64 --spectrum
"""

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import linearized_jacobian as lj

PHASE_CACHE_PATH = SCRIPT_DIR.parent / "bcenters_base_excited_phases_wide_fov.pkl"
ACC_VOLTAGE_EV = 60e3
PROBE_SEMIANGLE_MRAD = 30.0
PROBE_DEFOCUS_ANG = 0.0
SLICE_THICKNESS_ANG = 1.0
SCAN_STEP_SIZE_ANG = 0.5
SCAN_REGION_START = (7.0, 7.0)
SCAN_REGION_END = (17.0, 17.0)


# ---------------------------------------------------------------------------
# setup
# ---------------------------------------------------------------------------

def build_waves(device, n_positions=None, scan_step_ang=SCAN_STEP_SIZE_ANG):
    """Exit waves and camera waves at every scan position, as the notebook builds them."""
    with open(PHASE_CACHE_PATH, "rb") as handle:
        phase_cache = pickle.load(handle)
    phase_base = np.asarray(phase_cache["base"], dtype=np.float32)
    object_sampling_ang = (
        float(phase_cache["y"][1] - phase_cache["y"][0]),
        float(phase_cache["x"][1] - phase_cache["x"][0]),
    )
    potential = lj.make_phase_potential(
        phase_base, ACC_VOLTAGE_EV, SLICE_THICKNESS_ANG, object_sampling_ang
    )
    probe = lj.make_probe(ACC_VOLTAGE_EV, PROBE_SEMIANGLE_MRAD, PROBE_DEFOCUS_ANG, potential)
    detector_shape = lj.valid_detector_shape(probe)

    scan_positions_xy_ang = lj.grid_scan_positions_ang(
        potential, scan_step_ang, start=SCAN_REGION_START, end=SCAN_REGION_END
    ).reshape(-1, 2)
    if n_positions is not None:
        scan_positions_xy_ang = scan_positions_xy_ang[:n_positions]

    object_base = torch.as_tensor(
        np.exp(1j * phase_base).astype(np.complex64), device=device
    )
    probe_by_scan = torch.as_tensor(
        np.asarray(probe.build(scan=scan_positions_xy_ang.tolist()).compute().array
                   ).astype(np.complex64),
        device=device,
    )
    exit_wave_by_scan = probe_by_scan * object_base
    camera_wave_by_scan = crop_to_detector(
        centered_dft_torch(exit_wave_by_scan.to(torch.complex128)), detector_shape
    )
    return exit_wave_by_scan, camera_wave_by_scan, detector_shape, object_sampling_ang


def centered_dft_torch(array):
    return torch.fft.fftshift(
        torch.fft.fft2(torch.fft.ifftshift(array, dim=(-2, -1))), dim=(-2, -1)
    )


def crop_to_detector(array, detector_shape):
    n_y, n_x = array.shape[-2:]
    det_y, det_x = detector_shape
    row_start = (n_y - det_y) // 2
    col_start = (n_x - det_x) // 2
    return array[..., row_start:row_start + det_y, col_start:col_start + det_x]


def window_indices(n_object_y, n_object_x, window_width_px):
    """Object rows and columns taking part, as index tensors. ``0`` means the whole grid."""
    if not window_width_px:
        return np.arange(n_object_y), np.arange(n_object_x)
    row_start = (n_object_y - window_width_px) // 2
    col_start = (n_object_x - window_width_px) // 2
    return (np.arange(row_start, row_start + window_width_px),
            np.arange(col_start, col_start + window_width_px))


# ---------------------------------------------------------------------------
# the structured route
# ---------------------------------------------------------------------------

def detector_lookup_tables(camera_wave, object_shape, complex_dtype):
    """``T1`` and ``T2``: the two FFTs that replace the detector sum.

    ``T1 = F{zeropad(|Psi|^2)}`` and ``T2 = F{zeropad(conj(Psi)^2)}``, both on the object
    grid and in centred layout, so entry ``m`` holds the table at argument ``m - n // 2``.
    """
    n_object_y, n_object_x = object_shape
    det_y, det_x = camera_wave.shape[-2:]
    row_start = (n_object_y - det_y) // 2
    col_start = (n_object_x - det_x) // 2

    padded_intensity = torch.zeros(object_shape, device=camera_wave.device, dtype=complex_dtype)
    padded_square = torch.zeros_like(padded_intensity)
    padded_intensity[row_start:row_start + det_y, col_start:col_start + det_x] = (
        camera_wave.abs() ** 2
    ).to(complex_dtype)
    padded_square[row_start:row_start + det_y, col_start:col_start + det_x] = (
        torch.conj(camera_wave) ** 2
    ).to(complex_dtype)
    return centered_dft_torch(padded_intensity), centered_dft_torch(padded_square)


def _gram_block_kernel(table_difference, table_sum, gather_dy, gather_dx, gather_sy,
                       gather_sx, psi_block, psi_columns, column_stop):
    """One row block of ``A_R^T A_R``: gather the two tables and combine them.

    Written as one function so ``torch.compile`` can fuse it.  Unfused it costs roughly
    eight passes over a ``(block, N)`` array to produce one; fused it should approach the
    traffic of the Gram update alone.
    """
    difference_block = table_difference[
        gather_dy[:, :, None], gather_dx[:, None, :]
    ].reshape(-1, column_stop)
    sum_block = table_sum[
        gather_sy[:, :, None], gather_sx[:, None, :]
    ].reshape(-1, column_stop)
    difference_block = difference_block * psi_block[:, None] * torch.conj(psi_columns)[None, :]
    sum_block = sum_block * psi_block[:, None] * psi_columns[None, :]
    return 2.0 * (difference_block.real - sum_block.real)


def structured_gram_accumulate(exit_wave_by_scan, camera_wave_by_scan, object_rows,
                               object_cols, real_dtype, complex_dtype,
                               row_block=1024, symmetric=False, progress_every=0,
                               compile_kernel=False):
    """``sum_R A_R^T A_R`` over every scan position, via the two lookup tables.

    ``row_block`` object rows of the Gram are produced at a time.  The ``(p-q)`` and
    ``(p+q)`` index tables are never materialised at ``N x N``: they factor into a
    ``(block, len(rows))`` and a ``(block, len(cols))`` table, and the gather broadcasts
    them.

    With ``symmetric`` only the lower triangle is computed and mirrored at the end, which
    halves the work.
    """
    device = exit_wave_by_scan.device
    n_object_y, n_object_x = exit_wave_by_scan.shape[-2:]
    centre_y, centre_x = n_object_y // 2, n_object_x // 2

    row_index = torch.as_tensor(object_rows, device=device, dtype=torch.long)
    col_index = torch.as_tensor(object_cols, device=device, dtype=torch.long)
    n_rows, n_cols = len(row_index), len(col_index)
    n_window = n_rows * n_cols

    # flat object-pixel coordinates of the window, as raw grid indices
    pixel_y = row_index.repeat_interleave(n_cols)
    pixel_x = col_index.repeat(n_rows)

    exit_wave_window = exit_wave_by_scan[:, row_index][:, :, col_index].reshape(
        len(exit_wave_by_scan), -1
    ).to(complex_dtype)

    gram_block = (torch.compile(_gram_block_kernel, dynamic=False)
                  if compile_kernel else _gram_block_kernel)

    gram = torch.zeros(n_window, n_window, device=device, dtype=real_dtype)
    start_time = time.time()
    for scan_index in range(len(exit_wave_by_scan)):
        table_difference, table_sum = detector_lookup_tables(
            camera_wave_by_scan[scan_index], (n_object_y, n_object_x), complex_dtype
        )
        psi = exit_wave_window[scan_index]
        for block_start in range(0, n_window, row_block):
            block_stop = min(block_start + row_block, n_window)
            # The gather factors into (block x window_rows) and (block x window_cols), so
            # the column cut only saves work when it lands on a whole number of window
            # rows.  Round it up to one.
            if symmetric:
                n_rows_needed = min(n_rows, -(-block_stop // n_cols))
                column_stop = n_rows_needed * n_cols
            else:
                n_rows_needed = n_rows
                column_stop = n_window
            active_row_index = row_index[:n_rows_needed]

            block_y = pixel_y[block_start:block_stop]
            block_x = pixel_x[block_start:block_stop]
            # T1 is indexed at (p - q) + centre, T2 at (p + q) - centre, both mod n
            gather_dy = (block_y[:, None] - active_row_index[None, :] + centre_y) % n_object_y
            gather_dx = (block_x[:, None] - col_index[None, :] + centre_x) % n_object_x
            gather_sy = (block_y[:, None] + active_row_index[None, :] - centre_y) % n_object_y
            gather_sx = (block_x[:, None] + col_index[None, :] - centre_x) % n_object_x

            gram[block_start:block_stop, :column_stop] += gram_block(
                table_difference, table_sum, gather_dy, gather_dx, gather_sy, gather_sx,
                psi[block_start:block_stop], psi[:column_stop], column_stop,
            ).to(real_dtype)
        if progress_every and (scan_index + 1) % progress_every == 0:
            torch.cuda.synchronize()
            print(f"    {scan_index + 1:5d}/{len(exit_wave_by_scan)} positions "
                  f"({time.time() - start_time:6.1f} s)", flush=True)

    if symmetric:
        # Mirror the lower triangle into the upper one a block at a time: the obvious
        # ``gram + gram.T`` needs two more full-size copies, which on the full object grid
        # is another 32 GB.  Entries above the diagonal that were computed anyway (the
        # column cut rounds up to a whole window row) are correct, so overwriting them
        # with the mirror is harmless.
        for block_start in range(row_block, n_window, row_block):
            block_stop = min(block_start + row_block, n_window)
            gram[:block_start, block_start:block_stop] = (
                gram[block_start:block_stop, :block_start].T
            )
    return gram


# ---------------------------------------------------------------------------
# the dense route, for reference
# ---------------------------------------------------------------------------

def cropped_dft_rows(n_full, n_crop, window_index, complex_dtype, device):
    centre = n_full // 2
    crop_start = (n_full - n_crop) // 2
    frequency_index = torch.arange(crop_start, crop_start + n_crop,
                                   device=device, dtype=torch.float64) - centre
    position_index = torch.as_tensor(window_index, device=device, dtype=torch.float64) - centre
    phase = -2.0 * torch.pi * torch.outer(frequency_index, position_index) / n_full
    return torch.exp(1j * phase).to(complex_dtype)


def dense_gram_accumulate(exit_wave_by_scan, camera_wave_by_scan, object_rows, object_cols,
                          real_dtype, complex_dtype, progress_every=0, gram_row_block=4096,
                          rebuild_each_position=False):
    """``sum_R A_R^T A_R`` the way the notebook does it now: build ``A_R``, then a GEMM.

    ``rebuild_each_position`` rebuilds the Kronecker operator every position, which is
    what the in-place scaling forces (the scaling destroys the operator).  It costs one
    extra pass over the operator and is the only way the full object grid fits.
    """
    device = exit_wave_by_scan.device
    n_object_y, n_object_x = exit_wave_by_scan.shape[-2:]
    det_y, det_x = camera_wave_by_scan.shape[-2:]
    n_window = len(object_rows) * len(object_cols)

    def build_operator():
        return torch.kron(
            cropped_dft_rows(n_object_y, det_y, object_rows, complex_dtype, device),
            cropped_dft_rows(n_object_x, det_x, object_cols, complex_dtype, device),
        )

    rebuild_operator = build_operator if rebuild_each_position else None
    ft_operator = None if rebuild_each_position else build_operator()
    row_index = torch.as_tensor(object_rows, device=device, dtype=torch.long)
    col_index = torch.as_tensor(object_cols, device=device, dtype=torch.long)
    exit_wave_window = exit_wave_by_scan[:, row_index][:, :, col_index].reshape(
        len(exit_wave_by_scan), -1
    ).to(complex_dtype)
    conj_camera = torch.conj(camera_wave_by_scan).reshape(len(camera_wave_by_scan), -1).to(complex_dtype)

    gram = torch.zeros(n_window, n_window, device=device, dtype=real_dtype)
    start_time = time.time()
    for scan_index in range(len(exit_wave_by_scan)):
        # Scale in place and block the GEMM, as the notebook's iter_jacobian_by_scan does:
        # the out-of-place form needs a second full-size complex temporary and a full-size
        # GEMM output, neither of which fits beside the Gram on the full object grid.
        # NB: the in-place scaling below destroys whatever it is given, so the cached
        # operator must be cloned, never passed directly.
        jacobian_block = ft_operator.clone() if rebuild_operator is None else rebuild_operator()
        jacobian_block *= conj_camera[scan_index][:, None]
        jacobian_block *= exit_wave_window[scan_index][None, :]
        jacobian_block = (-2.0 * jacobian_block.imag).to(real_dtype)
        for block_start in range(0, n_window, gram_row_block):
            block_stop = min(block_start + gram_row_block, n_window)
            gram[block_start:block_stop] += (
                jacobian_block[:, block_start:block_stop].T @ jacobian_block
            )
        del jacobian_block
        if progress_every and (scan_index + 1) % progress_every == 0:
            torch.cuda.synchronize()
            print(f"    {scan_index + 1:5d}/{len(exit_wave_by_scan)} positions "
                  f"({time.time() - start_time:6.1f} s)", flush=True)
    del ft_operator
    torch.cuda.empty_cache()
    return gram


# ---------------------------------------------------------------------------
# modes
# ---------------------------------------------------------------------------

def dtypes_for(precision):
    return ((torch.float64, torch.complex128) if precision == "float64"
            else (torch.float32, torch.complex64))


def command_verify(args):
    """Exactness check: structured vs dense, on windows small enough to do both densely."""
    device = args.device
    exit_wave, camera_wave, detector_shape, _ = build_waves(device, n_positions=args.positions)
    print(f"object grid {tuple(exit_wave.shape[-2:])}, detector {detector_shape}, "
          f"{len(exit_wave)} scan positions\n")
    print(f"{'window':>8} {'N':>7} {'precision':>10} {'rel L2 vs dense float64':>26}")
    reference = {}
    for window in args.verify_windows:
        rows, cols = window_indices(*exit_wave.shape[-2:], window)
        gram_reference = dense_gram_accumulate(
            exit_wave, camera_wave, rows, cols, torch.float64, torch.complex128
        )
        reference[window] = gram_reference
        for precision in ("float64", "float32"):
            real_dtype, complex_dtype = dtypes_for(precision)
            gram_structured = structured_gram_accumulate(
                exit_wave, camera_wave, rows, cols, real_dtype, complex_dtype,
                row_block=args.row_block, symmetric=args.symmetric,
                compile_kernel=args.compile
            )
            error = float(torch.linalg.norm(gram_structured.double() - gram_reference)
                          / torch.linalg.norm(gram_reference))
            print(f"{window:>8} {len(rows)*len(cols):>7} {precision:>10} {error:>26.3e}")
        # and the dense route in float32, for the comparison that matters
        gram_dense32 = dense_gram_accumulate(
            exit_wave, camera_wave, rows, cols, torch.float32, torch.complex64
        )
        error = float(torch.linalg.norm(gram_dense32.double() - gram_reference)
                      / torch.linalg.norm(gram_reference))
        print(f"{'':>8} {'':>7} {'(dense f32)':>10} {error:>26.3e}")
    return reference


def command_benchmark(args):
    device = args.device
    exit_wave, camera_wave, detector_shape, _ = build_waves(device, n_positions=args.positions)
    rows, cols = window_indices(*exit_wave.shape[-2:], args.window)
    n_window = len(rows) * len(cols)
    n_detector = detector_shape[0] * detector_shape[1]
    real_dtype, complex_dtype = dtypes_for(args.precision)
    total_positions = len(lj.grid_scan_positions_ang(
        lj.make_phase_potential(
            np.zeros(tuple(exit_wave.shape[-2:]), dtype=np.float32), ACC_VOLTAGE_EV,
            SLICE_THICKNESS_ANG, (0.096, 0.096)),
        SCAN_STEP_SIZE_ANG, start=SCAN_REGION_START, end=SCAN_REGION_END).reshape(-1, 2))

    print(f"window {args.window or 'full grid'}  ->  N = {n_window} object px, "
          f"D = {n_detector} detector px, {args.precision}")
    print(f"timing on {len(exit_wave)} scan positions (the full scan is {total_positions})")
    print(f"dense  flops/position : {2 * n_window**2 * n_detector:.3e}")
    print(f"struct flops/position : {4 * n_window**2:.3e}   "
          f"(ratio {2 * n_window**2 * n_detector / (4 * n_window**2):.0f}x)\n")

    results = {}
    for name, fn in [("structured", lambda: structured_gram_accumulate(
                          exit_wave, camera_wave, rows, cols, real_dtype, complex_dtype,
                          row_block=args.row_block, symmetric=args.symmetric,
                          compile_kernel=args.compile)),
                     ("dense", lambda: dense_gram_accumulate(
                          exit_wave, camera_wave, rows, cols, real_dtype, complex_dtype,
                          gram_row_block=args.gram_row_block,
                          rebuild_each_position=args.rebuild_operator))]:
        if name == "dense" and args.skip_dense:
            print(f"{name:>11}: skipped")
            continue
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start_time = time.time()
        gram = fn()
        torch.cuda.synchronize()
        elapsed = time.time() - start_time
        per_position = elapsed / len(exit_wave)
        print(f"{name:>11}: {elapsed:7.2f} s   {per_position * 1e3:8.1f} ms/position   "
              f"peak {torch.cuda.max_memory_allocated() / 1e9:5.2f} GB   "
              f"-> full scan {per_position * total_positions:7.1f} s")
        # Two Grams of the full object grid do not fit beside each other, so only hold
        # them both when they are going to be compared.
        if args.no_compare:
            del gram
        else:
            results[name] = gram
        torch.cuda.empty_cache()

    if len(results) == 2:
        a, b = results["structured"].double(), results["dense"].double()
        print(f"\nagreement structured vs dense : rel L2 = "
              f"{float(torch.linalg.norm(a - b) / torch.linalg.norm(b)):.3e}")


def command_run(args):
    device = args.device
    exit_wave, camera_wave, detector_shape, object_sampling_ang = build_waves(device)
    rows, cols = window_indices(*exit_wave.shape[-2:], args.window)
    real_dtype, complex_dtype = dtypes_for(args.precision)
    n_window = len(rows) * len(cols)
    print(f"accumulating the Gram: N = {n_window}, {len(exit_wave)} scan positions, "
          f"{args.precision}, symmetric={args.symmetric}")
    print(f"Gram memory: {n_window**2 * (8 if args.precision == 'float64' else 4) / 1e9:.2f} GB")

    torch.cuda.synchronize()
    start_time = time.time()
    gram = structured_gram_accumulate(
        exit_wave, camera_wave, rows, cols, real_dtype, complex_dtype,
        row_block=args.row_block, symmetric=args.symmetric,
        progress_every=args.progress_every, compile_kernel=args.compile,
    )
    torch.cuda.synchronize()
    print(f"accumulated in {time.time() - start_time:.1f} s")

    if args.spectrum:
        start_time = time.time()
        eigenvalues = torch.linalg.eigvalsh(gram.double())
        singular_values = eigenvalues.flip(0).clamp(min=0).sqrt().cpu().numpy()
        print(f"eigh in {time.time() - start_time:.1f} s")
        floor = singular_values[0] * (1e-8 if args.precision == "float64" else 1e-5)
        print(f"sigma_1 = {singular_values[0]:.4e}   "
              f"modes above {floor:.3e} : "
              f"{int((singular_values > floor).sum())} / {len(singular_values)}")
        if args.save:
            np.save(args.save, singular_values)
            print(f"singular values -> {args.save}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["verify", "benchmark", "run"])
    parser.add_argument("--window", type=int, default=64,
                        help="square object window in pixels; 0 = the whole object grid")
    parser.add_argument("--verify-windows", type=int, nargs="+", default=[8, 16, 24])
    parser.add_argument("--positions", type=int, default=None,
                        help="use only the first N scan positions")
    parser.add_argument("--precision", choices=["float32", "float64"], default="float64")
    parser.add_argument("--row-block", type=int, default=1024)
    parser.add_argument("--symmetric", action="store_true",
                        help="compute the lower triangle only and mirror")
    parser.add_argument("--skip-dense", action="store_true")
    parser.add_argument("--compile", action="store_true",
                        help="torch.compile the inner block, fusing the gather and the "
                             "elementwise chain into one kernel")
    parser.add_argument("--gram-row-block", type=int, default=4096,
                        help="dense route: object rows of the Gram per GEMM")
    parser.add_argument("--rebuild-operator", action="store_true",
                        help="dense route: rebuild the Kronecker operator every position "
                             "(what the in-place scaling forces; needed on the full grid)")
    parser.add_argument("--no-compare", action="store_true",
                        help="free each Gram right after timing it; needed on the full grid")
    parser.add_argument("--spectrum", action="store_true")
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    {"verify": command_verify, "benchmark": command_benchmark, "run": command_run}[args.command](args)


if __name__ == "__main__":
    main()
