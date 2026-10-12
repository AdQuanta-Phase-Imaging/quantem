"""Local intensity Jacobian of a single-slice ptychographic forward model, and its
leading object-space modes.

Flat functions only -- no classes, no module state.  Everything takes its inputs
explicitly so the pieces can be used one at a time from a notebook.

Convention
----------
The object is a pure phase object perturbed multiplicatively,

    O_1 = O_0 * exp(+1j * h),     h real,

the exit wave at one scan position is psi_0 = P_R * O_0, the camera wave is

    Psi_0 = crop(F{psi_0}),       F = fftshift o fft2 o ifftshift,

and the linearized change in detector intensity is

    A_R h = -2 Im[ conj(Psi_0) * crop(F{psi_0 * h}) ].

`crop` keeps the central (det_y, det_x) pixels, which is what abTEM's
PixelatedDetector(max_angle="valid") returns.

As a matrix this is A_R = -2 Im[ diag(conj(Psi_0)) F diag(psi_0) ], where F is the
cropped centered-DFT matrix.  F separates into a Kronecker product of two 1-D factors,
so it never has to be assembled from an identity basis.
"""

import numpy as np
import torch

from abtem import PixelatedDetector
from abtem.core.energy import energy2sigma
from abtem.potentials.iam import PotentialArray
from abtem.scan import GridScan
from abtem.waves import Probe

__all__ = [
    "crop_center",
    "zero_pad_center",
    "centered_dft",
    "centered_dft_matrix_cropped",
    "make_phase_potential",
    "make_probe",
    "grid_scan_positions_ang",
    "valid_detector_shape",
    "exit_wave_at_position",
    "camera_wave_from_exit_wave",
    "jacobian_matvec",
    "jacobian_rmatvec",
    "build_jacobian_matrix",
    "leading_jacobian_object_modes",
]


# ---------------------------------------------------------------------------
# array geometry
# ---------------------------------------------------------------------------

def crop_center(array, target_shape):
    """Keep the central ``target_shape`` pixels of the last two axes."""
    full_y, full_x = array.shape[-2:]
    target_y, target_x = target_shape
    start_y = (full_y - target_y) // 2
    start_x = (full_x - target_x) // 2
    return array[..., start_y:start_y + target_y, start_x:start_x + target_x]


def zero_pad_center(array, full_shape):
    """Inverse of :func:`crop_center`: place the last two axes back into a zero array.

    This is the adjoint of the crop, which is what the Jacobian adjoint needs.
    """
    target_y, target_x = array.shape[-2:]
    full_y, full_x = full_shape
    start_y = (full_y - target_y) // 2
    start_x = (full_x - target_x) // 2
    padded = np.zeros(array.shape[:-2] + tuple(full_shape), dtype=array.dtype)
    padded[..., start_y:start_y + target_y, start_x:start_x + target_x] = array
    return padded


def centered_dft(array):
    """``fftshift(fft2(ifftshift(.)))`` over the last two axes.

    Both the position origin and the frequency origin sit at index ``n // 2``, so the
    kernel is ``exp(-2j pi (k - c)(p - c) / n)``.  The matrix of this operator is
    symmetric, which is why the adjoint below reuses the same transform.
    """
    return np.fft.fftshift(
        np.fft.fft2(np.fft.ifftshift(array, axes=(-2, -1)), axes=(-2, -1)),
        axes=(-2, -1),
    )


def centered_dft_matrix_cropped(n_full, n_crop, device="cuda"):
    """The rows of :func:`centered_dft` that :func:`crop_center` keeps, as a matrix.

    Returns ``(n_crop, n_full)`` complex64.  Evaluating the kernel directly costs
    ``n_crop * n_full`` entries instead of transforming an ``n_full**2`` identity basis.
    """
    centre = n_full // 2
    start = (n_full - n_crop) // 2
    frequency_index = (
        torch.arange(start, start + n_crop, device=device, dtype=torch.float64) - centre
    )
    position_index = torch.arange(n_full, device=device, dtype=torch.float64) - centre
    phase = -2.0 * torch.pi * torch.outer(frequency_index, position_index) / n_full
    return torch.exp(1j * phase).to(torch.complex64)


# ---------------------------------------------------------------------------
# abTEM forward-model pieces
# ---------------------------------------------------------------------------

def make_phase_potential(phase_image, acc_voltage_ev, slice_thickness_ang, sampling_ang):
    """Wrap a projected phase image as a one-slice abTEM ``PotentialArray``.

    abTEM applies ``exp(i sigma V dz)``, so a target phase image is divided by
    ``sigma * dz`` on the way in and comes back out as that phase exactly.
    """
    interaction_parameter = float(energy2sigma(acc_voltage_ev))
    projected_potential = phase_image / (interaction_parameter * slice_thickness_ang)
    return PotentialArray(
        array=projected_potential[np.newaxis].astype(np.float32),
        slice_thickness=slice_thickness_ang,
        sampling=sampling_ang,
    )


def make_probe(acc_voltage_ev, semiangle_mrad, defocus_ang, potential):
    """Build a coherent probe and match it to the potential's grid."""
    probe = Probe(
        energy=acc_voltage_ev,
        semiangle_cutoff=semiangle_mrad,
        defocus=defocus_ang,
    )
    probe.match_grid(potential)
    return probe


def grid_scan_positions_ang(potential, step_ang, endpoint=True, start=(0.0, 0.0), end=None):
    """Scan positions in Angstrom, shape ``(n_y, n_x, 2)``.

    ``step_ang`` is a *requested* sampling: ``GridScan`` rounds it so the grid fits the
    extent, so always read the realised positions back from here rather than assuming
    the requested step.
    """
    scan = GridScan(
        start=start,
        end=end if end is not None else potential.extent,
        potential=potential,
        sampling=step_ang,
        endpoint=endpoint,
    )
    return scan.get_positions().astype(np.float32)


def valid_detector_shape(probe, position_xy_ang=(0.0, 0.0)):
    """Detector shape that ``PixelatedDetector(max_angle="valid")`` will return.

    Measured by detecting one built probe wave rather than assumed, so it stays correct
    if the grid changes.  It depends on the grid, sampling and energy -- not on the
    semiangle or defocus -- so one call covers a whole probe-parameter sweep on a fixed
    object grid.
    """
    wave = probe.build(scan=[float(position_xy_ang[0]), float(position_xy_ang[1])])
    detected = PixelatedDetector(max_angle="valid").detect(wave.compute())
    return tuple(int(n) for n in detected.array.shape[-2:])


def exit_wave_at_position(probe, object_complex, position_xy_ang):
    """``psi_0 = P_R * O_0`` at one scan position, as a complex64 array."""
    wave = probe.build(scan=[float(position_xy_ang[0]), float(position_xy_ang[1])])
    probe_realspace = np.asarray(wave.compute().array)
    return (probe_realspace * object_complex).astype(np.complex64)


def camera_wave_from_exit_wave(exit_wave, detector_shape):
    """``Psi_0 = crop(F{psi_0})``, the base detector-plane wavefunction."""
    return crop_center(centered_dft(exit_wave), detector_shape).astype(np.complex64)


# ---------------------------------------------------------------------------
# the Jacobian, matrix-free
# ---------------------------------------------------------------------------

def jacobian_matvec(object_perturbation, exit_wave, camera_wave_base, detector_shape):
    """``A_R h`` for one or many real object perturbations.

    ``object_perturbation`` is ``(n_y, n_x)`` or ``(batch, n_y, n_x)``; the result is
    ``(det_y, det_x)`` or ``(batch, det_y, det_x)``, real.  Precision follows the input:
    a complex64 exit wave gives float32, complex128 gives float64.
    """
    perturbed = exit_wave * object_perturbation
    perturbed_camera = crop_center(centered_dft(perturbed), detector_shape)
    return -2.0 * np.imag(np.conj(camera_wave_base) * perturbed_camera)


def jacobian_rmatvec(detector_field, exit_wave, camera_wave_base):
    """``A_R^T g``, the adjoint, for one or many real detector fields.

    Derivation: with ``v = 1j * g * conj(Psi_0)`` and ``F`` symmetric,

        A_R^T g = 2 Re[ psi_0 * centered_dft(zero_pad(v)) ].

    ``detector_field`` is ``(det_y, det_x)`` or ``(batch, det_y, det_x)``; the result
    has the object shape, real, at the input's precision.
    """
    weighted = 1j * detector_field * np.conj(camera_wave_base)
    padded = zero_pad_center(weighted, exit_wave.shape[-2:])
    return 2.0 * np.real(exit_wave * centered_dft(padded))


# ---------------------------------------------------------------------------
# the Jacobian, dense
# ---------------------------------------------------------------------------

def build_jacobian_matrix(exit_wave, camera_wave_base, device="cuda", keep_ft_operator=False):
    """Assemble ``A_R`` densely as ``(det_y * det_x, n_y * n_x)`` float32 on ``device``.

    Scaling by a diagonal matrix is scaling rows and columns, so

        A_R[(ky,kx), (p,q)] = -2 Im( conj(Psi_0[ky,kx]) Fy[ky,p] Fx[kx,q] psi_0[p,q] ),

    and the Kronecker factors ``Fy``, ``Fx`` are the only matrices that ever have to be
    built from the kernel.  Peak device memory is ``2x`` the complex64 operator; the
    float32 result is half of it.  Returns ``A_R``, or ``(A_R, ft_operator)`` when
    ``keep_ft_operator`` -- the FT operator does not depend on the scan position, but it
    rebuilds in a few ms, so only keep it if device memory is otherwise free.
    """
    n_y, n_x = exit_wave.shape[-2:]
    det_y, det_x = camera_wave_base.shape[-2:]

    ft_rows_y = centered_dft_matrix_cropped(n_y, det_y, device=device)
    ft_rows_x = centered_dft_matrix_cropped(n_x, det_x, device=device)
    ft_operator = torch.kron(ft_rows_y, ft_rows_x)

    exit_wave_flat = torch.as_tensor(
        np.asarray(exit_wave).reshape(-1), device=device, dtype=torch.complex64
    )
    camera_wave_flat = torch.as_tensor(
        np.asarray(camera_wave_base).reshape(-1), device=device, dtype=torch.complex64
    )

    scaled = ft_operator * torch.conj(camera_wave_flat)[:, None]
    scaled *= exit_wave_flat[None, :]
    jacobian = -2.0 * scaled.imag          # fresh contiguous float32, not a view
    del scaled
    if keep_ft_operator:
        return jacobian, ft_operator
    del ft_operator
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return jacobian


# ---------------------------------------------------------------------------
# leading object-space modes
# ---------------------------------------------------------------------------

def leading_jacobian_object_modes(
    exit_wave,
    camera_wave_base,
    detector_shape,
    n_modes=4,
    n_oversample=20,
    n_power_iterations=10,
    seed=0,
):
    """Top ``n_modes`` right-singular vectors of ``A_R``, in object space.

    These are the real phase perturbations the measurement is most sensitive to at this
    probe position: mode ``i`` produces a detector response of norm ``singular_value[i]``
    per unit norm of perturbation, so a mode with a small singular value is a
    perturbation this probe barely sees.

    Randomized subspace iteration on the matrix-free operator -- the dense ``A_R`` is
    never formed.  The operator is evaluated in double precision and the basis is
    re-orthogonalized in both spaces every half-step, because the leading singular
    values of a localized probe are closely spaced and a float32 single-sided iteration
    loses the trailing modes.  Cost is ``2 * (n_modes + n_oversample)`` FFTs of the
    object grid per iteration.

    Convergence is governed by the spectral gap: nearly degenerate singular values make
    the individual vectors ill-defined even though their joint subspace is not, so check
    ``singular_values`` for near-ties before reading one mode on its own.

    Each mode's sign is fixed so its largest-magnitude pixel is positive, so panels from
    different probe settings can be compared without sign flipping.

    Returns
    -------
    object_modes : ``(n_modes, n_y, n_x)`` float32
    singular_values : ``(n_modes,)`` float32
    """
    n_y, n_x = exit_wave.shape[-2:]
    n_sketch = n_modes + n_oversample

    exit_wave = np.asarray(exit_wave, dtype=np.complex128)
    camera_wave_base = np.asarray(camera_wave_base, dtype=np.complex128)

    def forward(object_basis):
        return jacobian_matvec(
            object_basis.T.reshape(n_sketch, n_y, n_x),
            exit_wave, camera_wave_base, detector_shape,
        ).reshape(n_sketch, -1).T

    def adjoint(detector_basis):
        return jacobian_rmatvec(
            detector_basis.T.reshape(n_sketch, *detector_shape), exit_wave, camera_wave_base
        ).reshape(n_sketch, -1).T

    rng = np.random.default_rng(seed)
    sketch = rng.standard_normal((np.prod(detector_shape), n_sketch))
    object_basis = np.linalg.qr(adjoint(sketch))[0]

    for _ in range(n_power_iterations):
        detector_basis = np.linalg.qr(forward(object_basis))[0]
        object_basis = np.linalg.qr(adjoint(detector_basis))[0]

    projected = forward(object_basis)
    _, singular_values, right_t = np.linalg.svd(projected, full_matrices=False)

    object_modes = (object_basis @ right_t.T).T[:n_modes].reshape(n_modes, n_y, n_x)
    for mode in object_modes:
        if mode.flat[np.argmax(np.abs(mode))] < 0:
            mode *= -1.0
    return object_modes.astype(np.float32), singular_values[:n_modes].astype(np.float32)
