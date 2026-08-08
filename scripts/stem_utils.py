import numpy as np
import abtem

from abtem.waves import Probe
from abtem.core.energy import energy2sigma
from abtem.potentials.iam import PotentialArray
from abtem import PixelatedDetector
from abtem.scan import GridScan

from quantem.core.datastructures import Dataset4dstem
from quantem.diffractive_imaging.dataset_models import PtychographyDatasetRaster
# from quantem.diffractive_imaging.pftm import PFTM, PFTM_DIP
from quantem.core.datastructures import Dataset4dstem
from quantem.core.visualization.visualization import show_2d
from quantem.diffractive_imaging.dataset_models import PtychographyDatasetRaster
from quantem.diffractive_imaging.detector_models import DetectorPixelated
from quantem.diffractive_imaging.object_models import ObjectPixelated
from quantem.diffractive_imaging.probe_models import ProbePixelated
from quantem.diffractive_imaging.ptychography import Ptychography
from quantem.core.utils.diffractive_imaging_utils import fit_probe_circle


ACC_VOLTAGE     = 60e3   # eV
PROBE_SEMIANGLE = 30     # mrad
PROBE_DEFOCUS   = 100    # Å
SCAN_STEP_SIZE  = 0.5    # Å


import numpy as np
import abtem
from abtem import PotentialArray, GridScan, PixelatedDetector
from abtem.waves import Probe
from abtem.core.energy import energy2sigma  # wherever yours comes from

def pad_to_square(arr):
    Nx, Ny = arr.shape
    N = max(Nx, Ny)
    pad_x0 = (N - Nx) // 2
    pad_x1 = (N - Nx) - pad_x0
    pad_y0 = (N - Ny) // 2
    pad_y1 = (N - Ny) - pad_y0
    arr_pad = np.pad(arr, ((pad_x0, pad_x1), (pad_y0, pad_y1)), mode="constant", constant_values=0.0)
    return arr_pad, (pad_x0, pad_y0), (Nx, Ny), N

def generate_datacube(
    phase, x, y,
    acc_voltage,
    probe_semiangle,
    probe_defocus,
    scan_step_size,
    slice_thickness=1.0,
):
    # --- sampling ---
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])

    # IMPORTANT: if dx != dy you'll get anisotropic k-space scaling even if you make it square
    # If you want "square pixels" in k-space *and* correct scaling, you must resample/interpolate phase to dx==dy.
    # if abs(dx - dy) / max(dx, dy) > 1e-6:
    #     raise ValueError(f"dx != dy (dx={dx}, dy={dy}). Resample phase to isotropic sampling first.")

    # --- pad phase to square ---
    phase_sq, (pad_x0, pad_y0), (Nx, Ny), N = pad_to_square(phase)

    # --- phase -> potential slice ---
    sigma = float(energy2sigma(acc_voltage))
    V = phase_sq / (sigma * slice_thickness)
    V_slices = V[np.newaxis, :, :].astype(np.float32)

    potential = PotentialArray(
        array=V_slices,
        slice_thickness=slice_thickness,
        sampling=(dx, dy),
    )

    # --- probe on same grid as potential ---
    probe = abtem.Probe(
        energy=acc_voltage,
        semiangle_cutoff=probe_semiangle,
        defocus=probe_defocus,
    )
    probe.grid.match(potential)

    # --- scan only the *original* (unpadded) rectangle inside the padded potential ---
    # original physical extent:
    extent_x = Nx * dx
    extent_y = Ny * dy

    # start position offset to land scan inside the padded array:
    start_x = pad_x0 * dx
    start_y = pad_y0 * dy

    scan = GridScan(
        start=(start_x, start_y),
        end=(start_x + extent_x, start_y + extent_y),
        sampling=scan_step_size,
        endpoint=False,
    )

    # --- pixelated detector ---
    # max_angle='cutoff' tends to keep a symmetric crop; 'valid' can yield rectangles depending on antialiasing/windowing. :contentReference[oaicite:0]{index=0}
    detector = PixelatedDetector(max_angle="cutoff")

    meas = probe.scan(potential, scan=scan, detectors=detector)
    meas.compute()
    return meas


def generate_datacube_old(
        phase, 
        x, y,
        acc_voltage=ACC_VOLTAGE, 
        probe_semiangle=PROBE_SEMIANGLE,
        probe_defocus=PROBE_DEFOCUS, 
        scan_step_size=SCAN_STEP_SIZE
):

    probe = Probe(
        energy=acc_voltage,
        semiangle_cutoff=probe_semiangle, 
        defocus=probe_defocus
    )

    # phase: 2D numpy array, radians
    # x, y: 1D numpy arrays, Å
    # phase.shape == (len(x), len(y))  (or transpose accordingly)
    # --------------------------------

    # -----------------------
    # 0) Grid sampling
    # -----------------------
    dx = float(x[1] - x[0])  # Å
    dy = float(y[1] - y[0])  # Å

    # -----------------------
    # 1) Choose slice thickness for your "phase object"
    # -----------------------
    # If your phase is a *projected* phase (single transmission), use ONE slice.
    # dz can be the physical monolayer thickness, or just 1 Å as a bookkeeping slice.
    SLICE_THICKNESS = 1.0  # Å  (set e.g. 3.0 Å if you want a monolayer-ish thickness)

    # -----------------------
    # 2) Convert phase -> equivalent projected potential slice
    # -----------------------
    sigma = float(energy2sigma(acc_voltage))  # interaction parameter at your energy

    # V in "abTEM internal electrostatic potential units" consistent with sigma
    V = phase / (sigma * SLICE_THICKNESS)

    # PotentialArray expects a 3D array: (num_slices, Nx, Ny)
    V_slices = V[np.newaxis, :, :].astype(np.float32)

    potential = PotentialArray(
        array=V_slices,
        slice_thickness=SLICE_THICKNESS,
        sampling=(dx, dy),
    )

    # -----------------------
    # 3) Define probe
    # -----------------------
    probe = abtem.Probe(
        energy=acc_voltage,
        semiangle_cutoff=probe_semiangle,
        defocus=probe_defocus,
    )

    # Ensure probe & potential share a compatible grid (important!)
    probe.grid.match(potential)

    # -----------------------
    # 4) Scan + pixelated detector (4D-STEM)
    # -----------------------
    scan = GridScan(
        start=(0.0, 0.0),
        end=potential.extent,
        sampling=scan_step_size,
        endpoint=False,
    )

    detector = PixelatedDetector()

    measurement = probe.scan(potential, scan=scan, detectors=detector)
    measurement.compute()   # typically dims: (Nx_scan, Ny_scan, Nkx, Nky)

    return measurement

def ptycho_wrapper(
        dcube, 
        probe_energy, 
        probe_defoc, 
        step_size, 
        probe_semiangle,
        slice_size=32,
        padding=0,
        tv_weight=0,
        obj_lr=1e-2,
        probe_lr=1e-2,
        num_iters=10,
        batch_size=128,
        visualize=True,
):
    if slice_size is not None:
        slice_size = dcube.shape[0]
        dcube_slice = np.zeros((
                slice_size, slice_size, 
                dcube.shape[2], dcube.shape[3]), 
            dtype=np.float32
        )
        for i in range(slice_size):
            for j in range(slice_size):
                dcube_slice[i,j,:,:] = dcube[i,j,:,:] / dcube[i,j,:,:].sum()
    else:
        dcube_slice = dcube

    dset = Dataset4dstem.from_array(
        dcube_slice,
        # dcube_slice,
        name='dset',
        sampling=(step_size, step_size, 1, 1),
        units=("A", "A", "pixels", "pixels")
    )

    probe_qy0, probe_qx0, probe_R = fit_probe_circle(dset.dp_mean.array, show=False)
    dset.sampling[2] = probe_semiangle / probe_R
    dset.sampling[3] = probe_semiangle / probe_R
    dset.units[2:] = ["mrad", "mrad"]
    probe_R = probe_semiangle / dset.sampling[2]
    # print(dset)

    # dset.get_virtual_image(
    #     mode="annular",
    #     geometry=((probe_qy0, probe_qx0), (probe_R+10, probe_R*5)), 
    #     name="DF",
    #     show=True,
    # )
    # dset.get_virtual_image(
    #     mode="circle", 
    #     geometry=((probe_qy0, probe_qx0), probe_R+2), 
    #     name="BF",
    #     show=True,
    # )

    # dset.show_virtual_images(cmap='viridis')
    # print(dset.shape)


    pdset = PtychographyDatasetRaster.from_dataset4dstem(dset)

    pdset.preprocess(
        com_fit_function="constant",
        plot_rotation=False,
        plot_com=False,
        probe_energy=probe_energy,
        force_com_rotation=0, 
        force_com_transpose=False,
    )    


    # create a pixelated ptychography first, to fit the dft basis to.
    probe_params = {
        "energy" : probe_energy,
        "defocus" : probe_defoc,
        "semiangle_cutoff" : probe_semiangle, 
    }
    detector_model = DetectorPixelated() 

    # Set up ptychography model for excited state
    probe_model = ProbePixelated.from_params(
        num_probes=1,
        probe_params=probe_params,
    )
    obj_model = ObjectPixelated.from_uniform(
        num_slices=1, 
        slice_thicknesses=1,
        obj_type='pure_phase',
    )


    ptycho = Ptychography.from_models(
        dset=pdset,
        obj_model=obj_model,
        probe_model=probe_model,
        detector_model=detector_model,
        device='cuda',
    )

    ptycho.preprocess( 
        obj_padding_px=(padding, padding),
        batch_size=batch_size,
        plot_rotation=False,
        plot_com=False,
    )

    opt_params = { # except type, all args are passed to the optimizer (of type type)
            "object": {
                "type": "adam", 
                "lr": obj_lr
                , 
            },
            "probe": {
                "type": "adam", 
                "lr": probe_lr, 
            },
            # "dataset": { ### for optimizing over descan shifts and probe positions
            #     "type": "adam",
            #     "lr": 1e-4,
            # }
    }

    scheduler_params = {
        "object": { ## scheduler kwargs are passed to the scheduler (of type type)
            # "type": "exp",
            # "factor": 0.1,
            "type": "plateau", ## i like plateau for many cases
        },
        "probe": {
            # "type": "exp",
            "type": "plateau",
            # "threshold": 1e-2, # e.g. plateau kwargs 
            # "patience": 100,
            # "cooldown": 100,
        },
        # "dataset": { 
        #     "type": "exp",  ## exp is also frequently used 
        #     "factor": 0.1,
        # }
    }

    constraints = {
        "object": {
            "tv_weight_xy": tv_weight, ## these are mostly the defaults 
            "tv_weight_z": 0.,
            "fix_potential_baseline": False,
            "identical_slices": True, ## default for this is False 
            "apply_fov_mask": False,
        },
        "probe": {
            "center_probe": False,
            "orthogonalize_probe": True,
        },
        "dataset":{
            "descan_tv_weight": 0,
            "descan_shifts_constant": False, 
        }
    }

    # ptycho.remove_optimizer("probe")
    ptycho.reconstruct(
        num_iters=num_iters,
        reset=True,
        autograd=True, 
        device='cuda',
        constraints=constraints, 
        optimizer_params=opt_params,
        scheduler_params=scheduler_params,
        batch_size=batch_size,
    )
    if visualize:
        ptycho.visualize()

    return ptycho