import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

def phase(ds, unwrap=False):
    sol = xr.apply_ufunc(np.angle, ds, dask='allowed')
    if not unwrap:
        return sol
    return xr.apply_ufunc(np.unwrap, sol, dask='allowed')

def detect_idler_phase_coord(ds, idler_var, sig_var):
    bad_coords = set()
    delta_angle = phase(ds[idler_var] - ds[sig_var])

    for coord in ds.coords:
        averaged_delta = delta_angle.mean(dim=set(ds.coords) - {coord})
        dphase = np.max(averaged_delta) - np.min(averaged_delta)

        if dphase > 0.25 * np.pi:
            bad_coords.add(coord)

    return bad_coords

def change_reference_frame_idler(
    ds, idler_var, sig_var, average_skip_coords = None, plot=True, add_uncomp_idler=False, skip_avg=False
):

    comp_mat = np.exp(-1j * 4 * np.pi * ds.pump_frequency * ds.time)
    idl_data_comp = ds[idler_var] * comp_mat
    if add_uncomp_idler:
        ds["uncompensated_idler"] = ds[idler_var].copy()
    elif plot:
            raise ValueError("Can't not add uncompensated idler if plot is True")

    ds[idler_var] = np.conj(idl_data_comp)

    if average_skip_coords is None:
        bad_coords = detect_idler_phase_coord(ds, idler_var, sig_var)
    else:
        bad_coords = average_skip_coords

    # ds[idler_var] = 2*idl_data_comp.mean() - idl_data_comp

    mean_proj_dim = (
        set(ds.coords) - bad_coords
    )  # take mean on everything except for magnet to account for random rotations

    ds[idler_var] *= (
        np.exp(
            1j
            * (
                phase(ds[sig_var].mean(mean_proj_dim))
                - phase(ds[idler_var].mean(mean_proj_dim))
            )
        )
        * (1 if skip_avg else (
            abs(ds[sig_var].mean(mean_proj_dim))
            / abs(ds[idler_var].mean(mean_proj_dim))
        ))
    )

    all_coords_except_time = [i for i in ds.coords if i != "time"]

    if plot:
        plt.figure()
        phase(ds).isel(
            dict(zip(all_coords_except_time, [0] * len(all_coords_except_time)))
        ).uncompensated_idler.plot(alpha=0.4, label="uncompensated")
        phase(ds, unwrap=True).isel(
            dict(zip(all_coords_except_time, [0] * len(all_coords_except_time)))
        )[idler_var].plot(alpha=0.4, label="compensated")
        plt.legend()


def diversity_combine_idler(ds, *, idler_var, signal_var):

    change_reference_frame_idler(ds, idler_var, signal_var, plot=False)

    std_time = ds[signal_var].std('time')

    inner_detuning = list(std_time.coords)[-1]

    # estimate SNR Ratio
    s_noise_surrogate = std_time.min(inner_detuning)
    s_signal_surrogate = std_time.max(inner_detuning)

    s_snr_surrogate = (s_signal_surrogate / s_noise_surrogate)

    i_noise_surrogate = ds[idler_var].std('time').min(inner_detuning)
    i_signal_surrogate = ds[idler_var].std('time').max(inner_detuning)

    i_snr_surrogate = (i_signal_surrogate / i_noise_surrogate)

    ratio = (s_snr_surrogate / i_snr_surrogate)

    # Modified maximal ratio combining, when rho~0 (HEMT dominated), is the optimal solution
    if abs(ratio.median() - 1) < 0.05:
        signal_weight = 0.5
    else:
        signal_weight = ratio**2 / (1 + ratio**2)

    Vrf_comb = xr.Dataset({
        signal_var:     signal_weight * ds[signal_var] +
                    (1-signal_weight) * ds[idler_var]
    })
    Vrf_comb.attrs = ds.attrs.copy()

    return Vrf_comb
