from cq_conversion.diversity_combining import diversity_combine_idler

import xarray as xr
import numpy as np
from tqdm import tqdm

def autocorrelation_theory_model(x, deltaCq, tau, c):
    return deltaCq**2 / 4 * np.exp(-2 * x / tau) + c

def preprocess_wpwp_ds(ds, coarsen_time=1, project=True, div_comb=True):
    """
    Preprocess function to combine WPWP datasets.
    """

    # Replace gate voltages with indices for concatenation
    ds = ds.drop_vars(['VQD1', 'VQD3'], errors='ignore')

    for coord in ["DeltaVWP1", "DeltaVWP2"]:
        if coord in ds.coords:
            # Round WP values to 5uV
            rounded_coord = np.round(ds.coords[coord].values / 5e-6) * 5e-6
            ds = ds.assign_coords({coord: rounded_coord})

    if div_comb:
        ds = diversity_combine_idler(ds, signal_var="VQD1_signal", idler_var="VQD1_idler").VQD1_signal
    else:
        ds = ds.VQD1_signal

    if project:
        ds = find_quadrature_and_project(ds, snr_threshold=1)

    ds = ds.coarsen(
        time=coarsen_time,
        boundary="trim"
    ).mean().stack(detunings=["VQD1", "VQD3"])

    ds = ds.drop_vars(['detunings'], errors='ignore')
    return ds.assign_coords(detunings=np.arange(len(ds.detunings)))

def nonvec_crosscov(x, lags, demean=True) -> np.ndarray:
    """
    Compute the crosscov in python
    """
    r = np.zeros((*x.shape[:-1], len(lags)), dtype=complex)
    lx = x.shape[-1]
    m = len(lags)


    y = np.conj(x)

    if (y.shape[-1] != lx) or (r.shape[-1] != m):
        raise ValueError("Dimension mismatch")

    if any(abs(l) >= lx for l in lags):
        raise ValueError("Lags out of bounds")

    if demean:
        zx = x - np.mean(x, axis=-1)[..., None]
        zy = y - np.mean(y, axis=-1)[..., None]
    else:
        zx = x
        zy = y

    for k in range(m):
        l = lags[k]

        if l >= 0:
            r[..., k] = np.einsum('...k,...k->...', zx[..., :lx - l], zy[..., l:lx])
        else:
            r[..., k] = np.einsum('...k,...k->...', zx[..., -l:lx], zy[..., :lx + l])

    return r

def fast_xr_crosscov(ds, lags, time='time', dask=False, safe_crosscov=True):
    lags = np.sort(lags)

    Kxx = xr.apply_ufunc(
        nonvec_crosscov,
        ds,
        lags,
        input_core_dims=[[time], []],
        output_core_dims=[['lags']],
        dask='parallelized' if dask else 'forbidden'
    )

    return Kxx

def find_quadrature_and_project(dataset: xr.DataArray, snr_threshold: float = 3.0) -> xr.DataArray:
    def angle_snr(z):
        try:
            z = np.ravel(z)
            z = z[~np.isnan(z)]
            if z.size < 2:
                return np.array(np.nan), np.array(np.nan)
            x, y = z.real, z.imag
            w, v = np.linalg.eig(np.cov([x, y]))
            i, j = np.argmax(w), 1 - np.argmax(w)
            a = np.arctan2(v[1, i], v[0, i])
            if abs(a) >= np.pi / 2:
                a = (a + np.pi) % (2 * np.pi)
            snr = np.sqrt(abs(w[i])) / np.sqrt(abs(w[j]))
            return a, snr
        except Exception:
            return np.array(np.nan), np.array(np.nan)

    dims = list(dataset.dims)
    angle, snr = xr.apply_ufunc(
        angle_snr, dataset,
        input_core_dims=[dims],
        output_core_dims=[[], []],
        exclude_dims=set(dims),
    )
    angle = angle.where(snr >= snr_threshold, 0.0)
    out = (dataset / np.exp(1.0j * angle)).real
    out.attrs = dataset.attrs
    return out

def prepare_wpwp_for_Kxx_fit(wpwp_ds, max_lag=10, thresh=(0.2, 0.1)):

    lags = np.arange(0, max_lag+1)

    # rescale since Vrf units are tiny - better for curvefit
    Kxx = 1e6 * fast_xr_crosscov(wpwp_ds.stack(all_vars=['Bperp', 'detunings']), lags=lags).squeeze().real
    noise_pk = Kxx.sel(lags=0).quantile(0.25).data

    sigmas = 0.5*lags.copy()
    sigmas[0] = 2

    filt_lag1 = (Kxx.sel(lags=1) / noise_pk) > thresh[0]
    filt_lag2 = (Kxx.sel(lags=2) / noise_pk) > thresh[1]
    filt = filt_lag1 & filt_lag2

    Kxx_filt = Kxx.sel(all_vars=filt).reset_index('all_vars')

    # remove the noise peak (due to std^2). Instead of fitting with it, we can correct for it.
    Kxx_filt = Kxx_filt.copy()
    Kxx_filt.loc[dict(lags=0)] = Kxx_filt.sel(lags=0) - noise_pk

    return Kxx_filt, sigmas

def fit_wpwp_datasets(paths):

    fits_res = []
    std_res = []

    for path in tqdm(paths):

        with xr.open_dataset(path, engine='h5netcdf') as wpwp_ds:
            # we will do this in Vrf units instead of DeltaCq to save some compute
            # we can optionally also project to quadrature, but this adds extra compute, we can instead just
            # deal with the larger lag=0 peak, since the signal is not changed by projecting to the correct dimension...
            wpwp_ds = preprocess_wpwp_ds(wpwp_ds, project=False, div_comb=False)

            Kxx_for_fit, Kxx_sigmas_vs_lags = prepare_wpwp_for_Kxx_fit(wpwp_ds, max_lag=9)
            if Kxx_for_fit.all_vars.size == 0:
                print(f"Skipping {path.split('/')[-1]} since no points pass SNR filter")
                continue

            fit_result = Kxx_for_fit.curvefit(
                "lags",
                autocorrelation_theory_model,
                p0={"deltaCq": Kxx_for_fit.sel(lags=1), "tau": 5, "c": 2.5},
                errors='ignore',
                kwargs=dict(sigma=Kxx_sigmas_vs_lags)
            )

            fit_result['curvefit_coefficients'] = fit_result['curvefit_coefficients'].where(fit_result['curvefit_coefficients'].sel(param='tau') >= 0)
            fit_result['curvefit_coefficients'] = fit_result['curvefit_coefficients'].where(fit_result['curvefit_coefficients'].sel(param='deltaCq') >= 0)

            # there is a failure mode of fitting a long timescale, with some amplitude A and offset -A: a flat line. This should rule those out
            fit_result['curvefit_coefficients'] = fit_result['curvefit_coefficients'].where(fit_result['curvefit_coefficients'].sel(param='deltaCq') >= -2*fit_result['curvefit_coefficients'].sel(param='c'))

            fits_res.append(fit_result.expand_dims(['DeltaVWP1', "DeltaVWP2"]))
            std_res.append(wpwp_ds.std('time'))


    # Create coordinate arrays
    DeltaVWP1_coords = np.sort(np.unique([float(r.DeltaVWP1) for r in fits_res]))
    DeltaVWP2_coords = np.sort(np.unique([float(r.DeltaVWP2) for r in fits_res]))
    all_vars_coords = np.arange( max([r.all_vars.size for r in fits_res]) )
    param = ['deltaCq', 'tau', 'c']

    # Prealloc arrays for data
    fits_data = np.full((len(DeltaVWP1_coords), len(DeltaVWP2_coords), len(all_vars_coords), 3), np.nan)
    std_data = np.full((len(DeltaVWP1_coords), len(DeltaVWP2_coords), len(wpwp_ds.Bperp), len(wpwp_ds.detunings)), np.nan)

    wpwp_timescale_fits = xr.Dataset(
        {
            "fits": (["DeltaVWP1", "DeltaVWP2", "all_vars", "param"], fits_data),
            "std": (["DeltaVWP1", "DeltaVWP2", "Bperp", "detunings"], std_data),
        },
        coords={
            "DeltaVWP1": DeltaVWP1_coords,
            "DeltaVWP2": DeltaVWP2_coords,
            "Bperp": wpwp_ds.Bperp,
            "detunings": wpwp_ds.detunings,
            "all_vars": all_vars_coords,
            "param": param,
        }
    )

    for ds in fits_res:
        data = ds['curvefit_coefficients'].data
        assert len(data.shape) == 4
        assert data.shape[0] == 1
        assert data.shape[1] == 1
        assert data.shape[-1] == 3

        data = data[0, 0, :, :]

        wpwp_timescale_fits.fits.loc[dict(DeltaVWP1=ds.DeltaVWP1, DeltaVWP2=ds.DeltaVWP2, all_vars=range(0, data.shape[0]))] = data


    for ds in std_res:
        data = ds.data
        assert len(data.shape) == 4
        assert data.shape[0] == 1
        assert data.shape[1] == 1

        data = data[0, 0, :, :]

        wpwp_timescale_fits['std'].loc[dict(DeltaVWP1=ds.DeltaVWP1, DeltaVWP2=ds.DeltaVWP2)] = data

    return wpwp_timescale_fits
