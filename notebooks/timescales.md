---
jupyter:
  jupytext:
    custom_cell_magics: kql
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.17.3
  kernelspec:
    display_name: python
    language: python
    name: python
---

```python
%load_ext autoreload
%autoreload 2

import sys
sys.path.append("..")
```

```python
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from glob import glob

from matplotlib.gridspec import GridSpec

from analysis_code.plotting_helpers import apply_sublabels

from analysis_code.xmpr_wpwp_plot_helpers import (
    autocorrelation_theory_model,
    fast_xr_crosscov,
    fit_wpwp_datasets
)

from xarray_einstats.stats import kurtosis
```

```python
from paths import CONVERTED_DATA_FOLDER, RAW_DATA_FOLDER, PROCESSED_DATA_FOLDER, FIGURE_OUTPUT_FOLDER

Cq_bperp = xr.load_dataset(CONVERTED_DATA_FOLDER / "xmpr/xmpr_Cq.h5", engine='h5netcdf')
wpwp_kurts = xr.load_dataset(PROCESSED_DATA_FOLDER / "wpwp_kurtosis.h5", engine='h5netcdf')
```

### $B_\perp$ Data

```python
iVqd3 = 5
iVqd1 = 2

iB1 = 24
iB2 = 38 +3

color1 = 'blue'
color2 = 'green'


lags = np.arange(0, 16)
Kxx = fast_xr_crosscov(Cq_bperp.Cq, lags=lags).real / len(Cq_bperp.time)
noise_pk = 276 / (2 * 1.35) / np.sqrt(2) # from paper quoted SNR and DeltaCQ

Kxx_filt = Kxx.isel(VQD3=iVqd3, VQD1=iVqd1).squeeze().copy()
Kxx_filt.loc[dict(lags=0)] = Kxx_filt.sel(lags=0) - noise_pk**2

sigmas = (1+lags).copy()
# the lag=0 point should have high error bars, equivalent to the first 4 actual data points
# (where we usually see signal above the noise floor). This is given the assumption of roughly
# equal white noise across the dataset
sigmas[0] = 10

B_field_fit_result = Kxx_filt.curvefit(
    "lags",
    autocorrelation_theory_model,
    p0={"deltaCq": Kxx_filt.sel(lags=1)**0.5, "tau": 5, "c": Kxx_filt.sel(lags=[-1, -2, -3]).mean()},
    errors='ignore',
    kwargs=dict(sigma=sigmas * 100)
)
```

```python
# Sanity check fits
fig, axes = plt.subplots(4, 4, figsize=(16, 12), sharex=True, sharey=True)
axes = axes.flatten()

n_curves = len(Kxx_filt.Bperp)
curves_per_panel = int(np.ceil(n_curves / 16))

for i, ax in enumerate(axes):
    start = i * curves_per_panel
    end = min((i + 1) * curves_per_panel, n_curves)
    for j in range(start, end):
        ax.plot(
            Kxx_filt.lags,
            Kxx_filt.isel(Bperp=j),
            label=f"Bperp={Kxx_filt.Bperp[j].item():.4f}"
        )
        ax.plot(
            Kxx_filt.lags,
            autocorrelation_theory_model(
                Kxx_filt.lags,
                B_field_fit_result.isel(Bperp=j).sel(param='deltaCq').curvefit_coefficients,
                B_field_fit_result.isel(Bperp=j).sel(param='tau').curvefit_coefficients,
                B_field_fit_result.isel(Bperp=j).sel(param='c').curvefit_coefficients,
            ),
            color='tab:red',
            linestyle='--',
            alpha=0.3
        )
    ax.set_title(f"Panel {i+1}")
    ax.legend(fontsize=7, loc='best')

plt.suptitle("Visualizing all Bperp fits")
plt.tight_layout()
plt.show()
```

### WP-WP Fitting

```python
recalculate_timescale_fits = False
overwrite_saved_fits = False

if recalculate_timescale_fits:

    wpwp_dataset_paths = glob(str(RAW_DATA_FOLDER / "wpwp" / "*"))
    wpwp_timescale_fits = fit_wpwp_datasets(paths=wpwp_dataset_paths)

    if overwrite_saved_fits:
        wpwp_timescale_fits.to_netcdf(PROCESSED_DATA_FOLDER / "wpwp_timescale_fits.h5", engine='h5netcdf')
else:

    # LOAD PRE-GENERATED TIMESCALE FITS
    wpwp_timescale_fits = xr.load_dataset(PROCESSED_DATA_FOLDER / "wpwp_timescale_fits.h5", engine='h5netcdf')
```

```python
SNR_threshold = 1

# Find the noise baseline for wpwp maps - in units of Kxx
# sqrt(2) Complex -> Re std, sqrt(1800) accounting for length of time series, 1e3 since we scale crosscov by 1e6
wpwp_noise_baseline = 1e3 * np.sqrt(1800) * (wpwp_timescale_fits['std'].quantile(0.1) / np.sqrt(2))

signal_cutoff = (SNR_threshold * (2*wpwp_noise_baseline))
```

```python
# Only use SNR > 1 to infer timescales - below 1 (~-0.5 kurtosis) this is prone to poor timescale fits
tauX = wpwp_timescale_fits.fits.where(wpwp_timescale_fits.fits.sel(param='deltaCq') > signal_cutoff).quantile(0.99, ['all_vars']).sel(param='tau').squeeze()
# add a solid background to where we don't find fits
(wpwp_kurts*0).min(['Bperp', 'detunings']).VQD1_signal.plot(vmin=-1, vmax=2, add_colorbar=False, cmap='Grays')
tauX.plot()
plt.scatter([0], [0], color='red', marker='x')
```

```python
(
    wpwp_timescale_fits.fits.quantile(0.99, ['all_vars']).sel(param='deltaCq') / (2*wpwp_noise_baseline)
).squeeze().plot(x='DeltaVWP2')
plt.scatter([0], [0], color='red', marker='x')
plt.title("SNR")
```

## Figure

```python
fs = 8
lw = 1
plt.rcParams.update({
    'font.size': fs,
    'font.family': 'serif',
    'xtick.labelsize': fs,
    'ytick.labelsize': fs,
    'text.usetex': False,
    'text.latex.preamble': '\\usepackage{amsmath}',
    'axes.labelsize': fs,
})
```

```python
#convert to mT and mV

tauX = tauX.assign_coords(DeltaVWP2=tauX.DeltaVWP2 * 1e3, DeltaVWP1=tauX.DeltaVWP1 * 1e3)
wpwp_kurts = wpwp_kurts.assign_coords(Bperp=wpwp_kurts.Bperp * 1e3, DeltaVWP2=wpwp_kurts.DeltaVWP2 * 1e3, DeltaVWP1=wpwp_kurts.DeltaVWP1 * 1e3)
B_field_fit_result = B_field_fit_result.assign_coords(Bperp=B_field_fit_result.Bperp * 1e3)
Cq_bperp = Cq_bperp.assign_coords(Bperp=Cq_bperp.Bperp * 1e3)
```

```python
# Create figure with GridSpec layout
fig = plt.figure(figsize=(4, 7), dpi=300)
gs = GridSpec(4, 2, width_ratios=[20, 1], height_ratios=[2, 2, 2, 4], hspace=0.1, wspace=0.3)

# Top panel (same as before)
# Create the three bottom panels
ax = [fig.add_subplot(gs[i, 0]) for i in range(3)]
ax_wp_wp = fig.add_subplot(gs[3, 0])
# add a solid background to where we don't find fits
(wpwp_kurts*0).min(['Bperp', 'detunings']).VQD1_signal.plot(ax=ax_wp_wp,vmin=-1, vmax=2, add_colorbar=False, cmap='Grays')
tauX.plot(ax=ax_wp_wp, x='DeltaVWP2', vmax=10, cbar_kwargs={
    'label': '$\\tau_X\\; [\\mathrm{\\mu s}]$',
    'fraction': 0.026,
    "pad": 0.04
})

ax_wp_wp.set_title(None)
ax_wp_wp.set_ylabel("$\\Delta$ V$_{\\mathrm{WP1}}$ [mV]")
ax_wp_wp.set_xlabel("$\\Delta$ V$_{\\mathrm{WP2}}$ [mV]")

ax_wp_wp.set_aspect('equal')

delta_DeltaVWP1 = (tauX.DeltaVWP1[-1] - tauX.DeltaVWP1[0])
delta_DeltaVWP2 = (tauX.DeltaVWP2[-1] - tauX.DeltaVWP2[0])
dV12 = (tauX.DeltaVWP1[1] - tauX.DeltaVWP1[0])
dV34 = (tauX.DeltaVWP2[1] - tauX.DeltaVWP2[0])


# Add the plotting logic for the bottom three panels
for i in range(2):

    times = B_field_fit_result.isel(param=i).curvefit_coefficients
    times_error = np.sqrt(B_field_fit_result.isel(cov_i=i, cov_j=i).curvefit_covariance)

    times[times - times_error < 0] = np.nan

    if i==0:
        ax[i].set_ylabel("$\\Delta C_{\mathrm{Q}} [\\mathrm{aF}]$")
    elif i==1:
        ax[i].set_ylabel("$\\tau_X$ [$\\mathrm{\\mu s}$]")
        ax[i]


    ax[i].errorbar(
        times.Bperp, times, yerr=times_error, fmt='o', color='tab:blue', markersize=4
    )


timetrace_coarsening=2
# Last plot (kurtosis)
kurts = kurtosis(
    Cq_bperp.Cq.isel(VQD3=iVqd3,VQD1=iVqd1).coarsen(time=timetrace_coarsening,boundary="trim").mean(),
    dims="time"
)

kurts.plot(ax=ax[-1])
ax[-1].set_ylabel("Kurtosis")
ax[-1].set_xlabel("B$_{\\perp}$ [mT]")
ax[1].set_ylim(0, None)
ax[0].set_yticks([100, 200, 300])
ax[2].set_title("")

# Make the x-axis shared among bottom three plots
for i in range(1, 3):
    ax[i].sharex(ax[0])
    plt.setp(ax[i-1].get_xticklabels(), visible=False)


for i in range(0, 3):
    ax[i].axvline(Cq_bperp.Bperp[iB1], color=color1, linestyle='--', lw=lw)
    ax[i].axvline(Cq_bperp.Bperp[iB2], color=color2, linestyle='--', lw=lw)

# ax[1].axhfill(0, 2, color='tab:orange', linestyle='--', lw=1)
# ax[1].axhspan(0, 2, color='tab:red', alpha=0.3)
ax[1].axhline(1, color='tab:red', linestyle='--', lw=lw)

# Apply sublabels
apply_sublabels(ax + [ax_wp_wp], x=-25, y=[5,5,5,15])

plt.savefig(FIGURE_OUTPUT_FOLDER / "timescales.pdf", bbox_inches='tight')
plt.show()
```

```python

```
