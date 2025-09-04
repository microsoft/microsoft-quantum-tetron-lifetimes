---
jupyter:
  jupytext:
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
from matplotlib import pyplot as plt
import numpy as np
import xarray as xr
from scipy.optimize import curve_fit
```

```python
fs = 14
lw = 1


fs = 10
lw = 1
plt.rcParams.update({
    'font.size': fs,
    'font.family': 'serif',
    'xtick.labelsize': fs,
    'ytick.labelsize': fs,
    'axes.labelsize': fs,
})
plt.rcParams["figure.autolayout"] = True

```

```python
from analysis_code.timetrace_classification import threshold_traces
from analysis_code.error_metrics import get_error_statistics
from analysis_code.xmpr_analysis import crosscov
```

```python
from paths import CONVERTED_DATA_FOLDER, FIGURE_OUTPUT_FOLDER

ds = xr.load_dataset(CONVERTED_DATA_FOLDER / "long_time_record/long_time_record_Cq.h5", engine='h5netcdf')
```

```python
iV1 = 0
iV3 = 1
iB=114
L = 250

trace = ds.isel(VQD3=iV3, VQD1=iV1).isel(Bperp=iB).squeeze()


coarsened_trace = trace.Cq.coarsen(time=L, boundary="trim").mean()
coarsened_err_a = get_error_statistics(coarsened_trace, parallel=False)

boolean_traces,  boolean_traces_Cq = threshold_traces(
    coarsened_trace,
    mean1=coarsened_err_a.gmm_mean1,
    mean2=coarsened_err_a.gmm_mean2,
    cq_name=None,
    r=0.2
)
```

```python
from matplotlib.gridspec import GridSpec

fig = plt.figure(figsize=(4.5,2.5), layout="constrained")
axs = np.empty(2, dtype=object)
gs = GridSpec(1, 2, width_ratios=[1.5, 1], figure=fig)
fig.subplots_adjust(wspace=0.4)  # Add spacing between the panels
axs[0] = fig.add_subplot(gs[0, 0])
axs[1] = fig.add_subplot(gs[0, 1])

max_lag = 10**(-2)
dt = trace.time[1].data - trace.time[0].data
lags = np.unique(np.logspace(0, 4.5, 100, dtype=int))
# lags = np.concatenate([lags[::-1], lags])
lag_filt = lags < max_lag / dt

# First plot
trace.Cq.coarsen(time=L, boundary='trim').mean().plot(ax=axs[0], label="raw data", color='k')

classification = boolean_traces_Cq.data if (
    np.std(boolean_traces_Cq.data - trace.Cq.coarsen(time=L, boundary='trim').mean()) < np.std(-boolean_traces_Cq.data - trace.Cq.coarsen(time=L, boundary='trim').mean())
) else 2*np.mean(boolean_traces_Cq.data) - boolean_traces_Cq.data

# skip first point since random
axs[0].plot(boolean_traces_Cq.time.data[1:], classification[1:], color='tab:orange')

axs[0].set_xticks([0, 0.05, 0.1])
axs[0].set_xticklabels([f"{tick * 1e3:.0f}" for tick in axs[0].get_xticks()])
axs[0].set_xlabel("Time $[\\mathrm{ms}]$")
axs[0].set_title("")
axs[0].set_ylabel("$C_\\mathrm{Q}\\; [\\mathrm{aF}]$")

# Second plot
rescale_units = 1e3

filt_slow = lag_filt & (lags > 1e-4 / dt)
filt_fast = lag_filt & (lags < 1e-4 / dt)

lags_slow = lags[filt_slow]
lags_fast = lags[filt_fast]
lags_fast_fit = lags[:5]

simple_model = lambda x, a, b: a**2 / 4 * np.exp(-2 * x / b)


fit_slow_timescale, err_slow_timescale = curve_fit(
    simple_model,
    lags_slow,
    crosscov(trace.Cq, trace.Cq, lags_slow),
    p0=[(4 * 5270 * 0.27)**0.5, 9900],
    bounds=([0, 0], [np.inf, np.inf]),
)

fast_offset = simple_model(0, *fit_slow_timescale)
fit_fast_timescale, err_fast_timescale = curve_fit(
    simple_model,
    lags_fast_fit,
    crosscov(trace.Cq, trace.Cq, lags_fast_fit) - fast_offset,
    p0=[(4 * 5270 * 0.6)**0.5, 3.5],
    bounds=([0, 0], [np.inf, np.inf]),
)

fit_slow_err = np.sqrt(np.diag(err_slow_timescale))
fit_fast_err = np.sqrt(np.diag(err_fast_timescale))

axs[1].plot(lags[lag_filt] * dt*rescale_units, crosscov(trace.Cq, trace.Cq, lags[lag_filt]), lw=1, color='k', marker='.')

axs[1].plot(lags_slow * dt*rescale_units, simple_model(lags_slow, *fit_slow_timescale), linestyle=(0, (2, 4)), lw=3, color='tab:orange', label=f"${fit_slow_timescale[1]/1000:.1f} \\pm {fit_slow_err[1] / 1000 :.1f} \\mathrm{{ms}}$")
axs[1].plot(lags_fast * dt*rescale_units, fast_offset + simple_model(lags_fast, *fit_fast_timescale), linestyle=(0, (2, 4)), lw=3, color='tab:green', label=f"${fit_fast_timescale[1]:.1f} \\pm {fit_fast_err[1] :.1f} \\mathrm{{\\mu s}}$")

axs[1].set_xscale('log')
axs[1].set_ylim(200, None)

axs[1].set_xlabel('$t-t\'\\; [\\mathrm{ms}]$')
axs[1].set_ylabel("$\\langle C_\\mathrm{Q}(t) C_\\mathrm{Q}(t') \\rangle\\; [\\mathrm{aF}^2]$")

axs[1].legend(fontsize=fs, handlelength=0.7)

from string import ascii_lowercase
def apply_sublabels(axs, x=-50, y=0, size='medium', weight='bold', ha='left', va='bottom'):
    for n, ax in enumerate(axs):

        if isinstance(x, list):
            _x = x[n]
        else:
            _x = x

        ax.set_title(
            f"({ascii_lowercase[n]})",
            loc="left",
        )

apply_sublabels(axs)

# plt.tight_layout()
plt.savefig(FIGURE_OUTPUT_FOLDER / "long_time_record.pdf")
plt.show()
```

```python
fit_fast_timescale[0], fit_slow_timescale[0]
```

```python

```
