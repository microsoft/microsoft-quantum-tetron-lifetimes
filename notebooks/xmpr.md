---
jupyter:
  jupytext:
    custom_cell_magics: kql
    formats: md,ipynb
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
# Packages
import sympy
import numpy as np
import xarray as xr
from tqdm import tqdm
from glob import glob
```

```python
# Code in this repo
from analysis_code.plotting_helpers import apply_sublabels
from scipy.stats import kurtosis
from analysis_code.xmpr_analysis import (
    make_crosscov_fit_function,
    fit_and_sum_gaussians,
    extract_cross_lags_above_noise
)
from analysis_code.xmpr_wpwp_plot_helpers import (
    preprocess_wpwp_ds,
)
```

```python
# Plotting Imports
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
import matplotlib.ticker as ticker

from PIL import Image
```

```python
from paths import CONVERTED_DATA_FOLDER, RAW_DATA_FOLDER, PROCESSED_DATA_FOLDER, FIGURE_OUTPUT_FOLDER

ds = xr.load_dataset(CONVERTED_DATA_FOLDER / "xmpr/xmpr_Cq.h5", engine='h5netcdf')

ds_zoom1 = xr.load_dataset(CONVERTED_DATA_FOLDER / "xmpr/xmpr_zoomout_1_Cq.h5", engine='h5netcdf')
ds_zoom2 = xr.load_dataset(CONVERTED_DATA_FOLDER / "xmpr/xmpr_zoomout_2_Cq.h5", engine='h5netcdf')
```

```python
fs = 8
plt.rcParams.update({
    'font.size': fs,
    'font.family': 'STIXGeneral',
    'mathtext.fontset' : "cm",
    'xtick.labelsize': fs,
    'ytick.labelsize': fs,
    'axes.labelsize': fs,
    'text.usetex': False,
})

```

# Timescale model

Fit a 2-state telegraph signal model with one rate $\lambda$. In the presence of multiple rates between 2-states, $\lambda = 0.5(\lambda_1 + \lambda_2) = 0.5(\tau_1^{-1} + \tau_2^{-1})$

```python
expr, fit_f = make_crosscov_fit_function()

expr # this is the fit expression
```

## Get Traces

```python
iVqd3 = 5
iVqd1 = 2

iB1 = 24
iB2 = 41

trace1 = ds.Cq.isel(
    VQD3=iVqd3,
    VQD1=iVqd1,
    Bperp=iB1
).squeeze()

trace2 = ds.Cq.isel(
    VQD3=iVqd3,
    VQD1=iVqd1,
    Bperp=iB2
).squeeze()
```

```python
color1 = 'blue'
color2 = 'green'

timetrace_coarsening = 2

timetrace_lw = 0.3
```

```python
#calculate the kurtosis of the trace
kurts = [xr.apply_ufunc(
    kurtosis,
    data.Cq.coarsen(time=timetrace_coarsening,boundary="trim").mean(),
    input_core_dims=[["time"]],
    output_core_dims=[[]],
    vectorize=True,
    dask="parallelized",
    output_dtypes=[np.float64],
) for data in [ds, ds_zoom1, ds_zoom2]]
```

# wpwp fig

```python
# This is fairly compute intensive (loops over ~83Gb of data), so the result is cached for convinience. This takes ~40mins on a descent machine.
recompute_wpwp_kurtosis = False
# This will overwrite the downloaded cached wpwp_kurtosis file.
save_recomputed_wpwp_kurtosis = False
computed_wpwp_location = PROCESSED_DATA_FOLDER / "wpwp_kurtosis.h5"
```

```python
from xarray_einstats.stats import kurtosis as kurtosis_xr
```

```python
if recompute_wpwp_kurtosis:

    wpwp_dataset_paths = glob(str(RAW_DATA_FOLDER / "wpwp" / "*"))

    res = []

    for path in tqdm(wpwp_dataset_paths):

        with xr.open_dataset(path, engine='h5netcdf') as wpwp_ds:
            wpwp_ds = preprocess_wpwp_ds(wpwp_ds, coarsen_time=2)

            wpwp_kurt = kurtosis_xr(
                wpwp_ds,
                dims='time'
            )
            rounded_Bperp = np.round(wpwp_kurt.coords['Bperp'].values, 6)
            wpwp_kurt = wpwp_kurt.assign_coords(Bperp=rounded_Bperp)

            res.append(wpwp_kurt)

    wpwp_kurts = xr.combine_by_coords(res)
    if save_recomputed_wpwp_kurtosis:
        wpwp_kurts.to_netcdf(computed_wpwp_location, engine='h5netcdf')

else:
    wpwp_kurts = xr.open_dataset(computed_wpwp_location, engine='h5netcdf')
```

# Plot

```python
### Create figure layout

# Set a square aspect ratio for the figure
fig = plt.figure(figsize=(6,5), layout="constrained", dpi=300)

# Define the main grid layout with 3 rows - adjust height ratios to make top row taller
gs = GridSpec(3, 1, figure=fig, height_ratios=[0.5, 0.85, 0.5])

# First row: ax_avg (taller), ax_std, and ax_wp_wp
# Adjust the width ratios to give more space to ax_avg
gs_top = gs[0].subgridspec(1, 3, width_ratios=[2, 1, 2], wspace=0)

# Create a separate grid for ax_avg to control its height better
gs_avg = gs_top[0, 0].subgridspec(1, 1, hspace=0, wspace=0)
ax_avg = fig.add_subplot(gs_avg[0, 0])

# Other axes in top row
ax_std = fig.add_subplot(gs_top[0, 1])
ax_hist = fig.add_subplot(gs_top[0, 2])

# Second row: Time traces with histograms (unchanged internally)
gs_traces = gs[1].subgridspec(2, 3, width_ratios=[0.8, 0.3, 0.25], wspace=-1, hspace=-1)
ax_trace_max = fig.add_subplot(gs_traces[0, 0])
ax_trace_max_zoom = fig.add_subplot(gs_traces[0, 1])
ax_trace_hist_max = fig.add_subplot(gs_traces[0, 2])
ax_trace_min = fig.add_subplot(gs_traces[1, 0])
ax_trace_min_zoom = fig.add_subplot(gs_traces[1, 1])
ax_trace_hist_min = fig.add_subplot(gs_traces[1, 2])

# Third row: ax_hist and ax_autocorr
gs_bottom = gs[2].subgridspec(1, 2, width_ratios=[2, 1], wspace=0.15)
ax_autocorr = fig.add_subplot(gs_bottom[0, 0])
ax_wp_wp = fig.add_subplot(gs_bottom[0, 1])
;
```

```python
### Add schematic
png_path = "../schematics/xmpr_schematics.png"
png_image = Image.open(png_path)

# Add the SVG image to ax_avg with adjusted display
ax_avg.imshow(png_image)  # Use 'auto' to fill the available space
ax_avg.axis('off')  # Turn off axes and labels completely
```

```python
### Plot qd-qd map

# Define points for line paths through the detuning space
p1 = (-0.07,0.54)
p2 = (-0.03, 0)
p3 = (-0.09,-0.22)
p4 = (-0.08,-0.35)

grays_cmap = 'Greys_r'

# Plot kurtosis maps
kurtosis_ngng_plot_kwargs = {
        'vmin': min([k.quantile(0.0005,"Bperp").min() for k in kurts]).to_numpy(),
        'vmax': max([k.quantile(0.0005,"Bperp").max() for k in kurts]).to_numpy(),
        'cmap': grays_cmap
}

pcm_std1 = ax_std.pcolormesh(ds_zoom1.VQD1, ds_zoom1.VQD3,
                            kurts[1].quantile(0.0005,"Bperp"),
                            **kurtosis_ngng_plot_kwargs)
pcm_std2 = ax_std.pcolormesh(ds_zoom2.VQD1, ds_zoom2.VQD3,
                            kurts[2].quantile(0.0005,"Bperp"),
                            **kurtosis_ngng_plot_kwargs)
pcm_std3 = ax_std.pcolormesh(ds.VQD1, ds.VQD3,
                            kurts[0].quantile(0.0005,"Bperp"),
                            **kurtosis_ngng_plot_kwargs,
                            shading='auto')

# Add colorbar
fig.colorbar(pcm_std3, ax=ax_std, label='Kurtosis\n($\\mathrm{min}_{B_\perp}$)', ticks=[-0.2, -0.7, -1.2])

ax_std.set_ylabel('$\\Delta V_{\\rm QD3}$ [mV]')
ax_std.set_xlabel('$\\Delta V_{\\rm QD1}$ [mV]')

ax_std.set_xlim(-0.15, 0.05)

# We convert back the axes from detuning to voltage.
QD1_detuning_to_mv = 0.75
QD3_detuning_to_mv = 0.5
xlims_detuning = np.array(ax_std.get_xlim())
xlims_mV = xlims_detuning*QD1_detuning_to_mv
xlims_mV_centered = xlims_mV - xlims_mV.mean()
xlims_mV_ticks = np.array([-0.05,0.0, 0.05])
xlims_detuning_ticks = (xlims_mV_ticks + xlims_mV.mean())/QD1_detuning_to_mv
ax_std.set_xticks(xlims_detuning_ticks)
ax_std.set_xticklabels(xlims_mV_ticks)

ylims_detuning = np.array(ax_std.get_ylim())
ylims_mV = ylims_detuning*QD3_detuning_to_mv
ylims_mV_centered = ylims_mV - ylims_mV.mean()
ylims_mV_ticks = np.array([-0.2,0.0, 0.2])
ylims_detuning_ticks = (ylims_mV_ticks + ylims_mV.mean())/QD3_detuning_to_mv
ax_std.set_yticks(ylims_detuning_ticks)
ax_std.set_yticklabels(ylims_mV_ticks)

# Plot guides and markers
guide_color = 'k'
ax_std.plot([p1[0], p2[0]], [p1[1], p2[1]], '--', color=guide_color, linewidth=1.)
ax_std.plot([p2[0], p3[0]], [p2[1], p3[1]], '--', color=guide_color, linewidth=1.)
ax_std.plot([p3[0], p4[0]], [p3[1], p4[1]], '--', color=guide_color, linewidth=1.)
ax_std.plot(ds.VQD1[iVqd1], ds.VQD3[iVqd3], 'o', markersize=3,
           markeredgecolor="red", markerfacecolor="none")
```

```python
# ### Plot WP-WP map

scaled_wpwp_kurts = wpwp_kurts.assign_coords(
    DeltaVWP1=1e3*wpwp_kurts.DeltaVWP1,
    DeltaVWP2=1e3*wpwp_kurts.DeltaVWP2
).VQD1_signal


wpwp_plot_data = scaled_wpwp_kurts.min('Bperp').quantile(0.001, "detunings")
wpwp_plot_data.T.plot.imshow(ax=ax_wp_wp, x='DeltaVWP2', cmap=grays_cmap)
ax_wp_wp.set_title(None)
ax_wp_wp.set_xlabel("$\\Delta$ V$_{\\mathrm{WP2}}$ [mV]")
ax_wp_wp.set_ylabel("$\\Delta$ V$_{\\mathrm{WP1}}$ [mV]")
ax_wp_wp.images[0].colorbar.set_label('Kurtosis\n ($\\mathrm{min}_{V_{\\mathrm{QD1}}, V_{\\mathrm{QD3}}, B_\\perp}$)')
ax_wp_wp.images[0].colorbar.set_ticks([-0.5, -0.7, -0.9])

wpwp_nan_mask = np.isnan(wpwp_plot_data.data)

ax_wp_wp.set_aspect('equal')

delta_VWP1 = wpwp_plot_data.DeltaVWP1[-1] - wpwp_plot_data.DeltaVWP1[0]
delta_VWP2 = wpwp_plot_data.DeltaVWP2[-1] - wpwp_plot_data.DeltaVWP2[0]
dVWP1 = wpwp_plot_data.DeltaVWP1[1] - wpwp_plot_data.DeltaVWP1[0]
dVWP2 = wpwp_plot_data.DeltaVWP2[1] - wpwp_plot_data.DeltaVWP2[0]
rect = Rectangle((wpwp_plot_data.DeltaVWP2[0]-dVWP2, wpwp_plot_data.DeltaVWP1[0]-dVWP1), delta_VWP2+2*dVWP2, delta_VWP1+2*dVWP1,
    hatch='///', fill=False, edgecolor='red', linewidth=0, alpha=0.2, zorder=-20)
ax_wp_wp.add_patch(rect)
```

```python
### Plot flux dependent kurtosis

ax_hist.axvline(ds.Bperp[iB1], color=color1, linestyle="--")
ax_hist.axvline(ds.Bperp[iB2], color=color2, linestyle="--")
kurts[0].isel(VQD3=iVqd3,VQD1=iVqd1).T.plot(color="k", ax=ax_hist)
ax_hist.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x*1000:.0f}"))
ax_hist.set_ylabel("Kurtosis [$C_{\\mathrm{Q}}$]")
ax_hist.set_xlabel("B$_{\\perp}$ [mT]")
ax_hist.set_title("")
ax_hist.set_xticks([-0.002,-0.001,0,0.001,0.002]);
```

```python
from matplotlib.patches import ConnectionPatch

# Plot Time traces

# Plot max visibility timetrace
coarsened_trace1 = trace1.coarsen(time=timetrace_coarsening, boundary="trim").mean()
ax_trace_max.plot(1e3 * coarsened_trace1.time.to_numpy(), coarsened_trace1.to_numpy(),
                 color=color1, lw=timetrace_lw)
ax_trace_max.set_ylabel("$C_{\\mathrm{Q}}$ [aF]")
ax_trace_max.set_xlim(*(1e3 * coarsened_trace1.time.to_numpy()[[0,-1]]))
ax_trace_max.set_ylim(-200, 400)
ax_trace_max.set_xlim(0.2, 8)
ax_trace_min.set_xlim(0.2, 8)

cl = 6e-3
cr = 7e-3

ax_trace_max_zoom.plot(1e3 * coarsened_trace1.sel(time=slice(cl,cr)).coarsen(time=1).mean().time.to_numpy(), coarsened_trace1.sel(time=slice(cl,cr)).coarsen(time=1).mean().to_numpy(),
                 color=color1, linewidth=2*timetrace_lw)
ax_trace_max_zoom.set_yticklabels([])
ax_trace_max_zoom.set_ylim(*ax_trace_max.get_ylim())

# Plot min visibility timetrace
coarsened_trace2 = trace2.coarsen(time=timetrace_coarsening, boundary="trim").mean()
ax_trace_min.plot(1e3 * coarsened_trace2.time.to_numpy(), coarsened_trace2.to_numpy(),
                 color=color2, linewidth=timetrace_lw)
ax_trace_min.set_ylabel("$C_{\\mathrm{Q}}$ [aF]")
ax_trace_min.set_ylim(*ax_trace_max.get_ylim())
ax_trace_min.set_xlabel("Time [ms]")

ax_trace_min_zoom.plot(1e3 * coarsened_trace2.sel(time=slice(cl,cr)).coarsen(time=1).mean().time.to_numpy(), coarsened_trace2.sel(time=slice(cl,cr)).coarsen(time=1).mean().to_numpy(),
                 color=color2, linewidth=2*timetrace_lw)
ax_trace_min_zoom.set_yticklabels([])
ax_trace_min_zoom.set_ylim(*ax_trace_max.get_ylim())

ax_trace_max.set_xticklabels([])
ax_trace_max_zoom.set_xticklabels([])

ax_trace_max_zoom.set_xlim(cl*1e3, cr*1e3)
ax_trace_min_zoom.set_xlim(cl*1e3, cr*1e3)

ax_trace_min_zoom.set_xlabel("Time [ms]")

zorder_value = 10
offset = 20
frame_offset = -30

# Draw frames first
for ax, c in zip([ax_trace_max, ax_trace_min], ("k", 'k')):
    rect = Rectangle((cl*1e3, ax.get_ylim()[0]+frame_offset),
                    (cr-cl)*1e3, ax.get_ylim()[1]-frame_offset*2-ax.get_ylim()[0],
                    linewidth=1.0, edgecolor=c, facecolor='none',
                    linestyle='--', zorder=zorder_value, clip_on=False)
    ax.add_patch(rect)

# Make sure all drawing is done before adding connections
fig.canvas.draw()

# Add connecting lines between main plot and zoomed section with high zorder to ensure they're on top

# For the first trace
con1 = ConnectionPatch(xyA=(cr*1e3, ax_trace_max.get_ylim()[1]-offset),
                      xyB=(cl*1e3, ax_trace_max_zoom.get_ylim()[1]-offset),
                      coordsA="data", coordsB="data",
                      axesA=ax_trace_max, axesB=ax_trace_max_zoom,
                      color=color1, linestyle=":", linewidth=0.5, zorder=zorder_value)
ax_trace_max.add_artist(con1)

con1b = ConnectionPatch(xyA=(cr*1e3, ax_trace_max.get_ylim()[0]+offset),
                      xyB=(cl*1e3, ax_trace_max_zoom.get_ylim()[0]+offset),
                      coordsA="data", coordsB="data",
                      axesA=ax_trace_max, axesB=ax_trace_max_zoom,
                      color=color1, linestyle=":", linewidth=0.5, zorder=zorder_value)
ax_trace_max.add_artist(con1b)

# For the second trace
con2 = ConnectionPatch(xyA=(cr*1e3, ax_trace_min.get_ylim()[1]-offset),
                      xyB=(cl*1e3, ax_trace_min_zoom.get_ylim()[1]-offset),
                      coordsA="data", coordsB="data",
                      axesA=ax_trace_min, axesB=ax_trace_min_zoom,
                      color=color2, linestyle=":", linewidth=0.5, zorder=zorder_value)
ax_trace_min.add_artist(con2)

con2b = ConnectionPatch(xyA=(cr*1e3, ax_trace_min.get_ylim()[0]+offset),
                      xyB=(cl*1e3, ax_trace_min_zoom.get_ylim()[0]+offset),
                      coordsA="data", coordsB="data",
                      axesA=ax_trace_min, axesB=ax_trace_min_zoom,
                      color=color2, linestyle=":", linewidth=0.5, zorder=zorder_value)
ax_trace_min.add_artist(con2b)

# Force the figure to redraw again to ensure connections are visible
fig.canvas.draw()
```

```python
### Plot Histograms

# set n_bootstrap to 1000 or so for error bars
n_bootstrap = 1

# Plot max visibility histogram
bins = np.linspace(-273, 465, 50)
cnts_max, bins_max, _ = ax_trace_hist_max.hist(coarsened_trace1.to_numpy(), orientation='horizontal', bins=bins,
                       density=False, color=color1)
pdf_to_counts_max = (len(coarsened_trace1)*np.diff(bins_max).mean())

# Plot min visibility histogram
cnts_min, bins_min, _ = ax_trace_hist_min.hist(coarsened_trace2.to_numpy(), orientation='horizontal', bins=bins,
                       density=False, color=color2)
pdf_to_counts_min = (len(coarsened_trace2)*np.diff(bins_min).mean())

# Format histograms
ax_trace_hist_max.set_yticks(ax_trace_max.get_yticks())
ax_trace_hist_max.set_yticklabels([])
ax_trace_hist_max.set_ylim(*ax_trace_max.get_ylim())

ax_trace_hist_min.set_yticks(ax_trace_min.get_yticks())
ax_trace_hist_min.set_yticklabels([])
ax_trace_hist_min.set_ylim(*ax_trace_min.get_ylim())

x = np.linspace(ax_trace_max.get_ylim()[0], ax_trace_max.get_ylim()[1], 1000)

# Calculate the combined PDF
combined_pdf1, indiv_pdf1 = fit_and_sum_gaussians(coarsened_trace1.to_numpy(), x, label='Max Trace', n_bootstrap=n_bootstrap)
combined_pdf2, indiv_pdf2 = fit_and_sum_gaussians(coarsened_trace2.to_numpy(), x, label='Min Trace', n_bootstrap=n_bootstrap)

ax_trace_hist_max.plot(combined_pdf1*pdf_to_counts_max, x, color='red', lw=1, linestyle='--')
ax_trace_hist_min.plot(combined_pdf2*pdf_to_counts_min, x, color='red', lw=1, linestyle='--')
for ipdf in indiv_pdf1:
    ax_trace_hist_max.plot(ipdf*pdf_to_counts_max, x, color='tab:orange', alpha=1, lw=1, linestyle='--')
for ipdf in indiv_pdf2:
    ax_trace_hist_min.plot(ipdf*pdf_to_counts_min, x, color='tab:orange', alpha=1, lw=1, linestyle='--')

# Tie and remove x-axis from hist
ax_trace_hist_max.set_xticks(ax_trace_hist_min.get_xticks())
ax_trace_hist_max.set_xticklabels([])
ax_trace_hist_min.set_xlim(ax_trace_hist_min.get_xlim()) # Need to set xlim after ticks...
ax_trace_hist_max.set_xlim(ax_trace_hist_min.get_xlim())
ax_trace_hist_min.set_xlabel("Counts")
# takes 1s without bootstrapping
# and 1 min with 1000pt bootstrapping
```

```python
### Plot correlation functions
extract_cross_lags_above_noise(trace2.time, trace2.data,
                                     [4.9], fit_f, max_lag=80, plot=True, ax=ax_autocorr, ax_hist=None,
                                     minimum_lag=1, max_lag_fit=5, color=color2, do_fit=False)

amps, taus, fit_errs = extract_cross_lags_above_noise(trace1.time, trace1.data,
                                     [4.9], fit_f, max_lag=80, plot=True, ax=ax_autocorr, ax_hist=None,
                                     minimum_lag=1, max_lag_fit=15, color=color1, do_fit=True)

# Set correlation function plot styling
ax_autocorr.set_ylabel("$\\langle C_{\\mathrm{Q}}(t) C_{\\mathrm{Q}}(t') \\rangle\\,\\, \\mathrm{[aF^2]}$")
ax_autocorr.set_yscale("log")
ax_autocorr.set_ylim(1e2, 8.0e4)
ax_autocorr.set_xlabel("$t-t'$ [$\\mathrm{\\mu s}$]")
taus[0]
```

```python
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
```

```python
apply_sublabels(
    [ax_avg, ax_std, ax_hist, ax_trace_max, ax_trace_hist_max, ax_trace_min, ax_trace_hist_min, ax_autocorr, ax_wp_wp],
    x=[-5, -40, -35, -40, -5, -40, -5, -40, -50],
    y=10
)
```

```python
print("DeltaCQ", "stdCQ")
for delta_err in [0, -1, 1]:
    print(
        (4*(amps[0] + delta_err*fit_errs[0][0]))**0.5, (np.std(trace1.data)**2 - amps[0] - delta_err*fit_errs[0][0])**0.5
    )
```

```python
fig
```

```python
fig.savefig(FIGURE_OUTPUT_FOLDER / "xmpr.pdf", bbox_inches='tight', dpi=300)
```
