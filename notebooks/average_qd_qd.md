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

# Add the parent directory to sys.path
sys.path.append("..")
```

```python
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
```

```python
from paths import CONVERTED_DATA_FOLDER, FIGURE_OUTPUT_FOLDER

ds = xr.load_dataset(CONVERTED_DATA_FOLDER / "xmpr/xmpr_Cq.h5", engine='h5netcdf')

ds_zoom1 = xr.load_dataset(CONVERTED_DATA_FOLDER / "xmpr/xmpr_zoomout_1_Cq.h5", engine='h5netcdf')
ds_zoom2 = xr.load_dataset(CONVERTED_DATA_FOLDER / "xmpr/xmpr_zoomout_2_Cq.h5", engine='h5netcdf')

iVqd3 = 5
iVqd1 = 2
```

```python
fs = 8
lw = 1
# # SB: Temporary workaround without latex on cluster.
plt.rcParams.update({
    'font.size': fs,
    'font.family': 'STIXGeneral',
    'mathtext.fontset' : "cm",
    'xtick.labelsize': fs,
    'ytick.labelsize': fs,
    'axes.labelsize': fs,
})

# plt.rcParams.update({
#     'font.size': fs,
#     'font.family': 'serif',
#     'xtick.labelsize': fs,
#     'ytick.labelsize': fs,
#     'text.usetex': True,
#     'text.latex.preamble': '\\usepackage{amsmath}',
#     'axes.labelsize': fs,
# })
```

# Plot

```python
### Plot qd-qd map
fig = plt.figure(figsize=(0.7*3.5,0.7*3.))
ax = plt.gca()

# Define points for line paths through the detuning space
p1 = (-0.07,0.54)
p2 = (-0.03, 0)
p3 = (-0.09,-0.22)
p4 = (-0.08,-0.35)

# Define data ranges for QD-QD scans
v1_min, v1_max = ds_zoom1.VQD1.min().item(), ds_zoom1.VQD1.max().item()
v3_min, v3_max = ds_zoom1.VQD3.min().item(), ds_zoom1.VQD3.max().item()
v1_min2, v1_max2 = ds_zoom2.VQD1.min().item(), ds_zoom2.VQD1.max().item()
v3_min2, v3_max2 = ds_zoom2.VQD3.min().item(), ds_zoom2.VQD3.max().item()
v1_min3, v1_max3 = ds.VQD1.min().item(), ds.VQD1.max().item()
v3_min3, v3_max3 = ds.VQD3.min().item(), ds.VQD3.max().item()

grays_cmap = 'Greys_r'

# Plot on the existing ax_std axis using plt
pcm_avg1 = plt.pcolormesh(ds_zoom1.VQD1, ds_zoom1.VQD3,
                        ds_zoom1.Cq.mean(["time","Bperp"]))
pcm_avg2 = plt.pcolormesh(ds_zoom2.VQD1, ds_zoom2.VQD3,
                        ds_zoom2.Cq.mean(["time","Bperp"]))
pcm_avg3 = plt.pcolormesh(ds.VQD1, ds.VQD3,
                        ds.Cq.mean(["time","Bperp"]),
                        shading='auto')

# Add outlines of dataranges for each dataset
ax.plot([v1_min, v1_max, v1_max, v1_min, v1_min],
                [v3_min, v3_min, v3_max, v3_max, v3_min],
                'blue', linewidth=1)
ax.plot([v1_min2, v1_max2, v1_max2, v1_min2, v1_min2],
                [v3_min2, v3_min2, v3_max2, v3_max2, v3_min2],
                'blue', linewidth=1)
ax.plot([v1_min3, v1_max3, v1_max3, v1_min3, v1_min3],
                [v3_min3, v3_min3, v3_max3, v3_max3, v3_min3],
                'blue', linewidth=1)

# Add colorbar
fig.colorbar(pcm_avg3, ax=plt.gca(), label='Average $C_{\mathrm{Q}}$ [aF]\n(over time,${B_\perp}$)')

ax.set_ylabel('$\\Delta V_{\\rm QD3}$ [mV]')
ax.set_xlabel('$\\Delta V_{\\rm QD1}$ [mV]')

# Plot guides and markers
guide_color = 'tab:purple'
plt.plot([p1[0], p2[0]], [p1[1], p2[1]], '--', color=guide_color, linewidth=1.)
plt.plot([p2[0], p3[0]], [p2[1], p3[1]], '--', color=guide_color, linewidth=1.)
plt.plot([p3[0], p4[0]], [p3[1], p4[1]], '--', color=guide_color, linewidth=1.)
plt.plot(ds.VQD1[iVqd1], ds.VQD3[iVqd3], 'o', markersize=3,
       markeredgecolor="red", markerfacecolor="none")

# Set rasterized=True for all pcolormesh objects
pcm_avg1.set_rasterized(True)
pcm_avg2.set_rasterized(True)
pcm_avg3.set_rasterized(True)

# We convert back the axes from detuning to voltage.
QD1_detuning_to_mv = 0.75
QD3_detuning_to_mv = 0.5
xlims_detuning = np.array(plt.gca().get_xlim())
xlims_mV = xlims_detuning*QD1_detuning_to_mv
xlims_mV_centered = xlims_mV - xlims_mV.mean()
xlims_mV_ticks = np.array([-0.1,0.0,0.1])
xlims_detuning_ticks = (xlims_mV_ticks - 0.05*QD1_detuning_to_mv)/QD1_detuning_to_mv
ax.set_xticks(xlims_detuning_ticks)
ax.set_xticklabels(xlims_mV_ticks)

ylims_detuning = np.array(plt.gca().get_ylim())
ylims_mV = ylims_detuning*QD3_detuning_to_mv
ylims_mV_centered = ylims_mV - ylims_mV.mean()
ylims_mV_ticks = np.array([-0.2,0.0, 0.2])
ylims_detuning_ticks = (ylims_mV_ticks + ylims_mV.mean())/QD3_detuning_to_mv
ax.set_yticks(ylims_detuning_ticks)
ax.set_yticklabels(ylims_mV_ticks)

# Save the figure with vector text and rasterized image
plt.savefig(FIGURE_OUTPUT_FOLDER / "average_qd_qd.pdf", bbox_inches='tight', dpi=300)
```
