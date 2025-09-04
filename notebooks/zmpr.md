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
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import xarray as xr

from analysis_code.error_metrics import get_error_statistics, classify_gmm
from analysis_code.timetrace_classification import _dwell_time, threshold_traces, prepare_plot_dwells_bayesian, plot_dwells_bayesian
```

```python
from PIL import Image
from analysis_code.plotting_helpers import (
    COLUMNWIDTH,
    add_subfig_labelxy,
    savefig
)
```

```python
from paths import CONVERTED_DATA_FOLDER, FIGURE_OUTPUT_FOLDER

Cq_Z = xr.load_dataset(CONVERTED_DATA_FOLDER / "zmpr/zmpr_Cq.h5", engine='h5netcdf')

if "rescaled_to_milli" not in Cq_Z.attrs:
    Cq_Z.VQDL.data[:] = Cq_Z.VQDL.data[:]*1e3
    Cq_Z.Bperp.data[:] = Cq_Z.Bperp.data[:]*1e3

    Cq_Z.attrs["rescaled_to_milli"] = True
```

```python
# Load the schematic image
png_path = "../schematics/zmpr_schematics.png"
png_image = Image.open(png_path)
```

### Calculate plot quantities

```python
Cq_Z["kurtosis"] = xr.apply_ufunc(
    lambda x: np.mean((x - np.mean(x))**4) / np.std(x)**4 -3 ,
    Cq_Z.Cq,
    input_core_dims=[["time"]],
    output_core_dims=[[]],
    vectorize=True,
    dask="parallelized",
    output_dtypes=[np.float64],
)
```

```python
dwell_up = []
dwell_down = []

for i in range(len(Cq_Z.Bperp)):
    for j in range(len(Cq_Z.VQD4)):
        for k in range(len(Cq_Z.VQDL)):
            itrace = Cq_Z.isel(VQD4=j,VQDL=k,Bperp=i).coarsen(time=10, boundary="trim").mean()
            if itrace["kurtosis"]< -1.0:
                gmm = classify_gmm(itrace.Cq.to_numpy(), return_gmm=True)[1]
                m1,m2 = np.sort(gmm.means_[:,0])
                itrace["gmm.mean1.Cq"]=m1
                itrace["gmm.mean2.Cq"]=m2

                iout = threshold_traces(itrace, "Cq", r=0.1)
                idwell_up, idwell_down = _dwell_time(itrace.time, iout["thresh.digital_bool.Cq"])
                dwell_up.append(idwell_up)
                dwell_down.append(idwell_down)

dwell_up = np.concatenate(dwell_up)
dwell_down = np.concatenate(dwell_down)
# ~40s
```

```python
dt = float(np.mean(np.diff(Cq_Z.time.to_numpy())))
T = float(Cq_Z.time.to_numpy()[-1])
dt, T
```

```python
model_up = prepare_plot_dwells_bayesian(dwell_up, sample_rate=dt, trace_len=T)
model_down = prepare_plot_dwells_bayesian(dwell_down, sample_rate=dt, trace_len=T)
# ~1min
```

```python
erra = get_error_statistics(
    Cq_Z.Cq.coarsen(time=2,boundary="trim").mean(),
    parallel=True
)
# ~15s
```

## Generate Figure

```python
i_Bperp = 30+12
i_VQDL = 8
i_VQD4 = 0

trace = Cq_Z.Cq.isel(
    VQD4=i_VQD4,
    VQDL=i_VQDL,
    Bperp=i_Bperp
)
gmm = classify_gmm(trace.to_numpy(), return_gmm=True)[1]

trace["gmm.mean1.Cq"] = gmm.means_[0, 0]
trace["gmm.mean2.Cq"] = gmm.means_[1, 0]
trace["gmm.std1.Cq"] = np.sqrt(gmm.covariances_[0, 0])
trace["gmm.std2.Cq"] = np.sqrt(gmm.covariances_[1, 0])

out = threshold_traces(trace.to_dataset(), "Cq", r=0.1)
```

```python
print("SNR=",np.abs(trace["gmm.mean1.Cq"]-trace["gmm.mean2.Cq"])/(trace["gmm.std1.Cq"]+trace["gmm.std2.Cq"]))
```

```python
# fs = 8
# SB: Temporary workaround without latex on cluster.
# plt.rcParams.update({
#     'font.size': fs,
#     'font.family': 'STIXGeneral',
#     'mathtext.fontset' : "cm",
#     'xtick.labelsize': fs,
#     'ytick.labelsize': fs,
#     'axes.labelsize': fs,
# })

# plt.rcParams.update({
#     'font.size': fs,
#     'font.family': 'serif',
#     'xtick.labelsize': fs,
#     'ytick.labelsize': fs,
#     'text.usetex': True,
#     'text.latex.preamble': '\\usepackage{amsmath}',
#     'axes.labelsize': fs,
# })


font_size = 8
plt.rcParams.update({
    "font.size": font_size,
    "axes.labelsize": font_size,
    "axes.titlesize": font_size,
    "xtick.labelsize": font_size,
    "ytick.labelsize": font_size,
    "legend.fontsize": font_size,
    'font.family': 'STIXGeneral',
    'mathtext.fontset' : "cm",
})
```

```python
lw = 0.8

fig = plt.figure(figsize=(COLUMNWIDTH, COLUMNWIDTH*4.5/3.5), layout="constrained", dpi=600)

gs = GridSpec(3, 2, figure=fig, width_ratios=[0.5, 0.5], height_ratios=[0.28, 0.25, 0.2])

ax_schematic_null = fig.add_subplot(gs[0, :])
ax_z_2d = fig.add_subplot(gs[1, 0])
ax_z_hist = fig.add_subplot(gs[1, 1])
ax_z_dwell = fig.add_subplot(gs[2, 1])
ax_z_tr = fig.add_subplot(gs[2, 0])

### Schematic ###

ax_schematic_null.axis('off')
ax_schematic = fig.add_axes([0.06, 0.71, 0.930, 0.3]) # [left, bottom, width, height]
# Add the SVG image to ax_avg with adjusted display
ax_schematic.imshow(png_image)
ax_schematic.axis('off')

### Kurtosis 2D Map ###

ds_kurt = Cq_Z["kurtosis"].isel(VQD4=i_VQD4)

ds_kurt.plot.imshow(
    x="Bperp",
    cbar_kwargs={
        "label": r"Kurtosis [$C_\mathrm{Q}$]",
        "location": "top",
        "pad": -0.24,
    },
    ax=ax_z_2d,
    cmap="Greys_r",
    vmax=0,
    vmin=-1.5,
)

ax_z_2d.axhline(Cq_Z.VQDL[i_VQDL], color="red", linestyle="--",lw=lw)

ax_z_2d.set_xlabel(r"$B_\perp$ [mT]")
ax_z_2d.set_xticks(np.arange(-2, 2.1, 1))
ax_z_2d.set_ylabel(r"$\Delta V_\mathrm{QDL}$ [mV]")
ax_z_2d.set_title("")

ax_z_2d.plot(
    (Cq_Z.Bperp[i_Bperp]),
    Cq_Z.VQDL[i_VQDL],
    'o',
    markersize=5,
    markeredgecolor="tab:red",
    markerfacecolor="none",
)

### Kurtosis Flux Histogram ###

cut = (Cq_Z.Cq).isel(
    VQD4=i_VQD4,
    VQDL=i_VQDL)

bins = np.linspace(-200, +600, 100)
Z = np.zeros((len(Cq_Z.Bperp), len(bins)-1))
for Bperp in range(len(Cq_Z.Bperp)):
    Z[Bperp,:] = np.histogram(cut.isel(Bperp=Bperp), bins=bins)[0]

# Plot histogram
ax_z_hist.imshow(Z.T, extent=[(Cq_Z.Bperp)[0], (Cq_Z.Bperp)[-1], bins[0], bins[-1]],
               aspect="auto", origin="lower", cmap="Greys")
ax_z_hist.set_xlabel(r"$B_\perp$ [mT]")
ax_z_hist.set_ylabel(r"$C_{\mathrm{Q}}$ [aF]")
ax_z_hist.set_title("")
ax_z_hist.set_xticks(np.arange(-2, 2.1, 1))
# Add colorbar to histogram plot
ax_z_hist.figure.colorbar(
    ax_z_hist.get_images()[0],
    ax=ax_z_hist,
    label=r"Counts",
    location="top",
    pad=-0.24,
)

ax_z_hist.axvline(Cq_Z.Bperp[i_Bperp], color="red", linestyle="--",lw=lw)

### Time Trace ###

Cq_Z["Cq"].isel(
    VQD4=i_VQD4,
    VQDL=i_VQDL,
    Bperp=i_Bperp
).coarsen(time=1, boundary="trim").mean().plot(ax=ax_z_tr, color="black", linewidth=0.5)
ax_z_tr.set_title("")
ax_z_tr.set_ylabel(r"$C_\mathrm{Q}$ [aF]")
ax_z_tr.set_xticks(np.arange(0, 0.11, 0.05))
ax_z_tr.set_xlabel("Time [s]")

for ax,lbl,xi,yi in zip([ax_schematic_null, ax_z_2d, ax_z_hist,  ax_z_tr,ax_z_dwell], ["a", "b", "c", "d","e"], [-0.155, -0.42, -0.42, -0.42,-0.42], [0.93, 1.345, 1.345, 1.345, 1.345]):
    add_subfig_labelxy(
        ax,
        label=lbl,
        x=xi,
        y=yi,
        fontsize=9,
    )

### Dwell Time Plots ###

label_up = r"{\tau}_{Z\!\downarrow\!}"
label_dn = r"{\tau}_{Z\!\uparrow\!}"
for dwell, model, lbl,c  in [(dwell_up, model_up, label_up, "tab:blue"), (dwell_down, model_down, label_dn, "tab:red")]:
    plot_dwells_bayesian(
        ax_z_dwell,
        dwell,
        model_fit=model,
        label=lbl,
        color=c,
        bin_scale=100,
    )

ax_z_dwell.set_xlim(0, 50)
ax_z_dwell.set_xticks(np.arange(0, 51, 25))
ax_z_dwell.set_ylim(8, None)

ax_z_dwell.legend(
    frameon=False,
    loc="upper right",
    bbox_to_anchor=(1.05, 1.46),
    fontsize=7,
    labelspacing=0.25,   # less vertical space between labels
    handlelength=1.5,    # length of legend lines
    handletextpad=0.4,   # space between line and text
)

savefig(fig, FIGURE_OUTPUT_FOLDER / "zmpr.pdf")
plt.show()
```
