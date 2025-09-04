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
import numpy as np
import xarray as xr
from scipy.special import erf

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from analysis_code.error_metrics import _error_stats, get_error_statistics

from analysis_code.plotting_helpers import add_subfig_labelxy
```

```python
from paths import CONVERTED_DATA_FOLDER, FIGURE_OUTPUT_FOLDER

Cq_X = xr.load_dataset(CONVERTED_DATA_FOLDER / "xmpr/xmpr_Cq.h5", engine='h5netcdf')
Cq_Z = xr.load_dataset(CONVERTED_DATA_FOLDER / "zmpr/zmpr_Cq.h5", engine='h5netcdf')

Cq_X = Cq_X.Cq + 1j * Cq_X.iCq
Cq_Z = Cq_Z.Cq + 1j * Cq_Z.iCq
```

```python
fs = 8
lw = 1
plt.rcParams.update({
    'font.size': fs,
    'font.family': 'STIXGeneral',
    'mathtext.fontset' : "cm",
    'xtick.labelsize': fs,
    'ytick.labelsize': fs,
    'axes.labelsize': fs,
    'text.usetex': False,
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

## Select the traces

```python
%%time
erra = get_error_statistics(
    Cq_X.coarsen(time=2, boundary="trim").mean(),
    parallel=True
)
```

```python
iVqd3 = 5
iVqd1 = 2

iB = 24

erra.err_a.sel(
    err_quantile="median"
).isel(
    VQD3=iVqd3,
    VQD1=iVqd1,
   ).plot()
plt.axvline(erra.Bperp[iB], color="red", linestyle=":")

trace = Cq_X.isel(
    VQD3=iVqd3,
    VQD1=iVqd1,
).isel(Bperp=iB).squeeze()
```

```python
i_BperpZ = 30+12
i_VQDL = 8
i_VQD4 = 0

traceZ = 1e5*Cq_Z.isel(VQD4=i_VQD4, VQDL=i_VQDL, Bperp=i_BperpZ)
```

## Generate error matrices for panels a and c

```python
cXerra = 2
coarsened_trace = trace.coarsen(time=cXerra, boundary="trim").mean()

esX = get_error_statistics(coarsened_trace)
mtxX = np.array([[esX.isel(err_quantile=1).P00.values, esX.isel(err_quantile=1).P10.values],
[esX.isel(err_quantile=1).P01.values,esX.isel(err_quantile=1).P11.values]])

# assignment of GMM components is arbitrary, fix for determinism
if mtxX[0,0] > mtxX[1,1]:
    mtxX = mtxX[::-1, ::-1]
```

```python
cZerra = 1
coarsened_traceZ = traceZ.coarsen(time=cZerra, boundary="trim").mean()

esZ = get_error_statistics(coarsened_traceZ)
mtxZ = np.array([[esZ.isel(err_quantile=1).P00.values, esZ.isel(err_quantile=1).P10.values],
[esZ.isel(err_quantile=1).P01.values,esZ.isel(err_quantile=1).P11.values]])

# assignment of GMM components is arbitrary, fix for determinism
if mtxZ[0,0] > mtxZ[1,1]:
    mtxZ = mtxZ[::-1, ::-1]
```

## Get erra while coarsening

```python
cX = np.arange(1, 11)
yminX, yX, ymaxX = [], [], []

for coarsen in cX:
    tr_ = trace.coarsen(time=coarsen, boundary="trim").mean().to_numpy()
    erra_ = _error_stats(tr_)[0]
    yX += [erra_[1]]
    yminX += [erra_[0]]
    ymaxX += [erra_[2]]

dtX = trace.time[1].values-trace.time[0].values
```

```python
cZ = np.arange(1, 21)
yminZ, yZ, ymaxZ = [], [], []

for coarsen in cZ:
    tr_ = traceZ.coarsen(time=coarsen, boundary="trim").mean().to_numpy()
    erra_ = _error_stats(tr_)[0]
    yZ += [erra_[1]]
    yminZ += [erra_[0]]
    ymaxZ += [erra_[2]]

dtZ = traceZ.time[1].values-traceZ.time[0].values
```

## Model for err_a

```python
def erra_theory_curve(τ1, c_fit, Tlife, SNR_rate):
    return 0.5*(1-(
        np.exp(-τ1*c_fit/Tlife)*erf(SNR_rate * np.sqrt(τ1*c_fit) / np.sqrt(2))
    )**2)

# Model Parameters for Z and X measurements
τ1Z = dtZ
SNR_rateZ = 6.0 / np.sqrt(τ1Z)
TlifeZ = 12.4e-3

τ1X = 1e-6
SNR_rateX = 1.35 / np.sqrt(τ1X)
TlifeX = 14.5e-6
```

## Plot

```python
fig, ax = plt.subplots(nrows=2,ncols=2, figsize=(3.5,3), width_ratios=[0.4, 0.6], rasterized=True)

dividerX = make_axes_locatable(ax[0,0])
caxX = dividerX.append_axes("right",size="5%", pad=0.05)
dividerZ = make_axes_locatable(ax[1,0])
caxZ = dividerZ.append_axes("right", size="5%", pad=0.05)

# Show X error Matrix
imgX = ax[0,0].imshow(mtxX, vmin=0., vmax=1., cmap="Spectral")
plt.colorbar(imgX, cax=caxX, label="${P}(\\mathcal{M}_r^X | \\mathcal{M}_s^X)$")
for row in range(2):
    for col in range(2):
        ax[0,0].annotate(f"{mtxX[row,col]:.2f}", (row, col), ha="center", va="center", c="white" if mtxX[row,col] < 0.5 else "black")

# Show Z error Matrix
imgZ = ax[1,0].imshow(mtxZ, vmin=0., vmax=1., cmap="Spectral")
plt.colorbar(imgZ, cax=caxZ, label="${P}(\\mathcal{M}_r^Z | \\mathcal{M}_s^Z)$")
for row in range(2):
    for col in range(2):
        ax[1,0].annotate(f"{mtxZ[row,col]:.3f}", (row, col), ha="center", va="center", c="white")

# Plot X and Z erra from classified data vs coarsening
ax[0,1].scatter(cX * dtX * 1e6, yX, s=15)
ax[0,1].fill_between(cX * dtX * 1e6, yminX, ymaxX, alpha=0.2,color="C0")

ax[1,1].scatter(cZ * dtZ * 1e6, yZ, s=15)
ax[1,1].fill_between(cZ * dtZ * 1e6, yminZ, ymaxZ, alpha=0.2,color="C0")

c_fitX = np.linspace(1, 10, 1000)
c_fitZ = np.linspace(0.1, 20, 1000)

# X erra vs integration time theory curve
ax[0,1].plot(τ1X*c_fitX * 1e6, erra_theory_curve(τ1X, c_fitX, TlifeX, SNR_rateX), color="C1")
ax[0,1].axvline(dtX*cXerra*1e6,ls="--",c="k")

# Z erra vs integration time theory curve
ax[1,1].plot(τ1Z*c_fitZ * 1e6, erra_theory_curve(τ1Z, c_fitZ, TlifeZ, SNR_rateZ), color="C1")
ax[1,1].axvline(dtZ*cZerra*1e6,ls="--",c="k")

# Labels and ticks
ax[0,0].set_xticks([0., 1.], ["$\\mathcal{M}_+^X$", "$\\mathcal{M}_-^X$"])
ax[0,0].set_yticks([0., 1.], ["$\\mathcal{M}_+^X$", "$\\mathcal{M}_-^X$"])

ax[1,0].set_xticks([0., 1.], ["$\\mathcal{M}_+^Z$", "$\\mathcal{M}_-^Z$"])
ax[1,0].set_yticks([0., 1.], ["$\\mathcal{M}_+^Z$", "$\\mathcal{M}_-^Z$"])

ax[0,1].set_xlabel("Integration time [$\\mathrm{\\mu s}$]")
ax[0,1].set_ylabel("$\\mathrm{err}_a^X$")
ax[0,1].set_ylim(0.05,0.55)
ax[0,1].set_yscale("log")
ax[0,1].set_yticks([0.1,0.5])
ax[0,1].set_xlim(1,10)
ax[0,1].set_xticks([1,5,10])
ax[0,1].set_yticklabels([0.1,0.5])

ax[1,1].set_xlabel("Integration time [$\\mathrm{\\mu s}$]")
ax[1,1].set_ylabel("$\\mathrm{err}_a^Z$")
ax[1,1].set_ylim(0.001,0.55)
ax[1,1].set_xlim(1,400)
ax[1,1].set_xticks([1,200,400])
ax[1,1].set_yscale("log")

add_subfig_labelxy(ax[0,0],"a",y=1.3)
add_subfig_labelxy(ax[0,1],"b",y=1.23)
add_subfig_labelxy(ax[1,0],"c",y=1.3)
add_subfig_labelxy(ax[1,1],"d",y=1.23)

fig.tight_layout()
fig.savefig(FIGURE_OUTPUT_FOLDER / "error_metrics.pdf", dpi=300)
```

```python
SNR_rateZ*np.sqrt(1e-6)
```

```python
SNR_rateX*np.sqrt(1e-6)
```
