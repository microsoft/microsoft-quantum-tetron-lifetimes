---
jupyter:
  jupytext:
    formats: ipynb,md
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
import sys
sys.path.append("..")
```

```python
from copy import deepcopy

from matplotlib import pyplot as plt
import matplotlib.patches as patches

import numpy as np
import xarray as xr
```

```python
from analysis_code.plotting_helpers import font_size as fs, COLUMNWIDTH
```

# Load and prepare data:
## Experimental data

```python
from paths import CONVERTED_DATA_FOLDER, FIGURE_OUTPUT_FOLDER

exp_cq = dict(
    qd_1 = xr.load_dataset(CONVERTED_DATA_FOLDER / "qdmzm/qdmzm_qd1_Cq.h5", engine='h5netcdf'),
    qd_3 = xr.load_dataset(CONVERTED_DATA_FOLDER / "qdmzm/qdmzm_qd3_Cq.h5", engine='h5netcdf'),
)
```

```python
# Depending on the exact device configuration where the RF calibration curve was taken, there can be an offset in the Cq values
# As we define Cq=0 in the Coulomb valleys (any residual Cq in that configuration can be lumped with parasitics) we offset the data.
# Since most of the dataset should be in the valley, we take the position of the maximum of the histogram as the offset value.
offsets = dict()
for qd in exp_cq.keys():
    cnts, bins = np.histogram(exp_cq[qd].Cq, bins=41, density=True)
    xs = (bins[1:] + bins[:-1])/2
    offsets[qd] = xs[np.argmax(cnts)]
```

```python
# Rescale coordinates to mV and offset Cq to be zero in the Coulomb valleys
CQ1=deepcopy(exp_cq['qd_1'].Cq) - offsets['qd_1']
CQ3=deepcopy(exp_cq['qd_3'].Cq) - offsets['qd_3']
CQ1 = CQ1.assign_coords(VQD1=1e3*CQ1.VQD1.data, VQD3=1e3*CQ1.VQD3.data)
CQ3 = CQ3.assign_coords(VQD1=1e3*CQ3.VQD1.data, VQD3=1e3*CQ3.VQD3.data)
```

```python
# Estimated lever arms of the plunger gates
lever_arms = dict(qd_1=0.15, qd_3=0.25)

# Effective lever arms accounting for the sweep of both the depletion gate and the plunger gate.
effective_lever_arms = dict(
    qd_1=lever_arms['qd_1']*1.1,
    qd_3=lever_arms['qd_3']*1.346
)

def convert_mV_to_detuning_ueV(Vs, lever_arm, V_shift):
    return  (Vs - V_shift)*1e3*lever_arm
```

```python
# Prepare zoom-ins
dV_qd1=0.35
dV_qd3=0.22
p0=(0.76, -0.01)

# CQ1 zoom
V1_mask = np.logical_and(CQ1.VQD1>=p0[0]-dV_qd1, CQ1.VQD1<=p0[0]+dV_qd1)
V3_mask = np.logical_and(CQ1.VQD3>=p0[1]-dV_qd3, CQ1.VQD3<=p0[1]+dV_qd3)
CQ1_zoomin = deepcopy(CQ1.where(V1_mask).dropna(dim='VQD1').where(V3_mask).dropna(dim='VQD3'))

# CQ3 zoom
V1_mask = np.logical_and(CQ3.VQD1>=p0[0]-dV_qd1, CQ3.VQD1<=p0[0]+dV_qd1)
V3_mask = np.logical_and(CQ3.VQD3>=p0[1]-dV_qd3, CQ3.VQD3<=p0[1]+dV_qd3)
CQ3_zoomin = deepcopy(CQ3.where(V1_mask).dropna(dim='VQD1').where(V3_mask).dropna(dim='VQD3'))

# Evaluate shifts
V1_shift = CQ1_zoomin.VQD1.isel(VQD1=CQ1_zoomin.sum(dim="VQD3").argmax(dim="VQD1")).values.item()
V3_shift = CQ3_zoomin.VQD3.isel(VQD3=CQ3_zoomin.sum(dim="VQD1").argmax(dim="VQD3")).values.item()

# Convert axes to detunings
CQ1_zoomin = CQ1_zoomin.assign_coords(
    VQD1 = convert_mV_to_detuning_ueV(CQ1_zoomin.VQD1.data, effective_lever_arms['qd_1'], V1_shift),
    VQD3 = convert_mV_to_detuning_ueV(CQ1_zoomin.VQD3.data, effective_lever_arms['qd_3'], V3_shift),
)
CQ3_zoomin = CQ3_zoomin.assign_coords(
    VQD1 = convert_mV_to_detuning_ueV(CQ3_zoomin.VQD1.data, effective_lever_arms['qd_1'], V1_shift),
    VQD3 = convert_mV_to_detuning_ueV(CQ3_zoomin.VQD3.data, effective_lever_arms['qd_3'], V3_shift),
)
```

## Simulation data

```python
from paths import PROCESSED_DATA_FOLDER
sim_qd1 = xr.load_dataarray(PROCESSED_DATA_FOLDER / "qdmzm/simulation_data_QD1.h5", engine='h5netcdf')
sim_qd3 = xr.load_dataarray(PROCESSED_DATA_FOLDER / "qdmzm/simulation_data_QD3.h5", engine='h5netcdf')
```

# Plotting

```python
fig, ax = plt.subplots(1,2, figsize=(COLUMNWIDTH, 1.8),layout="constrained")
ax_qd1=ax[0]
ax_qd3=ax[1]
vm=350
img1 = CQ1.plot(ax=ax_qd1, x="VQD1", y="VQD3", vmin=-vm,vmax=vm, cmap="RdBu_r", add_colorbar=False,  rasterized=True)
img3 = CQ3.plot(ax=ax_qd3, x="VQD1", y="VQD3", vmin=-vm,vmax=vm, cmap="RdBu_r", add_colorbar=False, rasterized=True)

for ax in [ax_qd1, ax_qd3]:
    ax.set_title("")
    ax.set_xlabel(r"$\Delta V_{\rm QD1, DG1}$ [mV]")

    # Add rectangles identifying zoom-ins
    rect = patches.Rectangle(
        (p0[0]-dV_qd1, p0[1]-dV_qd3), 2*dV_qd1, 2*dV_qd3,
        linewidth=1, edgecolor='green', facecolor='none', alpha=0.8
    )
    ax.add_patch(rect)

# Y-axis labels
ax_qd3.set_ylabel("")
ax_qd3.set_yticklabels([])
ax_qd1.set_ylabel(r"$\Delta V_{\rm QD3, DG3}$ [mV]")

# Titles
ax_qd1.set_title(r"(a) $C_{\mathrm{Q}}({\rm QD1})$", fontsize=fs, loc="left")
ax_qd3.set_title(r"(b) $C_{\mathrm{Q}}({\rm QD3})$", fontsize=fs, loc="left")

# Colorbar
plt.colorbar(img1, ax=[ax_qd1, ax_qd3], label=r"$C_{\mathrm{Q}}$ [aF]")

plt.savefig(FIGURE_OUTPUT_FOLDER / "qdmzm_exp.pdf", bbox_inches="tight", pad_inches=0.01, transparent=True)
```

```python
fig, ax = plt.subplots(2,2, figsize=(COLUMNWIDTH, 2.5), sharex=True, sharey=True, layout="constrained")

img1 = CQ1_zoomin.plot(x="VQD1", y="VQD3", vmin=-vm,vmax=vm, cmap="RdBu_r", add_colorbar=False, ax=ax[0,0], rasterized=True)
img3 = CQ3_zoomin.plot(x="VQD1", y="VQD3", vmin=-vm,vmax=vm, cmap="RdBu_r", add_colorbar=False, ax=ax[0,1], rasterized=True)

x1 = ax[0,0].get_xlim()
x3 = ax[0,0].get_ylim()

sim_qd3.plot(x="detuning_1",ax=ax[1,1], vmin=-vm,vmax=vm, cmap="RdBu_r", add_colorbar=False, rasterized=True)
sim_qd1.plot(x="detuning_1",ax=ax[1,0], vmin=-vm,vmax=vm, cmap="RdBu_r", add_colorbar=False, rasterized=True)

ax[0,0].set_xlim(*x1)
ax[0,0].set_ylim(*x3)

# Let's erase the defaults from xarray
for a in ax.reshape(-1):
    a.set_title("")
    a.set_xlabel("")
    a.set_ylabel("")

# Let's add titles and labels:
ax[0,0].set_title(r"(c) $C_{\mathrm{Q}}({\rm QD1})$", fontsize=fs, loc="left")
ax[0,1].set_title(r"(d) $C_{\mathrm{Q}}({\rm QD3})$", fontsize=fs, loc="left")
ax[1,0].set_title(r"(e) $C_{\mathrm{Q}}({\rm QD1})$ (sim.)", fontsize=fs, loc="left")
ax[1,1].set_title(r"(f) $C_{\mathrm{Q}}({\rm QD3})$ (sim.)", fontsize=fs, loc="left")
ax[0,0].set_ylabel(r"$\Delta_{\rm QD3}$ [$\mu$eV]")
ax[1,0].set_ylabel(r"$\Delta_{\rm QD3}$ [$\mu$eV]")
ax[1,0].set_xlabel(r"$\Delta_{\rm QD1}$ [$\mu$eV]")
ax[1,1].set_xlabel(r"$\Delta_{\rm QD1}$ [$\mu$eV]")

cbar = plt.colorbar(img1, ax=ax, label=r"$C_{\mathrm{Q}}$ [aF]")
cbar.set_ticks([-200,0,200])
plt.savefig(FIGURE_OUTPUT_FOLDER / "qdmzm_fit.pdf", bbox_inches="tight", pad_inches=0.01, transparent=True)
```

```python

```
