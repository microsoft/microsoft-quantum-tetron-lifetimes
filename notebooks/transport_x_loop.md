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
import sys
sys.path.append("..")
```

```python
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from analysis_code.plotting_helpers import (
    COLUMNWIDTH,
    pcolormesh_kw,
    add_subfig_label,
    savefig,
    configure_matplotlib_params,
    auto_label_subplots,
)

from paths import FIGURE_OUTPUT_FOLDER, PROCESSED_DATA_FOLDER
```

```python
run_id = "transport_x_loop"
```

```python
configure_matplotlib_params(font_size=6)
```

```python
# Data load
tgps = [
    xr.open_dataset(PROCESSED_DATA_FOLDER / run_id / filename, engine="h5netcdf")
    for filename in ["top_wire_ds_right.h5", "bottom_wire_ds_right.h5"]
]

transport_dss = [
    xr.open_dataset(PROCESSED_DATA_FOLDER / run_id / filename, engine="h5netcdf")
    for filename in ["xloop_junction2.h5", "xloop_junction4.h5"]
]
```

```python
# Plot configuration
tgp_plot_params = [
    {
        "dataset": tgps[0],
        "index_label": "WP1",
        "gate_label": "2",
        "yticks": np.arange(-1.311, -1.305+1e-4, 0.001),
    },
    {
        "dataset": tgps[1],
        "index_label": "WP2",
        "gate_label": "4",
        "yticks": np.arange(-1.292, -1.286+1e-4, 0.001),
    },
]

transport_plot_params = [
    {
        "dataset": transport_dss[0],
        "data_var": "g_22",
        "index_label": "WP1",
        "gate_label": "2",
        "line_y": -1.307714,
        "yticks": [-1.308, -1.307],
    },
    {
        "dataset": transport_dss[1],
        "data_var": "g_44",
        "index_label": "WP2",
        "gate_label": "4",
        "line_y": -1.288646,
        "yticks": [-1.289, -1.288],
    },
]

fig, axs = plt.subplots(
    2,
    2,
    figsize=(1.1*COLUMNWIDTH, 0.8*COLUMNWIDTH),
    sharex=False,
    dpi=300,
)

# Plot TGP data (left column)
for i, params in enumerate(tgp_plot_params):
    ax = axs[i, 0]
    l = params["gate_label"]
    params["dataset"]['g_rr'].sel(cutter_pair_index = 0, B = 2.3).plot.pcolormesh(
        ax=ax,
        vmin=0,
        vmax=1,
        cbar_kwargs = {
            "label": f"$dI_{l}/dV_{l}$ [$e^2/h$]",
            "ticks": np.arange(0, 1.01, 0.5),
        },
        **pcolormesh_kw,
    )

    ax.set_ylabel(rf"$V_\mathrm{{{params['index_label']}}}$ [V]")
    ax.set_xlabel(rf"$V_{params['gate_label']}$ [mV]")
    ax.set_title("")
    ax.set_xlim(-0.0575, 0.0575)
    ax.set_yticks(params["yticks"])

# Plot transport data (right column)
for i, params in enumerate(transport_plot_params):
    ax = axs[i, 1]
    l = params["gate_label"]
    params["dataset"][params["data_var"]].plot.pcolormesh(
        ax=ax,
        vmin=0,
        vmax=1,
        cbar_kwargs = {
            "label": f"$dI_{l}/dV_{l}$ [$e^2/h$]",
            "ticks": np.arange(0, 1.01, 0.5),
        },
        **pcolormesh_kw,
    )

    ax.axhline(params["line_y"], color='tab:red', linestyle='--', linewidth=1)

    ax.set_ylabel(rf"$V_\mathrm{{{params['index_label']}}}$ [V]")
    ax.set_xlabel(rf"$V_{params['gate_label']}$ [mV]")
    ax.set_xlim(-0.0575, 0.0575)
    ax.get_yaxis().get_major_formatter().set_useOffset(False)
    ax.set_yticks(params["yticks"])

# Add horizontal lines between plots
for i in range(2):
    ylim = axs[i][1].get_ylim()
    for y in ylim:
        axs[i][0].axhline(y, c="k", ls="--", lw=1)

fig.tight_layout()
auto_label_subplots(axs, width=0.17, height=0.15)
savefig(fig, FIGURE_OUTPUT_FOLDER / f"{run_id}.pdf")
plt.show()
```

```python

```
