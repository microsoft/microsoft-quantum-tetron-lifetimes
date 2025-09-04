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
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from analysis_code.plotting_helpers import (
    COLUMNWIDTH,
    pcolormesh_kw,
    add_subfig_label,
    add_subfig_labelxy,
    savefig,
    configure_matplotlib_params,
    auto_label_subplots,
)
```

```python
configure_matplotlib_params(font_size=6)
```

```python
from pathlib import Path
from paths import FIGURE_OUTPUT_FOLDER, PROCESSED_DATA_FOLDER
run_id = "junctions"
```

```python
# Plot configuration
plot_params = [
    {
        "cutter_index": 1,
        "plunger_index": 12,
        "plunger_display_index": 1,
        "filename": "g_11_VWP1_VWC1.h5",
    },
    {
        "cutter_index": 2,
        "plunger_index": 12,
        "plunger_display_index": 1,
        "filename": "g_22_VWP1_VWC2.h5",
    },
    {
        "cutter_index": 3,
        "plunger_index": 34,
        "plunger_display_index": 2,
        "filename": "g_33_VWP2_VWC3.h5",
    },
    {
        "cutter_index": 4,
        "plunger_index": 34,
        "plunger_display_index": 2,
        "filename": "g_44_VWP2_VWC4.h5",
    },
]

fig, axs = plt.subplots(2, 2, figsize=(COLUMNWIDTH, COLUMNWIDTH*6.0/8.0), dpi=300)

for params, ax in zip(plot_params, axs.flatten()):
    g = xr.open_dataarray(PROCESSED_DATA_FOLDER / run_id / params["filename"], engine="h5netcdf")

    pl = g.plot.pcolormesh(
        ax=ax,
        vmin=0,
        vmax=6,
        cmap="viridis",
        **pcolormesh_kw,
    )
    ax.set_ylabel(r"$V_\mathrm{WP" + f"{params['plunger_display_index']}" + "}$ [V]")
    ax.set_xlabel(r"$V_\mathrm{WC" + f"{params['cutter_index']}" + "}$ [V]")
    pl.colorbar.set_label(f"$dI_{params['cutter_index']}/dV_{params['cutter_index']}$ [$e^2/h$]")
    ax.set_xticks(np.arange(-2, 2.1, 1))
    ax.set_title("")

auto_label_subplots(axs, width=0.17, height=0.16)

plt.tight_layout()
savefig(fig, FIGURE_OUTPUT_FOLDER / f"{run_id}.pdf")
plt.show()
```
