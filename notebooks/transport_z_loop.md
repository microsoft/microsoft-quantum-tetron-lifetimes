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
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

from analysis_code.plotting_helpers import (
    COLUMNWIDTH,
    pcolormesh_kw,
    add_subfig_label,
    savefig,
    configure_matplotlib_params,
    setup_pcolormesh_plot,
    auto_label_subplots,
)

from paths import FIGURE_OUTPUT_FOLDER, PROCESSED_DATA_FOLDER
```

```python
configure_matplotlib_params(font_size=6)
```

```python
run_id = "transport_z_loop"

transport_dss = [
    xr.open_dataset(PROCESSED_DATA_FOLDER / run_id / filename, engine="h5netcdf")
    for filename in ["zloop_junction2.h5", "zloop_junction4.h5"]
]
```

```python
plot_params = [
    {
        "data_var": "g_22",
        "xlabel": r"$V_2$ [mV]",
        "ylabel": r"$V_\mathrm{WP1}$ [V]",
        "cbar_kwargs": {"label": r"$dI_{2}/dV_{2}$ [$e^2/h$]"},
        "vmax": 1,
        "hlines": [{"y": -1.309, "color": "tab:red", "linestyle": "--", "linewidth": 1}],
        "yticks": np.arange(-1.310, -1.308 + 1e-4, 0.001),
        "ds": transport_dss[0],
    },
    {
        "data_var": "g_44",
        "xlabel": r"$V_4$ [mV]",
        "ylabel": r"$V_\mathrm{WP2}$ [V]",
        "cbar_kwargs": {"label": r"$dI_{4}/dV_{4}$ [$e^2/h$]"},
        "vmax": 1.25,
        "hlines": [{"y": -1.29116, "color": "tab:red", "linestyle": "--", "linewidth": 1}],
        "yticks": np.arange(-1.291, -1.290 + 1e-4, 0.001),
        "ds": transport_dss[1],
    },
]

fig, axs = plt.subplots(
    1, 2,
    figsize=(1.1 * COLUMNWIDTH, 0.4 * COLUMNWIDTH),
    dpi=300,
)

for ax, params in zip(axs, plot_params):
    # Update colorbar ticks based on vmax
    if "cbar_kwargs" in params:
        params["cbar_kwargs"]["ticks"] = np.arange(0, params["vmax"] + 1e-3, 0.5)

    setup_pcolormesh_plot(ax, params["ds"][params["data_var"]], params)
    ax.get_yaxis().get_major_formatter().set_useOffset(False)

plt.tight_layout()
auto_label_subplots(axs)
savefig(fig, FIGURE_OUTPUT_FOLDER / f"{run_id}.pdf")
plt.show()
```
