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

from analysis_code.tgp_analysis import (
    analyze_two,
    plot_phase_diagram,
    plot_overlapping_diagram
)
from analysis_code.plotting_helpers import (
    TEXTWIDTH,
    add_subfig_label,
    savefig,
    configure_matplotlib_params,
    auto_label_subplots,
)

from paths import PROCESSED_DATA_FOLDER, FIGURE_OUTPUT_FOLDER
```

```python
configure_matplotlib_params(font_size=7)
```

```python
run_id = "tgp_phase_diagram"

tgps = {
    f"{side}_{wire}": xr.open_dataset(PROCESSED_DATA_FOLDER / run_id /  f"{wire}_wire_ds_{side}.h5", engine="h5netcdf")
    for wire in ["top", "bottom"]
    for side in ["left", "right"]
}

zbps = [analyze_two(tgps[f"left_{wire}"], tgps[f"right_{wire}"]).zbp_ds for wire in ["top", "bottom"]]
```

```python
# Plot configuration
plot_config = {
    "figure": {
        "figsize": (1*TEXTWIDTH, 0.5*TEXTWIDTH),
        "dpi": 300,
        "gridspec_kw": {'width_ratios': [1, 1, 1, 1]},
    },
    "segments": [
        {
            "zbp_ds": zbps[0],
            "range": (-1.30875, -1.30675),
            "index_label": "WP1",
        },
        {
            "zbp_ds": zbps[1],
            "range": (-1.28975, -1.28775),
            "index_label": "WP2",
        },
    ],
    "cutter_titles": ["Cutter pair index 1", "Cutter pair index 2", "Cutter pair index 3"],
    "cutter_colors": ["C0", "C1", "C2"],
}

fig, axs = plt.subplots(
    2, 4,
    sharex=True,
    sharey="row",
    **plot_config["figure"]
)

for i, segment_params in enumerate(plot_config["segments"]):
    # Plot overlapping diagram (rightmost column)
    ax = axs[i, -1]
    plot_overlapping_diagram(segment_params["zbp_ds"], ax, add_colorbar=i==0)
    ax.vlines(2.3, *segment_params["range"], color='tab:red')

    # Plot phase diagrams for each cutter (first 3 columns)
    for cutter in range(3):
        ax = axs[i, cutter]
        im1, im2 = plot_phase_diagram(segment_params["zbp_ds"], ax=ax, cutter_value=cutter)
        ax.set_xticks(np.arange(2, 3.51, 0.5))

# Set labels and titles
for i, segment_params in enumerate(plot_config["segments"]):
    for j in range(4):
        ax = axs[i, j]
        if i == 0:
            if j < 3:
                ax.set_title(plot_config["cutter_titles"][j], color=plot_config["cutter_colors"][j])
            ax.set_xlabel("")
        else:
            ax.set_title("")
            ax.set_xlabel(r"$B$ [T]")

        if j == 0:
            ax.set_ylabel(rf"$V_\mathrm{{{segment_params['index_label']}}}$ [V]")
        else:
            ax.set_ylabel("")

# Add colorbars
cax = fig.add_axes([-0.05, 0.127, 0.015, 0.376])
cbar = fig.colorbar(im1, cax=cax, orientation="vertical", ticks=[0, 1])
cbar.ax.set_yticklabels(
    [r"Gapless", r"Gapped"],
    rotation=90,
    rotation_mode="anchor",
    va="baseline",
    ha="center",
)
cbar.ax.tick_params(pad=9)

cax = fig.add_axes([-0.05, 0.548, 0.015, 0.376])
cbar = fig.colorbar(im2, cax=cax, orientation="vertical", ticks=[0, 1])
cbar.ax.set_yticklabels(
    [r"Gapless$\,$&$\,$ZBP", r"Gapped$\,$&$\,$ZBP"],
    rotation=90,
    rotation_mode="anchor",
    va="baseline",
    ha="center",
)
cbar.ax.tick_params(pad=9)

plt.tight_layout()
auto_label_subplots(axs, width=0.12, height=0.12)
savefig(fig, FIGURE_OUTPUT_FOLDER / f"{run_id}.pdf")
plt.show()
```
```
