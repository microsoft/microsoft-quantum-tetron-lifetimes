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

from paths import RAW_DATA_FOLDER, FIGURE_OUTPUT_FOLDER

from analysis_code.plotting_helpers import (
    COLUMNWIDTH,
    savefig,
    configure_matplotlib_params,
)
```

```python
run_id = "depletion_voltage_width"
```

```python
configure_matplotlib_params(font_size=6)
```

```python
da = xr.open_dataarray(RAW_DATA_FOLDER / run_id / "depletion_voltage_width.h5", engine="h5netcdf")

fig, ax = plt.subplots(figsize=(COLUMNWIDTH, 0.6*COLUMNWIDTH), dpi=300)
da.plot(ax=ax, lw=3)
ax.axvline(40, c='tab:red', ls='--', label='Backbone width')
ax.axvline(60, c='tab:green', ls='--', label='Wire width')

ax.legend()

ax.set_xlim(38, 65)
ax.set_xlabel("Superconductor width [nm]")
ax.set_yticks(np.arange(-1.3, -0.89, 0.1))

ax.set_ylim(None, -0.9)
ax.set_ylabel("Simulated depletion voltage [V]")
ax.grid()

plt.tight_layout()
savefig(fig, FIGURE_OUTPUT_FOLDER / f"{run_id}.pdf")
plt.show()
```
