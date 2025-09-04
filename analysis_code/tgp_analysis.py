# +
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# -

import xarray as xr
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from types import SimpleNamespace
import tgp # package source lives in https://github.com/microsoft/azure-quantum-tgp
from matplotlib.colors import ListedColormap
from analysis_code.plotting_helpers import pcolormesh_kw

def analyze_two(
    ds_left: xr.Dataset,
    ds_right: xr.Dataset,
    min_cluster_size: int = 7,
    zbp_average_over_cutter: bool = True,
    zbp_probability_threshold: float = 0.6,
    gap_threshold_high: float = 70e-3,
    gap_threshold_factor: float = 0.05,
    cluster_gap_threshold = None,
    cluster_volume_threshold = None,
    cluster_percentage_boundary_threshold = 0.6,
):
    ds_left, ds_right = tgp.two.extract_gap(
        ds_left, ds_right, gap_threshold_factor=gap_threshold_factor
    )
    zbp_ds = tgp.two.zbp_dataset_derivative(
        ds_left,
        ds_right,
        average_over_cutter=zbp_average_over_cutter,
        zbp_probability_threshold=zbp_probability_threshold,
    )

    tgp.two.set_zbp_gap(zbp_ds, ds_left, ds_right)
    tgp.two.set_gap_threshold(zbp_ds, threshold_high=gap_threshold_high)

    zbp_ds = tgp.two.cluster_and_score(
        zbp_ds,
        min_cluster_size=min_cluster_size,
        cluster_gap_threshold=cluster_gap_threshold,
        cluster_volume_threshold=cluster_volume_threshold,
        cluster_percentage_boundary_threshold=cluster_percentage_boundary_threshold,
    )
    return SimpleNamespace(
        zbp_ds=zbp_ds,
        ds_left=ds_left,
        ds_right=ds_right,
    )

def plot_phase_diagram(
    zbp_ds,
    ax,
    cutter_value,
):
    pl_kw = {
        "add_colorbar": False,
        "shading": "nearest",
        "infer_intervals": False,
        "vmin": -0.5,
        "vmax": 1.5,
    }
    pl_kw.update(pcolormesh_kw)

    ds_sel = zbp_ds.sel(cutter_pair_index=cutter_value)
    gap_bool = 1.0 * ds_sel.gap_boolean.squeeze()
    cmap = mpl.colors.ListedColormap(["w", "tab:blue"])
    im1 = (
        gap_bool.squeeze()
        .transpose("V", "B")
        .plot
        .pcolormesh(ax=ax, cmap=cmap, zorder=1, **pl_kw)
    )

    zbp_bool = ds_sel.zbp.squeeze()
    cmap = mpl.colors.ListedColormap([np.array([255, 229, 82]) / 256, "tab:orange"])
    im2 = (
        gap_bool.where(zbp_bool, np.nan)
        .squeeze()
        .transpose("V", "B")
        .plot
        .pcolormesh(ax=ax, cmap=cmap, zorder=2, **pl_kw)
    )
    clusters = tgp.common.expand_clusters(
        ds_sel.gapped_zbp_cluster,
        dim="zbp_cluster_number",
    )

    reps = 20
    B = np.array(zbp_ds["B"])
    V = np.array(zbp_ds["V"])
    B1 = np.linspace(B.min(), B.max(), B.size * reps)
    V1 = np.linspace(V.min(), V.max(), V.size * reps)

    for cl in clusters.zbp_cluster_number.values:
        cluster_sel = clusters.sel(zbp_cluster_number=cl)
        (
            cluster_sel.astype(float)
            .interp(B=B1, V=V1, method="nearest")
            .plot
            .contour(
                x="B",
                y="V",
                ax=ax,
                levels=[0, 1],
                colors="k",
                linewidths=1.0,
                linestyles="-",
                add_colorbar=False,
                zorder=1000,
            )
        )
        ax.set_title("")

    return im1, im2

def plot_overlapping_diagram(zbp_ds, ax, add_colorbar=True):

    fig = ax.figure

    cutters = zbp_ds["cutter_pair_index"].values.astype(int)
    color_list = [f"C{i}" for i in cutters]
    cmap = ListedColormap(color_list)


    clusters = (zbp_ds.gapped_zbp_cluster > 0) * (zbp_ds.cutter_pair_index + 1)
    clusters_with_nans = clusters.where(clusters != 0, np.nan)
    for cutter in zbp_ds.cutter_pair_index.values:
        im = clusters_with_nans.sel(cutter_pair_index=cutter).T.plot(
            cmap=cmap,
            add_colorbar=False,
            ax=ax,
            alpha=0.5,
            vmax=max(cutters) + 1.5,
            vmin=0.5,
        )
    if add_colorbar:
        cax = fig.add_axes([0.995, 0.627, 0.015, 0.3])
        # cax = fig.add_axes([0.92, 0.704, 0.015, 0.2])
        cbar = fig.colorbar(
            im,
            cax=cax,
            # ax=ax,
            location="right",
            ticks=cutters + 1,
            label="Cutter pair index",
            aspect=14,
        )

    ax.set_title("")
