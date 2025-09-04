import matplotlib as mpl
import matplotlib.pyplot as plt
from string import ascii_lowercase

COLUMNWIDTH = 3.375 # in
TEXTWIDTH = 7 # in = 17.8 cm

font_size = 7
plt.rcParams.update({
    "font.size": font_size,
    "axes.labelsize": font_size,
    "axes.titlesize": font_size,
    "xtick.labelsize": font_size,
    "ytick.labelsize": font_size,
    "legend.fontsize": font_size,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
})

pcolormesh_kw = {
    "linewidth": 0,
    "rasterized": True,
}

def add_subfig_label(
    ax: mpl.axes.Axes,
    label: str,
    width: float = 0.1,
    height: float = 0.1,
    description: str | None = None,
) -> None:
    """Add a label to a subfigure."""
    x0, x1 = ax.get_xlim()
    w = width * (x1 - x0)
    y0, y1 = ax.get_ylim()
    h = height * (y1 - y0)
    ax.add_patch(mpl.patches.Rectangle((x0, y1), w, -h, color="k", zorder=1000))
    ax.text(
        x0 + w / 2,
        y1 - h / 2,
        label,
        c="w",
        ha="center",
        va="center",
        # size=8,
        zorder=1001,
    )
    if not description is None:
        ax.set_title(description)

def add_subfig_labelxy(
    ax: mpl.axes.Axes,
    label: str,
    x: float = -0.1,
    y: float = 1.1,
    description: str = "",
    fontsize=8,
) -> None:
    """Add a label to a subfigure at a given position."""
    ax.text(
        x,
        y,
        f"({label})",
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        fontsize=fontsize,
    )
    ax.set_title(description)

def configure_matplotlib_params(font_size=7):
    """Configure matplotlib parameters with given font size."""
    plt.rcParams.update({
        "font.size": font_size,
        "axes.labelsize": font_size,
        "axes.titlesize": font_size,
        "xtick.labelsize": font_size-0.5 if font_size > 0.5 else font_size,
        "ytick.labelsize": font_size-0.5 if font_size > 0.5 else font_size,
        "legend.fontsize": font_size,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
    })

def create_figure_with_subplots(nrows, ncols, figsize=None, dpi=300, **subplot_kw):
    """Create a figure with subplots using common parameters."""
    if figsize is None:
        figsize = (COLUMNWIDTH, COLUMNWIDTH * nrows / ncols)
    return plt.subplots(nrows, ncols, figsize=figsize, dpi=dpi, **subplot_kw)

def setup_pcolormesh_plot(ax, data, plot_params):
    """Setup a pcolormesh plot with common parameters."""
    plot_kwargs = {
        "vmin": plot_params.get("vmin", 0),
        "vmax": plot_params.get("vmax", 1),
        "cmap": plot_params.get("cmap", "viridis"),
        **pcolormesh_kw,
    }

    if "cbar_kwargs" in plot_params:
        plot_kwargs["cbar_kwargs"] = plot_params["cbar_kwargs"]

    pcm = data.plot.pcolormesh(ax=ax, **plot_kwargs)

    # Set labels
    if "xlabel" in plot_params:
        ax.set_xlabel(plot_params["xlabel"])
    if "ylabel" in plot_params:
        ax.set_ylabel(plot_params["ylabel"])
    if "title" in plot_params:
        ax.set_title(plot_params["title"])
    else:
        ax.set_title("")

    # Set axis limits and ticks
    if "xlim" in plot_params:
        ax.set_xlim(*plot_params["xlim"])
    if "ylim" in plot_params:
        ax.set_ylim(*plot_params["ylim"])
    if "xticks" in plot_params:
        ax.set_xticks(plot_params["xticks"])
    if "yticks" in plot_params:
        ax.set_yticks(plot_params["yticks"])

    # Add horizontal/vertical lines
    if "hlines" in plot_params:
        for line in plot_params["hlines"]:
            ax.axhline(**line)
    if "vlines" in plot_params:
        for line in plot_params["vlines"]:
            ax.axvline(**line)

    return pcm

def plot_with_params(ax, data, plot_type="pcolormesh", **params):
    """Generic plotting function that can handle different plot types."""
    if plot_type == "pcolormesh":
        return setup_pcolormesh_plot(ax, data, params)
    elif plot_type == "plot":
        # Handle line plots
        plot_kwargs = {k: v for k, v in params.items()
                      if k in ["color", "linewidth", "linestyle", "label", "alpha", "marker", "markersize"]}
        line = ax.plot(data, **plot_kwargs)

        # Set labels and formatting
        if "xlabel" in params:
            ax.set_xlabel(params["xlabel"])
        if "ylabel" in params:
            ax.set_ylabel(params["ylabel"])
        if "title" in params:
            ax.set_title(params["title"])
        else:
            ax.set_title("")

        return line
    else:
        raise ValueError(f"Unsupported plot type: {plot_type}")

def create_subplot_grid(rows, cols, figsize=None, **kwargs):
    """Create a standardized subplot grid."""
    if figsize is None:
        aspect_ratio = rows / cols if cols > 0 else 1
        figsize = (COLUMNWIDTH * (2 if cols > 1 else 1), COLUMNWIDTH * aspect_ratio)

    return plt.subplots(rows, cols, figsize=figsize, dpi=300, **kwargs)

def auto_label_subplots(axs, start_label="a", **label_kwargs):
    """Automatically add labels to all subplots in order."""
    if not hasattr(axs, '__iter__'):
        axs = [axs]
    else:
        axs = axs.flatten() if hasattr(axs, 'flatten') else list(axs)

    fig_label = start_label
    for ax in axs:
        add_subfig_label(ax, fig_label, **label_kwargs)
        fig_label = chr(ord(fig_label) + 1)

def savefig(fig, filename):
    plt.savefig(
        filename,
        dpi=fig.dpi,
        bbox_inches="tight",
        pad_inches=0.01,
        transparent=True,
    )

def apply_sublabels(axs, x=-50, y=0, size='medium', weight='bold', ha='right', va='top'):

    for n, ax in enumerate(axs):

        if isinstance(x, list):
            _x = x[n]
        else:
            _x = x

        ax.annotate(f"({ascii_lowercase[n]})",
                    xy=(0, 1),
                    xytext=(_x, y[n] if isinstance(y, list) else y),
                    xycoords='axes fraction',
                    textcoords='offset points',
                    size=size,
                    color='k',
                    weight=weight,
                    horizontalalignment=ha,
                    verticalalignment=va)
