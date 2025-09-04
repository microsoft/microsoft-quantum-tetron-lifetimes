# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# +
import numpy as np
from scipy import stats
import xarray as xr
import arviz as az
from pymc import Exponential, Model, Uniform, sample

def _detect_switches(trace, peak1, peak2, r=0):
    """
    Detect switches between two states in a trace based on thresholds derived from peak values.

    Parameters:
    trace (xr.DataArray): The input data trace where switches are to be detected.
    peak1 (float): The value of the first peak.
    peak2 (float): The value of the second peak.
    r (float): A parameter to adjust the threshold calculation. Default is 0.

    Returns:
    xr.DataArray: An array indicating the states detected with 1 representing above high threshold,
                  0 representing below low threshold, and NaN for in-between values.
    """
    peak_high = np.max([peak1, peak2])
    peak_low = np.min([peak1, peak2])

    t2 = peak_high * (1 - r) + peak_low * r
    t1 = peak_low * (1 - r) + peak_high * r

    out = xr.where(trace > t2, 1, np.nan)
    out = xr.where(np.isnan(out) & (trace < t1), 0, out)
    return out

def threshold_traces(ds: xr.Dataset, cq_name: str | None, mean1=None, mean2=None, r=0):
    if mean1 is None or mean2 is None:
        mean1 = ds[f"gmm.mean1.{cq_name}"]
        mean2 = ds[f"gmm.mean2.{cq_name}"]
    boolean_trace = xr.apply_ufunc(
        _detect_switches,
        ds[cq_name] if cq_name is not None else ds,
        mean1,
        mean2,
        input_core_dims=[["time"], [], []],
        output_core_dims=[["time"]],
        vectorize=True,
        kwargs={"r": r}
    )

    boolean_traces = boolean_trace.ffill(dim="time").bfill(dim="time")
    boolean_traces_Cq = xr.where(
        boolean_traces == 0, mean1, mean2
    )

    if cq_name is not None:
        ds[f"thresh.digital_bool.{cq_name}"] = boolean_traces
        ds[f"thresh.digital.{cq_name}"] = boolean_traces_Cq
        return ds

    return boolean_traces, boolean_traces_Cq


def _dwell_time(time, digital_bool):
    """
    Calculate dwell times in up and down states from a digitized trace.

    Parameters:
    time (np.ndarray): Array representing the time points.
    digital_bool (np.ndarray): Array representing the digitized states (binary).

    Returns:
    tuple: Two arrays containing the dwell times in the up state and down state respectively.
    """
    time = time[:-1]

    dts = np.diff(digital_bool)
    event_times = time[np.abs(dts) > 0]
    dwell_time = np.diff(event_times)
    if digital_bool[0] == 0.0:
        return dwell_time[::2], dwell_time[1::2]  # up times, down times
    else:
        return dwell_time[1::2], dwell_time[0::2]


def _dwells_fit_model(dwells, sample_rate, trace_len, plot_debug=True):
    """
    Fit a model to dwell times using Bayesian Inference.

    Parameters:
    dwells (list or np.ndarray): Array of dwell times to fit the model to.
    sample_rate (float): The sampling rate in seconds per sample. Default is 4.5e-6.
    trace_len (float): The total length of the trace in seconds. Default is 65e-3.
    plot_debug (bool): Whether to display a progress bar during sampling. Default is True.

    Returns:
    pd.DataFrame: A summary dataframe containing the Bayesian fit results.
    """
    # Calculate the longest and shortest feasible dwell time
    shortest_interval = sample_rate
    longest_interval = trace_len

    # Fit the resulting dataset using Bayesian Inference. This allows us to extract a model error
    rate_model = Model()
    with rate_model:
        # Define prior for scale parameter
        scale = Uniform("tau", lower=shortest_interval, upper=longest_interval)

        # Define likelihood
        _ = Exponential("dwells", scale=scale, observed=dwells)

        # Sample the distribution
        idata = sample(10000, progressbar=plot_debug)
    # Calculate stats with a 1 sigma interval, down to microsecond precision
    summary = az.summary(idata, round_to=6, hdi_prob=0.6827)
    return summary


def prepare_plot_dwells_bayesian(dwells, sample_rate, trace_len, model_fit=None):
    """
    Prepare Bayesian analysis results for plotting dwell times.

    Parameters:
    dwells (list or np.ndarray): Array of dwell times.
    model_fit (pd.DataFrame, optional): Precomputed Bayesian fit results. If None, the model
                                         will be fit to the provided dwell times.

    Returns:
    dict: A dictionary containing the mean and highest density interval (HDI) bounds.
    """
    if all(np.isnan(dwells)):
        return dict(
            mean=np.nan,
            hdi_minus=np.nan,
            hdi_plus=np.nan,
        )
    if model_fit is None:
        model_fit = _dwells_fit_model(dwells, sample_rate, trace_len, plot_debug=False)

    model_fit_res = dict(
        mean=model_fit["mean"].values[0],
        hdi_minus=model_fit["hdi_15.865%"].values[0],
        hdi_plus=model_fit["hdi_84.135%"].values[0],
    )
    return model_fit_res

def plot_dwells_bayesian(
    ax, dwells, model_fit, bin_scale=15, color="k", s=3.0, label="\\tau"
):

    n_dwells = len(dwells)
    counts, bins_e = np.histogram(
        dwells,
        bins=int(n_dwells / bin_scale),
        density=False,
    )
    bins = (bins_e[:-1] + bins_e[1:]) / 2

    mean = model_fit["mean"]

    w_plt = np.where(counts > 0.1)
    pdf_sample_points = stats.expon(scale=mean).pdf(bins[w_plt])
    pdf_scale = 1 / (np.sum(pdf_sample_points) / n_dwells)

    ax.scatter(bins * 1e3, counts, label="", color=color, s=s)

    sig = (model_fit["hdi_plus"] - model_fit["hdi_minus"]) * 1e3 / 2

    ax.plot(
        bins[w_plt] * 1e3,
        stats.expon(scale=mean).pdf(bins[w_plt]) * pdf_scale,
        color=color,
        label=rf"${label} = {mean * 1e3:1.1f}\pm{sig:1.1f}\,$ms",
    )

    ax.set_xlabel("Dwell time [ms]")
    ax.set_ylabel("Count")
    ax.set_yscale("log")

    return mean
