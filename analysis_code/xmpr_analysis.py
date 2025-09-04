from itertools import permutations
from math import prod
import numpy as np
import sympy
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
from tabulate import tabulate
from sklearn.utils import resample
from copy import deepcopy
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def _crossdot(x, y, lx, l) -> np.ndarray:
    """
    Helper function for crosscov
    """
    if l >= 0:
        return np.dot(x[: lx - l], y[l:lx])
    else:
        return np.dot(x[-l:lx], y[: lx + l])


def crosscov(x, y, lags, demean=True) -> np.ndarray:
    """
    Compute the crosscov in python
    """
    r = np.zeros(len(lags))
    lx = len(x)
    m = len(lags)

    y = np.conj(y)

    if (len(y) != lx) or (len(r) != m):
        raise ValueError("Dimension mismatch")

    def check_lags(lx, lags):
        if any(abs(l) >= lx for l in lags):
            raise ValueError("Lags out of bounds")

    check_lags(lx, lags)

    if demean:
        zx = x - np.mean(x)
        zy = y - np.mean(y)
    else:
        zx = x
        zy = y

    for k in range(m):
        r[k] = _crossdot(zx, zy, lx, lags[k]) / lx

    return r

def make_crosscov_fit_function(use_log=True):

    switching_rate = sympy.symbols("\\lambda", positive=True, real=True) # switching_rate is 1/tau
    signal = sympy.symbols("Smag", positive=True, real=True) # signal is related to DeltaCQ as DeltaCQ**2/4
    lag = sympy.Symbol("t_{\\mathrm{lag}}")

    fit_expr = signal * sympy.exp(-2 * switching_rate * lag)

    if use_log:
        fit_expr = sympy.log(fit_expr, evaluate=False)

    l_fit_expr = sympy.lambdify((lag, signal, switching_rate), fit_expr, "numpy")

    def fit_expr_eval(t, S, *p):
        return l_fit_expr(np.abs(np.asarray(t)), np.abs(S), *[np.abs(1 / τi) for τi in p])

    return fit_expr, fit_expr_eval

def bootstrap_gaussians(data, n_components, n_bootstrap):

    # Arrays to store bootstrap estimates
    mean_diffs = np.zeros(n_bootstrap)
    std_devs = np.zeros(n_bootstrap)
    weight_devs = np.zeros(n_bootstrap)

    # Bootstrap resampling
    for idx in range(n_bootstrap):
        sample = resample(data)
        gmm = GaussianMixture(n_components=n_components, covariance_type='tied', random_state=0)
        gmm.fit(sample.reshape(-1, 1))

        means = np.sort(gmm.means_.flatten())
        mean_diff = means[1] - means[0]
        std_dev = np.sqrt(gmm.covariances_[0])
        weight_dev = min(gmm.weights_)

        mean_diffs[idx] = mean_diff
        std_devs[idx] = std_dev
        weight_devs[idx] = weight_dev

    # Compute standard deviations as error bars
    mean_diff_error = np.std(mean_diffs)
    std_dev_error = np.std(std_devs)
    weight_dev_error = np.std(weight_devs)

    return mean_diff_error, std_dev_error, weight_dev_error


def fit_and_sum_gaussians(data, x_axis, label, n_components=2, n_bootstrap=1000):
    """
    Fits a Gaussian Mixture Model (GMM) to the data and returns the summed fitted Gaussian values.

    Parameters:
    - data: numpy.ndarray, the input data to fit the GMM.
    - x_axis: numpy.ndarray, the x-axis values to sample the fitted Gaussians.
    - n_components: int, the number of Gaussian components to fit.

    Returns:
    - combined_pdf: numpy.ndarray, the y-values of the summed fitted Gaussians.
    """

    # Fit Gaussian Mixture Model
    gmm = GaussianMixture(n_components=n_components, covariance_type='tied')
    gmm.fit(data.reshape(-1, 1))

    if n_bootstrap > 0:
        mean_diff_error, std_dev_error, weight_dev_error = bootstrap_gaussians(data, n_components=n_components, n_bootstrap=n_bootstrap)

    deltaCq = abs(gmm.means_[1] - gmm.means_[0])[0]
    std = np.sqrt(gmm.covariances_[0])[0]
    weight = min(gmm.weights_)

    # tabulate output deltaCq, std and weights
    table = tabulate(
        [
            ["Value", f"{deltaCq:.0f}", f"{std:.0f}", f"{weight:.2f}"],
            ["Error", f"{mean_diff_error:.0f}", f"{std_dev_error:.0f}", f"{weight_dev_error:.2f}"] if n_bootstrap>0 else ["Error", "-", "-", "-"]
        ],
        headers=["", "deltaCq", "$\\sigma_{CQ}$", "weight"],
        tablefmt="grid"
    )
    print(f"{label} Fitted Bimodal Gaussian Parameters:\n", table)

    # Compute the combined PDF
    combined_pdf = np.zeros_like(x_axis)
    indiv_pdf = []
    for mean, weight in zip(gmm.means_, gmm.weights_):
        indiv_pdf.append(weight * norm.pdf(x_axis, mean, std))
        combined_pdf += indiv_pdf[-1]

    return combined_pdf, indiv_pdf

def extract_cross_lags_above_noise(
    time,
    signal,
    timescale_guesses,
    fit_expr,
    *,
    do_fit=True,
    plot=False,
    ax=None,
    minimum_lag=2,
    max_lag=100,
    max_lag_fit=7,
    ax_hist=None,
    color='C0',
    fit_many=False # if noisy data, then you can run this when unsure about lags
):
    time = deepcopy(1e6 * time.to_numpy())

    if plot and ax is None:
        plt.figure()
        ax = plt.gca()

    signal = signal - np.mean(signal)
    time -= time[0]

    dtm = np.diff(time)
    assert np.all(np.isclose(dtm, dtm[0]))
    dtm = dtm[0]

    lags = np.arange(-max_lag, max_lag + 1, 1)
    R_sig = crosscov(signal, signal, lags)

    if plot:
        ax.plot(dtm * lags, R_sig, ".-", color=color, markersize=2)
        ax.set_yscale("log")

        ax.set_ylim(0.1, 50)
        ax.set_xlim(-dtm*max_lag / 2, +dtm*max_lag / 2)

    if not do_fit:
        return []

    Smag_guess = np.mean(crosscov(signal, signal, [-4, 4]))
    p0 = [Smag_guess, *timescale_guesses]

    def fit_expr_(t_lag, S, *p):
        return fit_expr(dtm * t_lag, S, *p)

    taus = []
    amps = []
    fit_errs = []

    for lag_window_0 in [0] if not fit_many else np.arange(minimum_lag, max_lag_fit):
        for lag_window_1 in [max_lag_fit] if not fit_many else np.arange(lag_window_0+1, max_lag_fit+1):

            selector = np.bitwise_and(
                np.abs(lags) >= lag_window_0,
                np.abs(lags) <= lag_window_1,
            )

            fit, fit_cov = curve_fit(
                fit_expr_, lags[selector], np.log(np.abs(R_sig[selector])), p0
            )

            # convert fit_cov to error bars
            fit_err = np.sqrt(np.diag(fit_cov))

            fit_y = np.exp(fit_expr_(lags, *fit))
            fit_yerr_p = np.exp(fit_expr_(lags, *(fit+fit_err)))
            fit_yerr_m = np.exp(fit_expr_(lags, *(fit-fit_err)))
            kerning = ""
            label = f"${kerning} \\tau_X{kerning}{kerning}={kerning}{kerning}{fit[1]:.1f}{kerning} \\pm{kerning} {fit_err[1]:.1} \\mathrm{{\\mu s}}$"

            if plot:
                if fit_many:
                    ax.plot(dtm * lags, fit_y, label=label, color="black", alpha=0.01)
                else:
                    lags_to_plot = abs(lags) <= max_lag_fit+1
                    ax.plot(dtm * lags[lags_to_plot], fit_y[lags_to_plot],
                        label=label, color="red", linestyle='solid', alpha=1, lw=1)
                    ax.fill_between(
                        dtm * lags[lags_to_plot],
                        fit_yerr_p[lags_to_plot],
                        fit_yerr_m[lags_to_plot],
                        color="red",
                        alpha=0.3,
                    )
                    ax.legend(fontsize=7, handlelength=1)
            amps += [fit[0]]
            taus += [fit[1]]
            fit_errs += [fit_err]

    if plot:
        ax.axvspan(-dtm*max_lag_fit, -dtm*minimum_lag, alpha=0.1, color="blue")
        ax.axvspan(dtm*minimum_lag, dtm*max_lag_fit, alpha=0.1, color="blue")

        if ax_hist is not None:
            ax_hist.hist(taus, bins=np.arange(0, 20, 0.25))

    return amps, taus, fit_errs
