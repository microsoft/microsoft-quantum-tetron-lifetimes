import numpy as np
import xarray as xr
from scipy import stats
from sklearn.mixture import GaussianMixture
import math

import os, tempfile
from pathlib import Path
from dask.distributed import Client, LocalCluster

from collections.abc import Callable, Iterable
from typing import Any, Literal

### Parallelization

class DaskClient:
    def __init__(self, threads_per_worker=1):
        self.threads_per_worker = threads_per_worker
        self._client = None

    def __getattr__(self, attr: str) -> Any:
        return getattr(self.client, attr)

    @property
    def client(self):
        if self._client is None:
            print("Starting a `dask.Client` with a `distributed.LocalCluster`.")
            cluster = LocalCluster(
                threads_per_worker=self.threads_per_worker, local_directory=TMP_DIR
            )
            self._client = Client(cluster)
        return self._client


dask_client = DaskClient()

def get_tmp_dir() -> Path:
    """Get temp directory"""
    return Path(tempfile.gettempdir())

TMP_DIR = get_tmp_dir()

def _optimal_chunks(da: xr.DataArray, client: DaskClient, ignore: Iterable[str] = ()):
    n, dim = max((n, d) for d, n in da.sizes.items() if str(d) not in ignore)
    n_cores = sum(client.ncores().values()) or 1
    chunk_size = max(math.ceil(n / n_cores), 1)
    return {dim: chunk_size}

def apply_func_along_dim(
    da: xr.DataArray,
    dim: str,
    func: Callable[..., Any],
    output_dtypes,
    output_sizes: dict[str, int],
    output_core_dims,
    func_kwargs: dict[str, Any] | None = None,
    parallel: bool | Literal["auto"] = "auto",
):
    """Apply function along one dimension using parallelization with optimal chunks

    Parameters
    ----------
    da : xr.DataArray
    dim : str
        dimension to apply function along
    func : Callable[..., Any]
        function to apply
    output_dtypes : _type_
        list of output dtypes, same size as number of ouputs
    output_sizes : Dict[str, int]
       mapping from dimension names to sizes for outputs
    output_core_dims : _type_
        List of the same length as the number of output arguments from func
    func_kwargs : Optional[Dict[str, Any]], optional
        key word arguments to go to func, by default None
    parallel : Union[bool, Literal['auto']], optional
        whether to run parallel computations,
        if 'auto' - does it if os name other than 'nt', by default "auto"

    Returns
    -------
    List of datasets
    """
    if parallel == "auto":
        parallel = os.name != "nt"

    if parallel:
        da = da.chunk(_optimal_chunks(da, dask_client, ignore=[dim]))

    x = xr.apply_ufunc(
        func,
        da[dim],
        da,
        kwargs=func_kwargs or {},
        input_core_dims=[[dim], [dim]],
        output_core_dims=output_core_dims,
        vectorize=True,
        dask="parallelized" if parallel else "allowed",
        output_dtypes=output_dtypes,
        dask_gufunc_kwargs={"output_sizes": output_sizes},
        keep_attrs=True,
    )
    if not parallel:
        return x
    return dask_client.compute(x, traverse=False, optimize_graph=False, sync=True)

### Error metrics

def err_a(m):
    """
    err_a(m)

    Empirical assignment error for a 2×2 stochastic matrix `m`, where
    `m[i,j]` represents the probability `Pr(i|j)` where `i` and `j`
    represent measurement outcomes. Takes max of all rows and columns of (m - id)

    `err_a([1 0; 0 1])` is zero.
    """
    return np.max(np.abs(np.stack(m) - np.eye(2)), axis=(1, 2))


def err_a_from_offdiagonals(m):
    # only takes off diagonals, required for output of beta function as the for quantiles!=0.5 diagonal and off diagonal outputs don't sum to one
    return np.max([m[0, 1], m[1, 0]])


def get_error_statistics_from_counts(counts, conf=0.95, alpha=0.5):
    """
    Calculates err_a and conditional probabilities P(i|j) and their upper and lower confidence interval from count 2x2 matrix `counts`.
    The latter counts instances of (0,0),(0,1),(1,0),(1,1) in the data. Uses the beta distribution to extract the confidence intervals of
    a binomial process and extracts err_a from the max of P(0|1), P(1|0).
    """

    ests = {}
    total_counts = counts.sum(axis=0, keepdims=True)
    probabilities = (counts + alpha) / (total_counts + 2 * alpha)
    lower_bounds = stats.beta.ppf(
        (1 - conf) / 2, counts + alpha, total_counts - counts + alpha
    )
    upper_bounds = stats.beta.ppf(
        1 - (1 - conf) / 2, counts + alpha, total_counts - counts + alpha
    )
    for i in range(2):
        for j in range(2):
            ests[f"P({i}|{j})"] = {
                "lower": lower_bounds[i, j],
                "median": probabilities[i, j],
                "upper": upper_bounds[i, j],
            }
    ests["err_a"] = {
        "lower": err_a_from_offdiagonals(lower_bounds),
        "median": err_a_from_offdiagonals(probabilities),
        "upper": err_a_from_offdiagonals(upper_bounds),
    }
    return ests


def nan_int(num):
    # returs an integer out of floats close to 0 and 1 and random choice for nan values
    if num == -1:
        return np.random.choice([0, 1])
    try:
        out = int(num)
    except ValueError:
        out = np.random.choice([0, 1])
    return out


def get_count_matrix(signal):
    # input of binary 0,1 signal
    # calculates count matrix used in `get_error_statistics_from_counts` from a boolean time trace
    counts = np.zeros((2, 2), dtype=int)
    for i in range(len(signal) - 1):
        counts[nan_int(signal[i]), nan_int(signal[i + 1])] += 1
    return counts


def classify_gmm(tr, return_gmm=False):
    """
    classifies (in general complex) time trace `tr` into boolean outcomes

    output of gmm will have two components corresponding to real and imaginary part
    even if input is real
    """
    gmm = GaussianMixture(n_components=2, covariance_type="diag")
    reshaped = np.asarray([np.real(tr), np.imag(tr)]).T
    nan_mask = np.isnan(reshaped).any(axis=1)
    nonnan = reshaped[~nan_mask]
    labels = np.full(len(tr), np.nan)
    try:
        gmm.fit(nonnan)
    except ValueError:
        if return_gmm:
            gmm.means_ = np.full((2, 2), np.nan)
            gmm.weights_ = np.full(2, np.nan)
            gmm.covariances_ = np.full((2, 2), np.nan)
            return labels, gmm
        else:
            return labels

    if len(nonnan) > 0:
        labels_non_nan = gmm.predict(nonnan)
        labels[~nan_mask] = labels_non_nan

    if return_gmm:
        return labels, gmm
    else:
        return labels


def _error_stats_2arg(x, signal, **kwargs):
    # adding dummy argument to work with standard parallelization function
    return _error_stats(signal, **kwargs)


def tuples_to_complex(tuples):
    # converts a list of tuples to a list of complex numbers
    return tuples[:, 0] + 1j * tuples[:, 1]

def covs_to_sigmas(covs):
    return tuples_to_complex(np.sqrt(covs))

def _error_stats(signal, **kwargs):
    # same as `get_error_statistics_from_counts` but uses raw signal as input instead of count matrix
    labels, gmm = classify_gmm(signal, return_gmm=True)
    cm = get_count_matrix(labels)
    error_metrics = get_error_statistics_from_counts(cm, **kwargs)
    metrics = ["err_a", "P(0|0)", "P(0|1)", "P(1|0)", "P(1|1)"]
    return tuple(
        [np.array(list(error_metrics[m].values())) for m in metrics]
        + list(gmm.weights_)
        + list(tuples_to_complex(np.array(gmm.means_)))
        + list(covs_to_sigmas(gmm.covariances_))
        + [np.nan_to_num(labels, nan=-1).astype(np.int8)]
    )


def _calc_error_stats(func, da, parallel=False, **kwargs):
    """
    xarray wrapper uses which uses `func=_error_stats` or its 2arg variant to calculate error metrics
    """
    if parallel:
        (
            err_a,
            P00,
            P01,
            P10,
            P11,
            gmm_weight1,
            gmm_weight2,
            gmm_mean1,
            gmm_mean2,
            gmm_sigma1,
            gmm_sigma2,
            labels,
        ) = apply_func_along_dim(
            da,
            dim="time",
            func=func,
            output_core_dims=[["err_quantile"]] * 5 + [[]] * 6 + [["time"]],
            output_dtypes=[float] * 5 + [float] * 2 + [complex] * 4 + [np.int8],
            output_sizes={"err_quantile": 3, "time": da.sizes["time"]},
        )

    else:
        (
            err_a,
            P00,
            P01,
            P10,
            P11,
            gmm_weight1,
            gmm_weight2,
            gmm_mean1,
            gmm_mean2,
            gmm_sigma1,
            gmm_sigma2,
            labels,
        ) = xr.apply_ufunc(
            func,
            da,
            input_core_dims=[["time"]],
            output_core_dims=[["err_quantile"]] * 5 + [[]] * 6 + [["time"]],
            output_dtypes=[float] * 5 + [float] * 2 + [complex] * 4 + [np.int8],
            output_sizes={"err_quantile": 3, "time": da.sizes["time"]},
            vectorize=True,
            kwargs=kwargs,
        )
    error_stats = xr.Dataset(
        {
            "err_a": err_a,
            "P00": P00,
            "P01": P01,
            "P10": P10,
            "P11": P11,
            "gmm_weight1": gmm_weight1,
            "gmm_weight2": gmm_weight2,
            "gmm_mean1": gmm_mean1,
            "gmm_mean2": gmm_mean2,
            "gmm_sigma1": gmm_sigma1,
            "gmm_sigma2": gmm_sigma2,
            "labels": labels,
        }
    )
    error_stats = error_stats.assign_coords(err_quantile=["lower", "median", "upper"])
    return error_stats


def get_error_statistics(da, parallel=False, **kwargs):
    """
    Calculates error metrics from a data arrray of raw (real) time traces. Allows parallelization to deal with classification.
    """
    if parallel:
        return _calc_error_stats(_error_stats_2arg, da, parallel=True, **kwargs)
    else:
        return _calc_error_stats(_error_stats, da, parallel=False, **kwargs)
