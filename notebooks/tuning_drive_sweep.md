---
jupyter:
  jupytext:
    formats: ipynb,md
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
from itertools import product

from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch

from scipy.optimize import curve_fit

import pandas as pd
```

```python
from analysis_code.plotting_helpers import font_size as fs, COLUMNWIDTH
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
```

# Load data

```python
from paths import CONVERTED_DATA_FOLDER, FIGURE_OUTPUT_FOLDER

def load_power_sweep_CQ_data():
    drive_voltages_mV = [2,4,6,8,10]
    dsets = []
    for Vd in drive_voltages_mV:
        ds = xr.load_dataset(CONVERTED_DATA_FOLDER / f"drive_sweep/drive_sweep_{Vd}mV_Cq.h5", engine='h5netcdf')
        ds = ds.expand_dims(Vd=[Vd])
        dsets.append(ds)
    CQs = xr.concat(dsets, dim="Vd")

    return CQs

cq_converted_data = load_power_sweep_CQ_data()
```

# SNR analysis

```python
# Fit functions
def gauss(x,mu,sigma,A):
    return A*np.exp(-(x-mu)**2/2/sigma**2)

def bimodal(x,mu1,mu2,sigma,A, w):
    return gauss(x,mu1,sigma,A*w)+gauss(x,mu2,sigma,A*(1-w))
```

```python
def _fit_distributions(traces, plot=True):
    """
    For a set of time traces where all parameters are fixed except the drive amplitude,
    fit a Gaussian model and a bimodal Gaussian model to extract signal and noise parameters.

    Fit parameters are saved to a dataframe.
    """
    Vds = traces.Vd.data
    NVd = len(Vds)
    outcomes=[]
    if plot:
        fig, ax = plt.subplots(1,NVd, sharex=False, sharey=True, figsize=(10,3))

    for jVd in range(NVd):
        trace = traces.isel(Vd=jVd)
        hist_kwargs = dict(bins=51)
        if plot:
            y, x, _ = trace.plot.hist(**hist_kwargs, histtype='step', ax=ax[jVd])
        else:
            y, x = np.histogram(trace, **hist_kwargs)

        # Convert bin edges to bin centers
        x=(x[1:]+x[:-1])/2

        # Single Gaussian fit:
        initial_parameters_gauss = (x.mean(), 750/Vds[jVd], 500)
        params_gauss, cov_gauss = curve_fit(gauss,x,y, initial_parameters_gauss)

        # Bimodal Gaussian fit:
        initial_parameters = [params_gauss[0]-50, params_gauss[0]+50, 750/Vds[jVd], 500, 0.5]
        bounds = ([x[0], x[0], 2, -np.inf, 0.4], [x[-1], x[-1], 1e4, np.inf, 0.6])
        params_bimodal, cov_bimodal = curve_fit(bimodal, x, y, initial_parameters, bounds=bounds)
        delta_cq= np.abs(params_bimodal[1] - params_bimodal[0])

        fit_results = dict(
            Vd=Vds[jVd],
            sigma_bimodal=params_bimodal[2],
            delta_cq=delta_cq,
            snr=0.5*delta_cq/params_bimodal[2],
            A=params_bimodal[3],
            w=params_bimodal[4],
            sigma_gauss=params_gauss[1],
            A_gauss=params_gauss[2],
        )

        if plot:
            ax[jVd].plot(x, gauss(x,*params_gauss),color='k', linestyle="--",lw=1,label='model', zorder=0)
            ax[jVd].plot(x, bimodal(x,*params_bimodal),color='tab:red',lw=1,label='model', zorder=-1)
            params_g1 = (params_bimodal[0], params_bimodal[2], params_bimodal[3]*params_bimodal[4])
            params_g2 = (params_bimodal[1], params_bimodal[2], params_bimodal[3]*(1-params_bimodal[4]))
            ax[jVd].plot(x, gauss(x, *params_g1),color='tab:green',lw=1,label='model', zorder=-2)
            ax[jVd].plot(x, gauss(x, *params_g2),color='tab:green',lw=1,label='model', zorder=-2)
            ax[jVd].set_title(
                f"$\\Delta C_{{\\mathrm{{Q}}}} = {delta_cq:.0f}$ aF\n$\\sigma_{{C_{{\\mathrm{{Q}}}}}}={params_bimodal[2]:.0f}$ aF",
                fontsize=10
            )
            plt.legend(["Data", "Gaussian Fit", "Bimodal Fit", "Individual Gaussians"])

        outcomes.append(fit_results)
    return outcomes

def fit_distributions(ds, iV1s, iV3s, iBs, **kwargs):
    outcomes = []
    for (iV1, iV3, iB) in product(iV1s, iV3s, iBs):
        trace = ds.Cq.isel(VQD3=iV3, VQD1=iV1, Bperp=iB)
        out = _fit_distributions(trace, **kwargs)
        df = pd.DataFrame(out)
        df['iV1'] = iV1
        df['iV3'] = iV3
        df['iB'] = iB
        outcomes.append(df)
    return pd.concat(outcomes,  ignore_index=True)
```

```python
# Example fits:
fit_distributions(cq_converted_data, iV1s=[2], iV3s=[7], iBs=[97], plot=True);
```

```python
# Fit multiple QD3 detunings and flux points to collect statistics
fit_results = fit_distributions(cq_converted_data, iV1s=[2], iV3s=range(8), iBs=[96,97,98], plot=False)
```

```python
def sigmaR_fit_function(x, readout_noise):
    return 2*x/readout_noise

def fit_readout_noise(fit_results, Vd_mV_to_uV, error_bars_Vd_mV_to_uV):
    data = 1/(fit_results.sigma_bimodal.to_numpy()*1e-3)
    Vd_mV = fit_results.Vd.to_numpy()
    params_sR, cov_sR = curve_fit(sigmaR_fit_function, Vd_mV*Vd_mV_to_uV, data, [4])
    params_sR_min, cov_sR = curve_fit(sigmaR_fit_function, Vd_mV*(Vd_mV_to_uV - error_bars_Vd_mV_to_uV), data, [4])
    params_sR_max, cov_sR = curve_fit(sigmaR_fit_function, Vd_mV*(Vd_mV_to_uV + error_bars_Vd_mV_to_uV), data, [4])
    error_bar  =np.mean([params_sR_max[0] - params_sR[0], params_sR[0] - params_sR_min[0]])
    print(f"sigma_R fit : {params_sR[0]} +/- {error_bar}")

    return dict(sigma_R = params_sR[0], sigma_R_min = params_sR_min[0], sigma_R_max = params_sR_max[0], sigmaR_err=error_bar)
```

```python
def _collect_snr_statistics(df: pd.DataFrame, Vd_mV_to_uV: float) -> pd. DataFrame:
    """
    Based on the fit results, collect statistics on signal and noise as a function of drive amplitude.

    Inputs:
    * df: Pandas dataframe obtained using the `fit_distributions` function
    * Vd_mV_to_uV:  Conversion factor accounting for the attenuation of the fridge lines.
                    Converts the drive amplitude in mV at the output of the room tempeature
                    signal generator to the drive amplitude in uV at the device.
    """
    avgd_data = []
    for (Vd_V,), dfA in df.groupby(["Vd"]):
        Vd = Vd_mV_to_uV*Vd_V
        Svec = Vd*dfA.delta_cq.to_numpy()/1e3
        sig_vec = dfA.sigma_bimodal.to_numpy()
        row = dict(
            Vd = Vd,
            Smax=Svec.max(),
            Smin=Svec.min(),
            S = np.mean(Svec),
            S_err = np.std(Svec),
            sig  = np.mean(sig_vec),
            sig_sig = np.std(sig_vec),
        )
        avgd_data.append(row)

    return pd.DataFrame(avgd_data)

def collect_snr_statistics(df: pd.DataFrame, readout_noise, Vd_mV_to_uV: float, error_bars_Vd_mV_to_uV: float) -> pd. DataFrame:
    snr_stats = _collect_snr_statistics(df, Vd_mV_to_uV)
    snr_stats_min_drive = _collect_snr_statistics(df, Vd_mV_to_uV - error_bars_Vd_mV_to_uV)
    snr_stats_max_drive = _collect_snr_statistics(df, Vd_mV_to_uV + error_bars_Vd_mV_to_uV)

    labels = dict(S="Maximum of mean", Smax="Maximum of max")
    for k in ['S', 'Smax']:
        i_max_mean_signal = np.argmax(snr_stats[k])
        S_max_mean_signal = snr_stats[k][i_max_mean_signal]
        lower_bound = snr_stats_min_drive[k][i_max_mean_signal]
        upper_bound = snr_stats_max_drive[k][i_max_mean_signal]
        error_bars = np.mean([upper_bound - S_max_mean_signal, S_max_mean_signal - lower_bound])
        print(f"{labels[k]} signal : {S_max_mean_signal} +/- {error_bars} fF.uV at Vd = {snr_stats.Vd[i_max_mean_signal]} uV")
        print(f"{labels[k]} SNR : {S_max_mean_signal/readout_noise['sigma_R']}")

    return snr_stats
```

```python
Vd_mV_to_uV = 2.5475
error_bars_Vd_mV_to_uV = 0.1875
readout_noise = fit_readout_noise(fit_results, Vd_mV_to_uV, error_bars_Vd_mV_to_uV)
snr_statistics = collect_snr_statistics(fit_results, readout_noise, Vd_mV_to_uV, error_bars_Vd_mV_to_uV)
```

## Plotting

```python
def _add_subfig_labels(ax):
    ax[0,0].text(0.01, 0.98, "$\\bf(a)$", verticalalignment='top', horizontalalignment='left', transform=ax[0,0].transAxes)
    ax[1,0].text(0.01, 0.98, "$\\bf(b)$", verticalalignment='top', horizontalalignment='left', transform=ax[1,0].transAxes)
    ax[0,1].text(0.01, 0.98, "$\\bf(c)$", verticalalignment='top', horizontalalignment='left', transform=ax[0,1].transAxes)
    ax[1,1].text(0.01, 0.98, "$\\bf(d)$", verticalalignment='top', horizontalalignment='left', transform=ax[1,1].transAxes)

def _plot_panel_a(cqs, fit_results, iVQD3, iVQD1, ax):
    Bperp_mT = 1e3*cqs.Bperp.values
    NVd = len(cqs.Vd)
    for j in range(NVd):
        Vd = cqs.Vd.isel(Vd=j).values.item()
        y = cqs.Cq.isel(Vd=j).std(dim="time").isel(VQD3=iVQD3).isel(VQD1=iVQD1).squeeze(drop=True).values
        ax.plot(Bperp_mT, y, label=f"{np.round(Vd*Vd_mV_to_uV)}", color=colors[j])

    legend = ax.legend(
        fontsize=fs-2,
        title="$V_d$ [$\\mathrm{\\mu V}$]",
        frameon=False,
        ncol=3,
        bbox_to_anchor=(0.5,1.0),
        loc="lower center",
        columnspacing=0.5
    )
    legend.get_title().set_fontsize(fs)

    iBs= np.sort(np.unique(fit_results.iB.to_numpy()))
    ax.axvspan(Bperp_mT[iBs[0]], Bperp_mT[iBs[-1]], alpha=0.3, color="gray")
    ax.set_xlim(Bperp_mT[0], Bperp_mT[-1])
    ax.set_ylabel(r"std[$C_{\mathrm{Q}}$] [aF]")
    ax.set_xlabel(r"$B_\perp$ [mT]")
    ax.set_ylim(0,375)

def _plot_panel_b(cqs, iVQD1, iVQD3, iB, iVd, ax):
    trace = cqs.Cq.isel(VQD3=iVQD3, VQD1=iVQD1, Bperp=iB, Vd=iVd)
    y, x, _ = trace.plot.hist(bins=41, histtype='step', ax=ax)
    x=(x[1:]+x[:-1])/2
    Vd = cqs.Vd[iVd].item()

    # Bimodal Gaussian fit
    expected=[trace.mean()-50, trace.mean()+50, 750/Vd, 500, 0.5]
    fit_params,cov=curve_fit(bimodal,x,y,expected, bounds=([x[0],x[0],2,-np.inf,0.4], [x[-1],x[-1], 1e4,np.inf,0.6]))

    ax.plot(x,bimodal(x,*fit_params),color='tab:red',lw=1,label='model', zorder=-1)
    ax.plot(x,gauss(x,fit_params[0], fit_params[2], fit_params[3]*fit_params[4]),color='tab:green',lw=1,label='model', zorder=-2)
    ax.plot(x,gauss(x,fit_params[1], fit_params[2], fit_params[3]*(1-fit_params[4])),color='tab:green',lw=1,label='model', zorder=-2)
    ax.set_title('')
    ax.set_ylabel("Counts")
    ax.set_xlabel(r"$C_{\mathrm{Q}}$ [aF]")
    ax.set_ylim(0,460)

    # Adding annotations to distribution example
    # The arrows don't quite match those coordinates on the plot scale.
    # Adding +/- 25 to match to actual center of the Gaussians
    arrows_style = dict(arrowstyle='<|-|>', mutation_scale=6, linewidth=0.5, color='k', zorder=10)
    text_style = dict(fontsize=fs-2, verticalalignment="baseline", horizontalalignment="center")
    ymax = max(bimodal(x,*fit_params))
    y0= ymax*np.exp(-0.5)
    arrow = FancyArrowPatch((fit_params[0]-25, ymax), (fit_params[1]+25, ymax),  **arrows_style)
    ax.add_patch(arrow)
    ax.text((fit_params[0]+fit_params[1])/2,ymax+15, r"$\Delta C_{\mathrm{Q}}$", **text_style)

    text_style['verticalalignment'] = "top"
    for i in range(2):
        arrow = FancyArrowPatch(
            (fit_params[i]-fit_params[2]-25, y0), (fit_params[i]+fit_params[2]+25, y0), **arrows_style
        )
        ax.add_patch(arrow)
        ax.text(fit_params[i]+10,y0-15, r"$2\sigma_{C_{\mathrm{Q}}}$", **text_style)

def _plot_panel_c(snr_statistics, readout_noise, ax):
    sigmaR = readout_noise['sigma_R']
    ax.plot(snr_statistics.Vd, snr_statistics.S,".", c=colors[0])
    ax.errorbar(snr_statistics.Vd, snr_statistics.S,yerr=snr_statistics.S_err, linestyle="none", color=colors[0])
    ax.set_ylabel(r"$S\ $ [$\mu{\rm V} \cdot $fF]")
    ax.set_xlabel(r"$V_d$ [$\mu$V]")
    ax.set_yticks([0,1.5,3,4.5])

    # We add a right y axis with the same range as the signal, but in SNR units.
    a2 = ax.twinx()
    ax.set_ylim(0,1.4*sigmaR)
    a2.set_ylim(ax.get_ylim()/(sigmaR))
    a2.set_yticks([0,0.3,0.6,0.9,1.2])
    a2.set_ylabel("SNR")

def _plot_panel_d(snr_statistics, readout_noise, ax):
    # 1e-3 convert aF to fF
    y0 = 1/(1e-3*snr_statistics.sig)
    yminus = y0 - 1/(1e-3*(snr_statistics.sig+snr_statistics.sig_sig))
    yplus = 1/(1e-3*(snr_statistics.sig-snr_statistics.sig_sig)) - y0
    yerr = np.stack([yminus, yplus])

    ax.plot(snr_statistics.Vd, y0,".", c=colors[0])
    ax.errorbar(snr_statistics.Vd, y0,yerr=yerr, linestyle="none", color=colors[0])
    ax.set_ylabel(r"$1/\sigma_{C_{\mathrm{Q}}}$ [fF$^{-1}$]")
    ax.set_xlabel(r"$V_d$ [$\mu$V]")

    xs = np.linspace(5.0, 26, 101)
    sigmaR = readout_noise["sigma_R"]
    ax.plot(xs, sigmaR_fit_function(xs, sigmaR), c="k", ls="--", label=rf"$\sigma_R={ sigmaR:.2f}$")

def generate_plot(cqs, fit_results, snr_statistics, readout_noise, Vd_mV_to_uV):
    figsize = (COLUMNWIDTH, 3.2)
    fig, ax = plt.subplots(2, 2, figsize=figsize, sharex=False)
    _add_subfig_labels(ax)
    _plot_panel_a(cqs, fit_results, iVQD1=2, iVQD3=7, ax=ax[0,0])
    _plot_panel_b(cqs, iVQD1=2, iVQD3=7, iB=97, iVd=2, ax=ax[1,0])
    _plot_panel_c(snr_statistics, readout_noise, ax[0,1])
    _plot_panel_d(snr_statistics, readout_noise, ax[1,1])
    ax[0,1].set_xlim(ax[1,1].get_xlim())

    plt.subplots_adjust(hspace=0.5, wspace=0.475, bottom=0.125, left=0.13, right=0.89, top=0.85)
    plt.savefig(FIGURE_OUTPUT_FOLDER / "tuning_drive_sweep.pdf", bbox_inches="tight", pad_inches=0.01, transparent=True)

generate_plot(cq_converted_data, fit_results, snr_statistics, readout_noise, Vd_mV_to_uV)
```

```python

```
