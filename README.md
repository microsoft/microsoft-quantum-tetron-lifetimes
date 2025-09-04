# Data and analysis scripts for paper "Distinct Lifetimes for X and Z Loop Measurements in a Majorana Tetron Device"

This repository contains the code used to generate the analysis in our [pre-print arxiv:2507.08795](https://arxiv.org/abs/2507.08795).
The source data is available for download at [10.5281/zenodo.16987492](https://zenodo.org/records/16987492).

### TLDR;

Run the following in a Linux shell:
```sh
curl -LsSf https://astral.sh/uv/install.sh | sh # see Requirements for more details
uv sync
source .venv/bin/activate

python prepare_data.py # Choose option 1 when asked, see Data for more details
jupytext --execute $( ls notebooks/*.md | grep -v spectrum.md ) # see Running the analysis for more details
```

---

**Table of Contents**

- [Requirements](#requirements)
- [Data](#data)
    - [Downloading the Data](#downloading-the-data)
    - [Data Paths](#data-paths)
- [Running the analysis](#running-the-analysis)
    - [Converting notebook formats](#converting-notebook-formats)
    - [Running all notebooks](#running-all-notebooks)
    - [Map between Figure Number and Notebook](#map-between-figure-number-and-notebook)
- [Data Processing](#data-processing)
    - [CQ Conversion](#cq-conversion)
    - [Conductance Corrections](#conductance-corrections)
- [Microsoft Open Source Code of Conduct](#microsoft-open-source-code-of-conduct)
- [Trademarks](#trademarks)
---

## Requirements

The data analysis and figure generating scripts in this repository were run using Python 3.10.9 and the environment specified in [`uv.lock`](uv.lock)
(we also provide a [`requirements.txt`](requirements.txt) for advanced users who do not wish to use `uv`).

The quickest way to reproduce the environment used for data analysis and figure generation is to use the
tool [`uv`](https://docs.astral.sh/uv/), which can be installed by following [these instructions](https://docs.astral.sh/uv/getting-started/installation/).

Once `uv` is installed, running the following commands will install the necessary software into an isolated
environment, and activate that environment for future use:
```sh
uv sync # download Python 3.10.9, and all dependencies, creating a virtual environment with everything installed
source .venv/bin/activate  # activate the virtual environment
```

## Data

Measurement datasets are available on [Zenodo](https://zenodo.org/records/16987492). You can download it manually and point to the correct folders in [paths.py](paths.py).

### Downloading the Data

For convinience, you can optionally use the `prepare_data.py` script that will do the data download and the correct layout of the folder
structure outlined in [Data Paths](#data-paths). You can change the target directory this prep is done in by editing `DATA_FOLDER` in `paths.py`.

The 4 options the script allows for are:
- (0) Skip download: If you donwloaded the datasets already and placed them directly in `DATA_FOLDER`, this will unzip all the files and place them in the correct directory.
- (1) Download Minimal Dataset: Download only the datasets required to reproduce the plots with preprocessing (e.g. CQ Converted RF data, kurtosis computed on all WPWP maps to avoid the ~80Gb download, etc.).
- (2) Download All Datasets: Download all the raw, converted and processed datasets - this should allow you to regenerate processed datasets (such as CQ conversion and computing kurtosis on all WPWP data). 
- (3) Download Custom Set: Download a set of files of your choosing, to be specified in the `custom_datasets` variable in `prepare_data.py` (e.g. if you want to replicate just the CQ conversion of the data in figure 4, download `xmpr.zip`).

If you face issues getting the directory structure to map correctly, you can reference [Data Paths](#data-paths) to setup the structure manually. If you face issues with the
download, you can use your browser to manually download it from [Zenodo](https://zenodo.org/records/16987492), place the zips in `DATA_FOLDER` (`./data` by default) and then
run `prepare_data.py` with option (0).

### Data Paths
As an interface for all our code, all raw and generated data are organized into the following folder variables:

- `DATA_FOLDER`: Variable for a folder containing four subdirectories outlined below
- `RAW_DATA_FOLDER`: Variable for a folder containing all measured data to be used for analysis and plotting.
- `CONVERTED_DATA_FOLDER`: Variable for a target folder to contain all CQ Converted datasets generated from running `cq_conversion/workflow.py` on the data in `RAW_DATA_FOLDER`
- `PROCESSED_DATA_FOLDER`: Variable for a target folder that contains processed data, including, cached computations of functions on large datasets, simulation data and transport data.
- `CQ_CONV_INTERMEDIATE_FIGURE_FOLDER`: Variable for a target folder to store intermediate figures for debugging the output of various stages of CQ Conversion
- `FIGURE_OUTPUT_FOLDER`: Output location for generated figures

Different figures require different files, if you are interested in reproducing one figure, you can reference the source code for which files to download from zenodo.
The minimal set of files to download to reproduce *all figures* is: `converted_data`, `processed_data` and `depletion_voltage_width` - totalling around 5Gbs.

# Running the analysis

Each figure has a corresponding notebook in the `notebooks/` folder. The analysis code lives in
`analysis_code` and is imported across the notebooks. For RF data, preprocessing is done using the
`cq_conversion` module, see [CQ Conversion](#cq-conversion) for more information. Using the uv
environment outlined in [Requirements](#requirements), you can run the notebooks in jupyter as a
notebook (see how to convert notebooks below) or the jupytext cli tool to regenerate the figures.

The only exception to this is `spectrum.md` which requires Julia.

### Converting between notebook formats

To convert from the `.md` format to a `.ipynb` and run in an interactive Jupyter Notebook client,
you can use `jupytext` as follows: `jupytext notebooks/*.md --to ipynb`. This will create a Jupyter
notebook for every markdown notebook in the folder.

### Running all notebooks

To run all notebooks in python, load up the correct environment in your shell
(`source .venv/bin/activate`) then run:
`jupytext --execute $( ls notebooks/*.md | grep -v spectrum.md )`

This will regenerate all figures except for those in `spectrum.md`. You can launch these separately
and run them.

### Map between Figure Number and Notebook

To reproduce a specific figure, reference this map from figure number to notebook. Any figures
not listed here (Fig. 1, A5, A6, A9) are schematics.

**Main Figures**
- Figure 2: `xmpr.md`
- Figure 3: `zmpr.md`
- Figure 4: `error_metrics.md`

**Appendix Figures**
- A1 & A2: `spectrum.md`
- A3: `long_time_record.md`
- A4: `timescales.md`
- A7: `junctions.md`
- A8: `depletion_voltage_width.md`
- A10: `tgp_phase_diagram.md`
- A11: `qdmzm.md`
- A12: `transport_x_loop.md`
- A13: `average_qd_qd.md`
- A14: `tuning_drive_sweep.md`
- A15: `transport_z_loop.md`

# Data Processing

## CQ Conversion

This module contains all the datasets that need to be CQ converted in `datasets.py` along with information about the resonator and background. CQ conversion implements the same method
as in [Nature 638, 651–655 (2025)](https://doi.org/10.1038/s41586-024-08445-2) to convert the RF measurement in units of voltage to units of capacitance shift at the device. This is run
on all our RF measurements, except for the Wireplunger maps (~80Gb), to reduce unnecessary computation time - although, in principle, you could run this with the same conversion parameters
as the other QD1 runs (e.g. `xmpr` run). Otherwise, all analyses of RF data in our analysis notebooks expect inputs to be CQ converted.

To run CQ conversion on a specific datasets (for example the `xmpr` and `zmpr` datasets), run `python cq_conversion/workflow.py xmpr zmpr` in your terminal. To run CQ conversion on all datasets, use `python cq_conversion/workflow.py --all`. This takes longer for datasets with timetraces - regenerating all could easily take 40-60 minutes on slower machines due to the size of the timetrace datasets.

This module is similar to the one in our previous paper's repo `microsoft/azure-quantum-parity-readout`(https://github.com/microsoft/azure-quantum-parity-readout), with a few updates, namely:
- Diversity combining information from measured idler data to improve the SNR. (`diversity_combining.py`)
- TWPA On CQ Conversion has been updated to use a rescaled TWPA Off calib for less sensitivity to the TWPA background. (`twpa_effective_calib.py`)
- Changes on loading of datasets and workflow run to accomodate the updates above. (`cq_conversion.py` and `workflow.py`)
- Minor refactoring.

## Conductance Corrections

- Transport and TGP datasets have had their conductance values corrected following the approach described in
[10.1103/PhysRevB.107.245423](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.107.245423). First, the currents and
voltages at the sample terminals are evaluated from those measured and applied at the measurement instrument level via a
transfer function that has been determined from pre-calibrated resistive and capacitive network of the measurement lines
of the setup. Second, they are corrected for voltage divider effects by accounting for the finite line resistance of the
drain connection to the superconducting wire.

## Microsoft Open Source Code of Conduct

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).

Resources:

- [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/)
- [Microsoft Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/)
- Contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with questions or concerns
- Employees can reach out at [aka.ms/opensource/moderation-support](https://aka.ms/opensource/moderation-support)

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.