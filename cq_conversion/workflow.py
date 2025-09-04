# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from twpa_effective_calib import get_effective_calib
from diversity_combining import diversity_combine_idler
import xarray as xr
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt

from cq_plot import plot_cq_interpolators
from cq_conversion import process_CqProjection, CQConversionInput

from timeit import default_timer as timer

from datasets import datasets, RAW_DATA_FOLDER, CONVERTED_DATA_FOLDER, CQ_CONV_INTERMEDIATE_FIGURE_FOLDER


def cq_convert_ds(cq_conv_input: CQConversionInput, target_dir: Path=CONVERTED_DATA_FOLDER):
    print(">"*20, f"\nBegin CQ Converting {cq_conv_input.name}")
    tn = timer()

    raw_ds = xr.load_dataset(RAW_DATA_FOLDER / cq_conv_input.data_path, engine="h5netcdf") # can open_datasets with chunks='auto' if too big to fit in memory w/ drawback of speed

    ed = cq_conv_input.elec_delay
    f0 = cq_conv_input.resonance_frequency
    fd = raw_ds.drive_frequency

    calib_off_ds: xr.Dataset = xr.load_dataset(RAW_DATA_FOLDER / cq_conv_input.calib_path_twpa_off, engine="h5netcdf", chunks='auto').squeeze()
    calib_off_ds = calib_off_ds * np.exp(-1j*2*np.pi*ed*calib_off_ds.frequency.to_numpy())

    if cq_conv_input.calib_path_twpa_on is not None:
        calib_on_ds:  xr.Dataset = xr.load_dataset(RAW_DATA_FOLDER / cq_conv_input.calib_path_twpa_on, engine="h5netcdf", chunks='auto').squeeze()
        calib_on_ds = calib_on_ds * np.exp(-1j*2*np.pi*ed*calib_on_ds.frequency.to_numpy())

        calib_ds = get_effective_calib(
            data=raw_ds,
            calib_off=calib_off_ds,
            calib_on=calib_on_ds,
            sig_var=cq_conv_input.data_rf_parameter
        )
    else:
        calib_ds = calib_off_ds
        calib_on_ds = calib_off_ds

    tp, tn = tn, timer()
    print(f"Loaded Dataset {cq_conv_input.name}, time:", tn-tp)
    print(raw_ds.data_vars)

    fs = calib_ds.frequency.to_numpy()
    calib = calib_ds[cq_conv_input.data_rf_parameter].to_numpy()

    tp, tn = tn, timer()
    print("Processed Calibration Curve, time:", tn-tp)

    if cq_conv_input.isel is not None:
        corrected_data = raw_ds.isel(**cq_conv_input.isel)
    else:
        corrected_data = raw_ds

    if cq_conv_input.coarsen_time is not None and cq_conv_input.coarsen_time > 1:
        corrected_data = corrected_data.coarsen(time=cq_conv_input.coarsen_time, boundary='trim').mean()

        tp, tn = tn, timer()
        print("Coarsened dataset, time:", tn-tp)

    # correct electrical delay
    corrected_data *= np.exp(-1j*2*np.pi*ed*fd)

    tp, tn = tn, timer()
    print("Corrected data electrical delay", tn-tp)

    print(f"> {cq_conv_input.name} ready for CQ Conversion")

    if cq_conv_input.div_comb:
        corrected_data = diversity_combine_idler(
            ds=corrected_data,
            idler_var=cq_conv_input.idler_rf_parameter,
            signal_var=cq_conv_input.data_rf_parameter
        )

    ## Plot CQ interpolators and save

    plot_cq_interpolators(
        fs=fs,
        calib=calib,
        data=corrected_data,
        analysis_input=cq_conv_input,
        calib_drive_power=calib_on_ds.drive_power,
    )

    tp, tn = tn, timer()

    plt.savefig(CQ_CONV_INTERMEDIATE_FIGURE_FOLDER / f"{cq_conv_input.name}_2_cq_interpolators.png")
    print("Plot Data and Calib Interpolators in IQ space, time:", tn-tp)
    print(f"Saved Data and Calib Interpolators Figure {cq_conv_input.name}_2_cq_interpolators.png")

    cq_conv_ds: xr.Dataset = process_CqProjection(
        fs=fs,
        calib=calib,
        data=corrected_data,
        analysis_input=cq_conv_input,
        amp_ratio=(corrected_data.drive_power/calib_on_ds.drive_power)
    )

    target_path = target_dir / cq_conv_input.target_name
    target_path.parent.mkdir(exist_ok=True)

    # in case a dataset already exists at that location and has a lock, delete and re-write instead of
    # amending to not lose the conversion.
    if target_path.exists():
        target_path.unlink()

    cq_conv_ds.to_netcdf(target_path, engine="h5netcdf")
    print(f"Saved Final Dataset at {target_path}")


def main():

    from dask.distributed import Client

    _ = Client(threads_per_worker=32, n_workers=1)

    parser = argparse.ArgumentParser(description='Run workflow on specified datasets.')
    parser.add_argument('keys', nargs='*', help='Keys to run on')
    parser.add_argument('--all', action='store_true', help='Run on all datasets')

    args = parser.parse_args()

    if args.all:
        args.keys = datasets.keys()

    if len(args.keys)==0:
        raise ValueError(f"""

Enter the name of the dataset you want to CQ-convert or `--all` to convert all.
Available datasets:
{list(datasets.keys())}

""")

    for key in args.keys:
        if isinstance(datasets[key], CQConversionInput):
            cq_convert_ds(datasets[key])
        elif isinstance(datasets[key], list):
            for _input in datasets[key]:
                cq_convert_ds(_input)
        else:
            raise ValueError(f"datasets entry for {key} is of the wrong format")

if __name__ == '__main__':
    main()
