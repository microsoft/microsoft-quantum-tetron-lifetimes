# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from glob import glob
from cq_conversion import CQConversionInput
import numpy as np

import sys
sys.path.insert(1, '.')
sys.path.insert(1, '..')

# these imports are an interface that will be used by other classes
from paths import RAW_DATA_FOLDER, CONVERTED_DATA_FOLDER, CQ_CONV_INTERMEDIATE_FIGURE_FOLDER # noqa: E402

# The inductance of the readout resonators
L_ind_QD1 = 173e-9
L_ind_QDL = 247e-9
L_ind_QD3 = 235e-9

datasets: dict[str, CQConversionInput | list[CQConversionInput]] = {
    'xmpr': CQConversionInput(
        name="xmpr",
        data_path="xmpr/xmpr_Vrf.h5",
        calib_path_twpa_off="xmpr/xmpr_calib_twpa_off.h5",
        calib_path_twpa_on="xmpr/xmpr_calib_twpa_on.h5",
        elec_delay=37.94e-9,
        L_ind=L_ind_QD1,
        gaussian_sigma=10,
        fdelta=2e6,

        resonance_frequency=808.15e6,
        div_comb=True
    ),

    'xmpr_zoomout_1': CQConversionInput(
        name="xmpr_zoomout_1",
        data_path="xmpr/xmpr_zoomout_1_Vrf.h5",
        calib_path_twpa_off="xmpr/average_qd_qd_calib_twpa_off.h5",
        calib_path_twpa_on="xmpr/average_qd_qd_calib_twpa_on.h5",
        elec_delay=37.94e-9,
        L_ind=L_ind_QD1,
        gaussian_sigma=10,
        fdelta=2e6,

        resonance_frequency=808.15e6,
        div_comb=True
    ),

    'xmpr_zoomout_2': CQConversionInput(
        name="xmpr_zoomout_2",
        data_path="xmpr/xmpr_zoomout_2_Vrf.h5",
        calib_path_twpa_off="xmpr/average_qd_qd_calib_twpa_off.h5",
        calib_path_twpa_on="xmpr/average_qd_qd_calib_twpa_on.h5",
        elec_delay=37.94e-9,
        L_ind=L_ind_QD1,
        gaussian_sigma=10,
        fdelta=2e6,

        resonance_frequency=808.15e6,
        div_comb=True
    ),

    'zmpr': CQConversionInput(
        name="zmpr",
        data_path="zmpr/zmpr_Vrf.h5",
        calib_path_twpa_off="zmpr/zmpr_calib_twpa_off.h5",
        calib_path_twpa_on="zmpr/zmpr_calib_twpa_on.h5",
        elec_delay=15.2e-9,
        L_ind=L_ind_QDL,
        gaussian_sigma=10,
        fdelta=20e6,

        resonance_frequency=722.328e6,
        div_comb=True,

        data_rf_parameter="VQDL_signal",
        idler_rf_parameter="VQDL_idler",
    ),

    'long_time_record': CQConversionInput(
        name="long_time_record",
        data_path="long_time_record/long_time_record_Vrf.h5",
        calib_path_twpa_off="long_time_record/long_time_record_calib_twpa_off.h5",
        calib_path_twpa_on="long_time_record/long_time_record_calib_twpa_on.h5",
        elec_delay=37.94e-9,
        L_ind=L_ind_QD1,
        gaussian_sigma=5,
        fdelta=3e6,

        resonance_frequency=808.15e6,
        div_comb=True,

        minimum_Cq_res = 2e-19
    ),

    "qdmzm_qd1": CQConversionInput(
        name="qdmzm_qd1",
        data_path="qdmzm/qdmzm_qd1_Vrf.h5",
        calib_path_twpa_off="qdmzm/calib_twpa_off_qdmzm_qd1.h5",
        calib_path_twpa_on=None,
        elec_delay=37.94e-9,
        L_ind=L_ind_QD1,
        gaussian_sigma=6,
        fdelta=3e6,
        minimum_Cq_res=2e-19,

        resonance_frequency=808.154e6,
        div_comb=False,

        data_rf_parameter="VQD1_signal",
        idler_rf_parameter=None,
    ),

    "qdmzm_qd3": CQConversionInput(
        name="qdmzm_qd3",
        data_path="qdmzm/qdmzm_qd3_Vrf.h5",
        calib_path_twpa_off="qdmzm/calib_twpa_off_qdmzm_qd3.h5",
        calib_path_twpa_on=None,
        elec_delay=30.23e-9,
        L_ind=L_ind_QD3,
        gaussian_sigma=10,
        fdelta=3e6,

        resonance_frequency=693.79e6,
        div_comb=False,

        data_rf_parameter="VQD3_signal",
        idler_rf_parameter=None,
    ),

    "drive_sweep": []

}

for drive_power in glob(str(RAW_DATA_FOLDER / "drive_sweep/drive_sweep_*_Vrf.h5")):

    drive_power = drive_power.split("_Vrf")[0].split("_")[-1]

    assert isinstance(datasets['drive_sweep'], list) # assertion to make the type checking happy

    datasets['drive_sweep'].append(
        CQConversionInput(
            name=f"drive_sweep_{drive_power}",
            data_path=f"drive_sweep/drive_sweep_{drive_power}_Vrf.h5",
            calib_path_twpa_off=f"drive_sweep/calib_twpa_off_{drive_power}.h5",
            calib_path_twpa_on=f"drive_sweep/calib_twpa_on_{drive_power}.h5",
            elec_delay=37.94e-9,
            L_ind=L_ind_QD1,
            gaussian_sigma=10,
            fdelta=3e6,

            resonance_frequency=808.15e6,
            div_comb=True
        )
    )
