import numpy as np
import xarray as xr
from numpy.typing import NDArray

def create_effective_twpa_on_cal(
    cal_twpa_on: xr.DataArray | xr.Dataset,
    cal_twpa_off: xr.DataArray | xr.Dataset,
    fd: float | NDArray[np.floating],
    sig_var: str
) -> xr.DataArray | xr.Dataset:

    if isinstance(cal_twpa_on, xr.Dataset) and isinstance(cal_twpa_off, xr.Dataset):
        S11_fd_twpa_on = cal_twpa_on.sel(
            {"frequency": fd}, method="nearest"
        )[sig_var].to_numpy()
        S11_fd_twpa_off = cal_twpa_off.sel(
            {"frequency": fd}, method="nearest"
        )[sig_var].to_numpy()
    else:
        raise TypeError("cal_twpa_on and cal_twpa_off need to be xarray datasets")

    prefactor = S11_fd_twpa_on / S11_fd_twpa_off

    effective_data: xr.DataArray | xr.Dataset = prefactor * cal_twpa_off  # pyright: ignore
    return effective_data

def get_effective_calib(
        data: xr.DataArray | xr.Dataset,
        calib_off: xr.DataArray | xr.Dataset,
        calib_on: xr.DataArray | xr.Dataset,
        sig_var: str
) -> xr.DataArray | xr.Dataset:

    assert calib_off.twpa_status == 0
    assert calib_on.twpa_status == 1

    if data.twpa_status == 0:
        return calib_off

    drive_frequency = data.drive_frequency

    return create_effective_twpa_on_cal(
        cal_twpa_on=calib_on,
        cal_twpa_off=calib_off,
        fd=drive_frequency,
        sig_var=sig_var
    )
