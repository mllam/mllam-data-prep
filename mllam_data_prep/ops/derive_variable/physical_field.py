"""
Contains functions used to derive physical fields. This can be both
fields that can be derived from analytical expressions and are functions
of coordinate values (e.g. top-of-atmosphere incoming radiation is a function
of time and lat/lon location), but also of other physical fields, such as
wind speed, which is a function of both meridional and zonal wind components.
"""
import datetime

import numpy as np
import xarray as xr
from loguru import logger


def calculate_toa_radiation(lat, lon, time):
    """
    Function for calculating top-of-atmosphere incoming radiation

    Parameters
    ----------
    lat : Union[xr.DataArray, float]
        Latitude values. Should be in the range [-90, 90]
    lon : Union[xr.DataArray, float]
        Longitude values. Should be in the range [-180, 180] or [0, 360]
    time : Union[xr.DataArray, datetime.datetime]
        Time

    Returns
    -------
    toa_radiation : Union[xr.DataArray, float]
        Top-of-atmosphere incoming radiation
    """
    logger.info("Calculating top-of-atmosphere incoming radiation")

    # Solar constant
    solar_constant = 1366  # W*m**-2

    # Different handling if xr.DataArray or datetime object
    if isinstance(time, xr.DataArray):
        day = time.dt.dayofyear
        hour_utc = time.dt.hour
    elif isinstance(time, datetime.datetime):
        day = time.timetuple().tm_yday
        hour_utc = time.hour
    else:
        raise TypeError(
            "Expected an instance of xr.DataArray or datetime object,"
            f" but got {type(time)}."
        )

    # Eq. 1.6.1a in Solar Engineering of Thermal Processes 4th ed.
    # dec: declination - angular position of the sun at solar noon w.r.t.
    # the plane of the equator
    dec = np.pi / 180 * 23.45 * np.sin(2 * np.pi * (284 + day) / 365)

    utc_solar_time = hour_utc + lon / 15
    hour_angle = 15 * (utc_solar_time - 12)

    # Eq. 1.6.2 with beta=0 in Solar Engineering of Thermal Processes 4th ed.
    # cos_sza: Cosine of solar zenith angle
    cos_sza = np.sin(lat * np.pi / 180) * np.sin(dec) + np.cos(
        lat * np.pi / 180
    ) * np.cos(dec) * np.cos(hour_angle * np.pi / 180)

    # Where TOA radiation is negative, set to 0
    toa_radiation = xr.where(solar_constant * cos_sza < 0, 0, solar_constant * cos_sza)

    if isinstance(toa_radiation, xr.DataArray):
        # Add attributes
        toa_radiation.name = "toa_radiation"
        toa_radiation.attrs["long_name"] = "top-of-atmosphere incoming radiation"
        toa_radiation.attrs["units"] = "W*m**-2"

    return toa_radiation
