"""
Contains functions used to derive time component fields, such as e.g. day of year
and hour of day.
"""
import datetime

import numpy as np
import xarray as xr
from loguru import logger


def calculate_hour_of_day(time, component):
    """
    Function for calculating hour of day features with a cyclic encoding

    Parameters
    ----------
    time: Union[xr.DataArray, datetime.datetime]
        Time
    component: str
        String indicating if the sine or cosine component of the encoding
        should be returned

    Returns
    -------
    hour_of_day_encoded: Union[xr.DataArray, float]
        sine or cosine of the hour of day
    """
    logger.info("Calculating hour of day")

    # Get the hour of the day
    if isinstance(time, xr.DataArray):
        hour_of_day = time.dt.hour
    elif isinstance(time, datetime.datetime):
        hour_of_day = time.hour
    else:
        raise TypeError(
            "Expected an instance of xr.DataArray or datetime object,"
            f" but got {type(time)}."
        )

    # Cyclic encoding of hour of day
    if component == "sin":
        hour_of_day_encoded = np.sin((hour_of_day / 24) * 2 * np.pi)
    elif component == "cos":
        hour_of_day_encoded = np.cos((hour_of_day / 24) * 2 * np.pi)
    else:
        raise ValueError(
            f"Invalid value of `component`: '{component}'. Expected one of: 'cos' or 'sin'."
            " Please update the config accordingly."
        )

    if isinstance(hour_of_day_encoded, xr.DataArray):
        # Add attributes
        hour_of_day_encoded.name = "hour_of_day_" + component
        hour_of_day_encoded.attrs[
            "long_name"
        ] = f"{component.capitalize()} component of cyclically encoded hour of day"
        hour_of_day_encoded.attrs["units"] = "1"

    return hour_of_day_encoded


def calculate_day_of_year(time, component):
    """
    Function for calculating day of year features with a cyclic encoding

    Parameters
    ----------
    time : Union[xr.DataArray, datetime.datetime]
        Time
    component: str
        String indicating if the sine or cosine component of the encoding
        should be returned

    Returns
    -------
    day_of_year_encoded: Union[xr.DataArray, float]
        sine or cosine of the day of year
    """
    logger.info("Calculating day of year")

    # Get the day of year
    if isinstance(time, xr.DataArray):
        day_of_year = time.dt.dayofyear
    elif isinstance(time, datetime.datetime):
        day_of_year = time.timetuple().tm_yday
    else:
        raise TypeError(
            "Expected an instance of xr.DataArray or datetime object,"
            f" but got {type(time)}."
        )

    # Cyclic encoding of day of year - use 366 to include leap years!
    if component == "sin":
        day_of_year_encoded = np.sin((day_of_year / 366) * 2 * np.pi)
    elif component == "cos":
        day_of_year_encoded = np.cos((day_of_year / 366) * 2 * np.pi)
    else:
        raise ValueError(
            f"Invalid value of `component`: '{component}'. Expected one of: 'cos' or 'sin'."
            " Please update the config accordingly."
        )

    if isinstance(day_of_year_encoded, xr.DataArray):
        # Add attributes
        day_of_year_encoded.name = "day_of_year_" + component
        day_of_year_encoded.attrs[
            "long_name"
        ] = f"{component.capitalize()} component of cyclically encoded day of year"
        day_of_year_encoded.attrs["units"] = "1"

    return day_of_year_encoded
