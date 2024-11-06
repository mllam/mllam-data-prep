import dask.array as da
import numpy as np
import xarray as xr
from loguru import logger


def derive_forcings(ds, variables):
    """
    Derive the specified forcings

    Parameters
    ---------
    ds : xr.Dataset
        The loaded and subsetted dataset
    variables: list/dict
        List or dictionary with variables

    Returns
    -------
    ds : xr.Dataset
        Dataset with derived variables included
    """
    variables_to_derive = {
        k: v for elem in variables if isinstance(elem, dict) for (k, v) in elem.items()
    }

    if variables_to_derive == {}:
        pass
    else:
        logger.info("Deriving additional forcings")
        for _, derived_var in variables_to_derive.items():
            # Get the function defined in the config file
            func = globals()[derived_var.method]
            # Currently, we're passing the whole dataset
            ds = func(ds)

        # Drop all the unneeded variables that have only been used to derive the
        # forcing variables. HOWEVER, it's necessary to keep variables that are
        # also coordinates!
        derived_variable_dependencies = []
        for _, derived_var in variables_to_derive.items():
            derived_variable_dependencies += derived_var.dependencies
        variables_to_drop = [
            var
            for var in derived_variable_dependencies
            if var not in list(ds._coord_names)
        ]
        ds = ds.drop_vars(variables_to_drop)

    return ds


def derive_toa_radiation(ds):
    """
    Derive approximate TOA radiation (instantaneous values [W*m**-2])

    Parameters
    ----------
    ds : xr.Dataset
        The dataset with variables needed to derive TOA radiation

    Returns
    -------
    ds: xr.Dataset
        The dataset with TOA radiation added
    """
    logger.info("Calculating top-of-atmosphere radiation")

    # Need to construct a new dataset with chunks since
    # lat and lon are coordinates and are therefore eagerly loaded
    ds_dict = {}
    ds_dict["lat"] = (list(ds.lat.dims), da.from_array(ds.lat.values, chunks=(-1, -1)))
    ds_dict["lon"] = (list(ds.lon.dims), da.from_array(ds.lon.values, chunks=(-1, -1)))
    ds_dict["t"] = (list(ds.time.dims), da.from_array(ds.time.values, chunks=(10)))
    ds_chunks = xr.Dataset(ds_dict)

    # Calculate TOA radiation
    toa_radiation = calc_toa_radiation(ds_chunks)

    # Assign to the dataset
    ds = ds.assign(toa_radiation=toa_radiation)

    return ds


def calc_toa_radiation(ds):
    """
    Function for calculation top-of-the-atmosphere radiation

    Parameters
    ----------
    ds : xr.Dataset
        The dataset with variables needed to derive TOA radiation

    Returns
    -------
    toa_radiation: xr.DataArray
        TOA radiation data-array
    """
    # Solar constant
    E0 = 1366  # W*m**-2

    day = ds.t.dt.dayofyear
    hr_utc = ds.t.dt.hour

    # Eq. 1.6.1a in Solar Engineering of Thermal Processes 4th ed.
    dec = np.pi / 180 * 23.45 * np.sin(2 * np.pi * (284 + day) / 365)

    hr_lst = hr_utc + ds.lon / 15
    hr_angle = 15 * (hr_lst - 12)

    # Eq. 1.6.2 with beta=0 in Solar Engineering of Thermal Processes 4th ed.
    cos_sza = np.sin(ds.lat * np.pi / 180) * np.sin(dec) + np.cos(
        ds.lat * np.pi / 180
    ) * np.cos(dec) * np.cos(hr_angle * np.pi / 180)

    # Where TOA radiation is negative, set to 0
    toa_radiation = xr.where(E0 * cos_sza < 0, 0, E0 * cos_sza)

    return toa_radiation


def derive_hour_of_day(ds):
    """
    Derive hour of day features with a cyclic encoding

    Parameters
    ----------
    ds : xr.Dataset
        The dataset with variables needed to derive hour of day

    Returns
    -------
    ds: xr.Dataset
        The dataset with hour of day added
    """
    logger.info("Calculating hour of day")

    # Get the hour of the day
    hour_of_day = ds.time.dt.hour

    # Cyclic encoding of hour of day
    hour_of_day_cos, hour_of_day_sin = cyclic_encoding(hour_of_day, 24)

    # Assign to the dataset
    ds = ds.assign(hour_of_day_sin=hour_of_day_sin)
    ds = ds.assign(hour_of_day_cos=hour_of_day_cos)

    return ds


def derive_day_of_year(ds):
    """
    Derive day of year features with a cyclic encoding

    Parameters
    ----------
    ds : xr.Dataset
        The dataset with variables needed to derive day of year

    Returns
    -------
    ds: xr.Dataset
        The dataset with day of year added
    """
    logger.info("Calculating day of year")

    # Get the day of year
    day_of_year = ds.time.dt.dayofyear

    # Cyclic encoding of day of year - use 366 to include leap years!
    day_of_year_cos, day_of_year_sin = cyclic_encoding(day_of_year, 366)

    # Assign to the dataset
    ds = ds.assign(day_of_year_sin=day_of_year_sin)
    ds = ds.assign(day_of_year_cos=day_of_year_cos)

    return ds


def derive_time_of_year(ds):
    """
    Derive time of year features with a cyclic encoding

    Parameters
    ----------
    ds : xr.Dataset
        The dataset with variables needed to derive time of year

    Returns
    -------
    ds: xr.Dataset
        The dataset with time of year added
    """

    logger.info("Calculating time of year")

    # Get the number of seconds a datetime corresponds to
    number_of_seconds = (
        (ds.time.dt.dayofyear - 1) * 60 * 60 * 24
        + ds.time.dt.hour * 60 * 60
        + ds.time.dt.minute * 60
        + ds.time.dt.second
    )

    # Cyclic encoding using both leap year and non-leap year
    # number of seconds to be able to choose later where to
    # include which values using xr.where()
    time_of_year_cos_non_leap, time_of_year_sin_non_leap = cyclic_encoding(
        number_of_seconds, 31622400
    )
    time_of_year_cos_leap, time_of_year_sin_leap = cyclic_encoding(
        number_of_seconds, 31536000
    )

    time_of_year_cos = xr.where(
        ds.time.dt.is_leap_year,
        time_of_year_cos_leap,
        time_of_year_cos_non_leap,
    )
    time_of_year_sin = xr.where(
        ds.time.dt.is_leap_year,
        time_of_year_sin_leap,
        time_of_year_sin_non_leap,
    )

    # Assign to the dataset
    ds = ds.assign(time_of_year_sin=time_of_year_sin)
    ds = ds.assign(time_of_year_cos=time_of_year_cos)

    return ds


def cyclic_encoding(da, da_max):
    """
    Cyclic encoding of data

    Parameters
    ----------
    da : xr.DataArray
        xarray data-array that should be cyclically encoded
    da_max: int/float
        Maximum possible value of input data-array

    Returns
    -------
    da_cos: xr.DataArray
        Cosine part of cyclically encoded input data-array
    da_sin: xr.DataArray
        Sine part of cyclically encoded input data-array
    """

    da_sin = np.sin((da / da_max) * 2 * np.pi)
    da_cos = np.cos((da / da_max) * 2 * np.pi)

    return da_cos, da_sin
