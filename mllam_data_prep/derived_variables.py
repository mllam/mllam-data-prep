import importlib
import sys

import numpy as np
import xarray as xr
from loguru import logger


def derive_variables(fp, derived_variables, chunking):
    """
    Load the dataset, and derive the specified variables

    Parameters
    ---------
    fp : str
        Filepath to the source dataset, for example the path to a zarr dataset
        or a netCDF file (anything that is supported by `xarray.open_dataset` will work)
    derived_variables : dict
        Dictionary with the variables to derive
        with keys as the variable names and values with entries for
        kwargs and function to use in the calculation
    chunking: dict
        Dictionary with keys as the dimensions to chunk along and values
        with the chunk size

    Returns
    -------
    ds : xr.Dataset
        Dataset with derived variables included
    """
    logger.info("Deriving variables")

    try:
        ds = xr.open_zarr(fp)
    except ValueError:
        ds = xr.open_dataset(fp)

    ds_subset = xr.Dataset()
    ds_subset.attrs.update(ds.attrs)
    for _, derived_variable in derived_variables.items():
        required_variables = derived_variable.kwargs
        function_name = derived_variable.function
        ds_input = ds[required_variables.keys()]

        # Any coordinates needed for the derivation, for which chunking should be performed,
        # should be converted to variables since it is not possible for coordinates to be
        # chunked dask arrays
        chunks = {d: chunking.get(d, int(ds_input[d].count())) for d in ds_input.dims}
        required_coordinates = [
            req_var
            for req_var in required_variables.keys()
            if req_var in ds_input.coords
        ]
        ds_input = ds_input.drop_indexes(required_coordinates, errors="ignore")
        for req_coord in required_coordinates:
            if req_coord in chunks:
                ds_input = ds_input.reset_coords(req_coord)

        # Chunk the data variables
        ds_input = ds_input.chunk(chunks)

        # Calculate the derived variable
        kwargs = {v: ds_input[k] for k, v in required_variables.items()}
        func = get_derived_variable_function(function_name)
        derived_field = func(**kwargs)

        # Some of the derived variables include two components, since
        # they are cyclically encoded (cos and sin parts)
        if isinstance(derived_field, xr.DataArray):
            derived_field = _return_dropped_coordinates(
                derived_field, ds_input, required_coordinates, chunks
            )
            ds_subset[derived_field.name] = derived_field
        elif isinstance(derived_field, tuple):
            for field in derived_field:
                field = _return_dropped_coordinates(
                    field, ds_input, required_coordinates, chunks
                )
                ds_subset[field.name] = field

    return ds_subset


def get_derived_variable_function(function_namespace):
    """
    Function for returning the function to be used to derive
    the specified variable.

    1. Check if the function to use is in globals()
    2. If it is in globals then call it
    3. If it isn't in globals() then import the necessary module
        before calling it
    """
    # Get the name of the calling module
    calling_module = globals()["__name__"]

    if "." in function_namespace:
        # If the function name is a full namespace, get module and function names
        module_name, function_name = function_namespace.rsplit(".", 1)

        # Check if the module_name is pointing to here (the calling module),
        # and if it does then use globals() to get the function otherwise
        # import the correct module and get the correct function
        if module_name == calling_module:
            function = globals().get(function_name)
        else:
            # Check if the module is already imported
            if module_name in sys.modules:
                module = module_name
            else:
                module = importlib.import_module(module_name)

            # Get the function from the module
            function = getattr(module, function_name)
    else:
        # If function name only get it from the calling module (here)
        function = globals().get(function_namespace)
        if not function:
            raise TypeError(
                f"Function '{function_namespace}' was not found in '{calling_module}'."
                f" Check that you have specified the correct function name"
                " and/or that you have defined the full function namespace if you"
                " want to use a function defined outside of of the current module"
                f" '{calling_module}'."
            )

    return function


def _return_dropped_coordinates(derived_field, ds_input, required_coordinates, chunks):
    """Return coordinates that have been reset."""
    for req_coord in required_coordinates:
        if req_coord in chunks:
            derived_field.coords[req_coord] = ds_input[req_coord]

    return derived_field


def calculate_toa_radiation(lat, lon, time):
    """
    Function for calculating top-of-the-atmosphere radiation

    Parameters
    ----------
    lat : xr.DataArray or float
        Latitude values
    lon : xr.DataArray or float
        Longitude values
    time : xr.DataArray or datetime object
        Time

    Returns
    -------
    toa_radiation: xr.DataArray or float
        TOA radiation data
    """
    logger.info("Calculating top-of-atmosphere radiation")

    # Solar constant
    E0 = 1366  # W*m**-2

    day = time.dt.dayofyear
    hr_utc = time.dt.hour

    # Eq. 1.6.1a in Solar Engineering of Thermal Processes 4th ed.
    dec = np.pi / 180 * 23.45 * np.sin(2 * np.pi * (284 + day) / 365)

    hr_lst = hr_utc + lon / 15
    hr_angle = 15 * (hr_lst - 12)

    # Eq. 1.6.2 with beta=0 in Solar Engineering of Thermal Processes 4th ed.
    cos_sza = np.sin(lat * np.pi / 180) * np.sin(dec) + np.cos(
        lat * np.pi / 180
    ) * np.cos(dec) * np.cos(hr_angle * np.pi / 180)

    # Where TOA radiation is negative, set to 0
    toa_radiation = xr.where(E0 * cos_sza < 0, 0, E0 * cos_sza)

    if isinstance(toa_radiation, xr.DataArray):
        # Add attributes
        toa_radiation.name = "toa_radiation"

    return toa_radiation


def calculate_hour_of_day(time):
    """
    Function for calculating hour of day features with a cyclic encoding

    Parameters
    ----------
    time : xr.DataArray or datetime object
        Time

    Returns
    -------
    hour_of_day_cos: xr.DataArray or float
        cosine of the hour of day
    hour_of_day_sin: xr.DataArray or float
        sine of the hour of day
    """
    logger.info("Calculating hour of day")

    # Get the hour of the day
    hour_of_day = time.dt.hour

    # Cyclic encoding of hour of day
    hour_of_day_cos, hour_of_day_sin = cyclic_encoding(hour_of_day, 24)

    if isinstance(hour_of_day_cos, xr.DataArray):
        # Add attributes
        hour_of_day_cos.name = "hour_of_day_cos"

    if isinstance(hour_of_day_sin, xr.DataArray):
        # Add attributes
        hour_of_day_sin.name = "hour_of_day_sin"

    return hour_of_day_cos, hour_of_day_sin


def calculate_day_of_year(time):
    """
    Function for calculating day of year features with a cyclic encoding

    Parameters
    ----------
    time : xr.DataArray or datetime object
        Time

    Returns
    -------
    day_of_year_cos: xr.DataArray or float
        cosine of the day of year
    day_of_year_sin: xr.DataArray or float
        sine of the day of year
    """
    logger.info("Calculating day of year")

    # Get the day of year
    day_of_year = time.dt.dayofyear

    # Cyclic encoding of day of year - use 366 to include leap years!
    day_of_year_cos, day_of_year_sin = cyclic_encoding(day_of_year, 366)

    if isinstance(day_of_year_cos, xr.DataArray):
        # Add attributes
        day_of_year_cos.name = "day_of_year_cos"

    if isinstance(day_of_year_sin, xr.DataArray):
        # Add attributes
        day_of_year_sin.name = "day_of_year_sin"

    return day_of_year_cos, day_of_year_sin


def cyclic_encoding(data_array, da_max):
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

    data_array_sin = np.sin((data_array / da_max) * 2 * np.pi)
    data_array_cos = np.cos((data_array / da_max) * 2 * np.pi)

    return data_array_cos, data_array_sin
