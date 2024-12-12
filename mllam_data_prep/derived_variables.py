import datetime
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
        required_kwargs = derived_variable.kwargs
        function_name = derived_variable.function
        derived_variable_attributes = derived_variable.attributes or {}

        # Separate the lat,lon from the required variables as these will be derived separately
        latlon_coords_to_include = {}
        for k, v in list(required_kwargs.items()):
            if k in ["lat", "lon"]:
                latlon_coords_to_include[k] = required_kwargs.pop(k)

        # Subset the dataset
        ds_input = ds[required_kwargs.keys()]

        # Any coordinates needed for the derivation, for which chunking should be performed,
        # should be converted to variables since it is not possible for *indexed* coordinates
        # to be chunked dask arrays
        chunks = {d: chunking.get(d, int(ds_input[d].count())) for d in ds_input.dims}
        required_coordinates = [
            req_var for req_var in required_kwargs.keys() if req_var in ds_input.coords
        ]
        ds_input = ds_input.drop_indexes(required_coordinates, errors="ignore")
        for req_coord in required_coordinates:
            if req_coord in chunks:
                ds_input = ds_input.reset_coords(req_coord)

        # Chunk the dataset
        ds_input = _chunk_dataset(ds_input, chunks)

        # Add function arguments to kwargs
        kwargs = {}
        if len(latlon_coords_to_include):
            latlon = get_latlon_coords_for_input(ds)
            for k, v in latlon_coords_to_include.items():
                kwargs[v] = latlon[k]
        kwargs.update({v: ds_input[k] for k, v in required_kwargs.items()})
        func = _get_derived_variable_function(function_name)
        # Calculate the derived variable
        derived_field = func(**kwargs)

        # Check the derived field(s)
        derived_field = _check_field(derived_field, derived_variable_attributes)

        # Add the derived field(s) to the subset
        if isinstance(derived_field, xr.DataArray):
            ds_subset[derived_field.name] = derived_field
        elif isinstance(derived_field, tuple) and all(
            isinstance(field, xr.DataArray) for field in derived_field
        ):
            for field in derived_field:
                ds_subset[field.name] = field
        else:
            raise TypeError(
                "Expected an instance of xr.DataArray or tuple(xr.DataArray),"
                f" but got {type(derived_field)}."
            )

        # Add back dropped coordinates
        ds_subset = _return_dropped_coordinates(
            ds_subset, ds_input, required_coordinates, chunks
        )

    return ds_subset


def _chunk_dataset(ds, chunks):
    """
    Chunk dataset and check the chunk size.

    Parameters
    ----------
    ds: xr.Dataset
        Dataset to be chunked
    chunks: dict
        Dictionary with keys as dimensions to be chunked and
        chunk sizes as the values

    Returns
    -------
    ds: xr.Dataset
        Dataset with chunking applied
    """
    # Define the memory limit check
    memory_limit_check = 1 * 1024**3  # 1 GB

    # Check the chunk size
    for var_name, var_data in ds.data_vars.items():
        total_size = 1

        for dim, chunk_size in chunks.items():
            dim_size = ds.sizes.get(dim, None)
            if dim_size is None:
                raise KeyError(f"Dimension '{dim}' not found in the dataset.")
            total_size *= chunk_size

        dtype = var_data.dtype
        bytes_per_element = np.dtype(dtype).itemsize

        memory_usage = total_size * bytes_per_element

        if memory_usage > memory_limit_check:
            logger.warning(
                f"The chunk size for '{var_name}' exceeds '{memory_limit_check}' GB."
            )

    # Try chunking
    try:
        ds = ds.chunk(chunks)
    except Exception as ex:
        raise Exception(f"Error chunking dataset: {ex}")

    return ds


def _get_derived_variable_function(function_namespace):
    """
    Function for getting the function for deriving
    the specified variable.

    Parameters
    ----------
    function_namespace: str
        The full function namespace or just the function name
        if it is a function included in this module.

    Returns
    -------
    function: object
        Function for deriving the specified variable
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


def _check_field(derived_field, derived_field_attributes):
    """
    Check the derived field.

    Parameters
    ----------
    derived_field: xr.DataArray or tuple
        The derived variable
    derived_field_attributes: dict
        Dictionary with attributes for the derived variables.
        Defined in the config file.

    Returns
    -------
    derived_field: xr.DataArray or tuple
        The derived field
    """
    if isinstance(derived_field, xr.DataArray):
        derived_field = _check_attributes(derived_field, derived_field_attributes)
    elif isinstance(derived_field, tuple) and all(
        isinstance(field, xr.DataArray) for field in derived_field
    ):
        for field in derived_field:
            field = _check_attributes(field, derived_field_attributes)
    else:
        raise TypeError(
            "Expected an instance of xr.DataArray or tuple(xr.DataArray),"
            f" but got {type(derived_field)}."
        )

    return derived_field


def _check_attributes(field, field_attributes):
    """
    Check the attributes of the derived variable.

    Parameters
    ----------
    field: xr.DataArray or tuple
        The derived field
    field_attributes: dict
        Dictionary with attributes for the derived variables.
        Defined in the config file.

    Returns
    -------
    field: xr.DataArray or tuple
        The derived field
    """
    for attribute in ["units", "long_name"]:
        if attribute not in field.attrs or field.attrs[attribute] is None:
            if attribute in field_attributes.keys():
                field.attrs[attribute] = field_attributes[attribute]
            else:
                # The expected attributes are empty and the attributes have not been
                # set during the calculation of the derived variable
                raise ValueError(
                    f"The attribute '{attribute}' has not been set for the derived"
                    f" variable '{field.name}' (most likely because you are using a"
                    " function external to `mlllam-data-prep` to derive the field)."
                    " This attribute has not been defined in the 'attributes' section"
                    " of the config file either. Make sure that you add it to the"
                    f" 'attributes' section of the derived variable '{field.name}'."
                )
        else:
            if attribute in field_attributes.keys():
                logger.warning(
                    f"The attribute '{attribute}' of the derived field"
                    f" {field.name} is being overwritten from"
                    f" '{field.attrs[attribute]}' to"
                    f" '{field_attributes[attribute]}' according"
                    " to specification in the config file."
                )
                field.attrs[attribute] = field_attributes[attribute]
            else:
                # Attributes are set and nothing has been defined in the config file
                pass

    return field


def _return_dropped_coordinates(ds_subset, ds_input, required_coordinates, chunks):
    """Return the coordinates that have been reset."""
    for req_coord in required_coordinates:
        if req_coord in chunks:
            ds_subset.coords[req_coord] = ds_input[req_coord]

    return ds_subset


def calculate_toa_radiation(lat, lon, time):
    """
    Function for calculating top-of-the-atmosphere radiation

    Parameters
    ----------
    lat : xr.DataArray or float
        Latitude values. Should be in the range [-90, 90]
    lon : xr.DataArray or float
        Longitude values. Should be in the range [-180, 180] or [0, 360]
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

    # Different handling if xr.DataArray or datetime object
    if isinstance(time, xr.DataArray):
        day = time.dt.dayofyear
        hr_utc = time.dt.hour
    elif isinstance(time, datetime.datetime):
        day = time.timetuple().tm_yday
        hr_utc = time.hour
    else:
        raise TypeError(
            "Expected an instance of xr.DataArray or datetime object,"
            f" but got {type(time)}."
        )

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
        toa_radiation.attrs["long_name"] = "top-of-the-atmosphere radiation"
        toa_radiation.attrs["units"] = "W*m**-2"

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
    hour_of_day_cos, hour_of_day_sin = cyclic_encoding(hour_of_day, 24)

    if isinstance(hour_of_day_cos, xr.DataArray):
        # Add attributes
        hour_of_day_cos.name = "hour_of_day_cos"
        hour_of_day_cos.attrs[
            "long_name"
        ] = "Cosine component of cyclically encoded hour of day"
        hour_of_day_cos.attrs["units"] = "1"

    if isinstance(hour_of_day_sin, xr.DataArray):
        # Add attributes
        hour_of_day_sin.name = "hour_of_day_sin"
        hour_of_day_sin.attrs[
            "long_name"
        ] = "Sine component of cyclically encoded hour of day"
        hour_of_day_sin.attrs["units"] = "1"

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
    day_of_year_cos, day_of_year_sin = cyclic_encoding(day_of_year, 366)

    if isinstance(day_of_year_cos, xr.DataArray):
        # Add attributes
        day_of_year_cos.name = "day_of_year_cos"
        day_of_year_cos.attrs[
            "long_name"
        ] = "Cosine component of cyclically encoded day of year"
        day_of_year_cos.attrs["units"] = "1"

    if isinstance(day_of_year_sin, xr.DataArray):
        # Add attributes
        day_of_year_sin.name = "day_of_year_sin"
        day_of_year_sin.attrs[
            "long_name"
        ] = "Sine component of cyclically encoded day of year"
        day_of_year_sin.attrs["units"] = "1"

    return day_of_year_cos, day_of_year_sin


def cyclic_encoding(data, data_max):
    """
    Cyclic encoding of data

    Parameters
    ----------
    data : xr.DataArray, float, or int
        Data that should be cyclically encoded
    data_max: int or float
        Maximum possible value of input data. Should be greater than 0.

    Returns
    -------
    data_cos: xr.DataArray, float, or int
        Cosine part of cyclically encoded input data
    data_sin: xr.DataArray, float, or int
        Sine part of cyclically encoded input data
    """

    data_sin = np.sin((data / data_max) * 2 * np.pi)
    data_cos = np.cos((data / data_max) * 2 * np.pi)

    return data_cos, data_sin


def get_latlon_coords_for_input(ds_input):
    """Dummy function for getting lat and lon."""
    return ds_input[["lat", "lon"]].chunk(-1, -1)
