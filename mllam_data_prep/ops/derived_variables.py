"""
Handle deriving new variables (xr.DataArrays) from an individual input dataset
that has been loaded. This makes it possible to for example add fields that can
be derived from analytical expressions and are functions of coordinate values
(e.g. top-of-atmosphere incoming radiation is a function of time and lat/lon location),
but also of other physical fields (wind-speed is a function of both meridional
and zonal wind components).
"""
import datetime
import importlib
import sys

import numpy as np
import xarray as xr
from loguru import logger

from .chunking import check_chunk_size

REQUIRED_FIELD_ATTRIBUTES = ["units", "long_name"]


def derive_variables(ds, derived_variables, chunking):
    """
    Load the dataset, and derive the specified variables

    Parameters
    ---------
    ds : xr.Dataset
        Source dataset
    derived_variables : Dict[str, DerivedVariable]
        Dictionary with the variables to derive with keys as the variable
        names and values with entries for kwargs and function to use in
        the calculation
    chunking: Dict[str, int]
        Dictionary with keys as the dimensions to chunk along and values
        with the chunk size

    Returns
    -------
    xr.Dataset
        Dataset with derived variables included
    """

    ds_derived_vars = xr.Dataset()
    ds_derived_vars.attrs.update(ds.attrs)
    for _, derived_variable in derived_variables.items():
        required_kwargs = derived_variable.kwargs
        function_name = derived_variable.function
        expected_field_attributes = derived_variable.attrs

        # Separate the lat,lon from the required variables as these will be derived separately
        logger.warning(
            "Assuming that the lat/lon coordinates are given as variables called"
            " 'lat' and 'lon'."
        )
        latlon_coords_to_include = {}
        for key in list(required_kwargs.keys()):
            if key in ["lat", "lon"]:
                latlon_coords_to_include[key] = required_kwargs.pop(key)

        # Get input dataset for calculating derived variables
        ds_input = ds[required_kwargs.keys()]

        # Any coordinates needed for the derivation, for which chunking should be performed,
        # should be converted to variables since it is not possible for *indexed* coordinates
        # to be chunked dask arrays
        chunks = {
            dim: chunking.get(dim, int(ds_input[dim].count())) for dim in ds_input.dims
        }
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
            for key, val in latlon_coords_to_include.items():
                kwargs[val] = latlon[key]
        kwargs.update({val: ds_input[key] for key, val in required_kwargs.items()})
        func = _get_derived_variable_function(function_name)
        # Calculate the derived variable
        derived_field = func(**kwargs)

        # Check that the derived field has the necessary attributes (REQUIRED_FIELD_ATTRIBUTES)
        # set and add it to the dataset
        if isinstance(derived_field, xr.DataArray):
            derived_field = _check_for_required_attributes(
                derived_field, expected_field_attributes
            )
            ds_derived_vars[derived_field.name] = derived_field
        elif isinstance(derived_field, tuple) and all(
            isinstance(field, xr.DataArray) for field in derived_field
        ):
            for field in derived_field:
                field = _check_for_required_attributes(field, expected_field_attributes)
                ds_derived_vars[field.name] = field
        else:
            raise TypeError(
                "Expected an instance of xr.DataArray or tuple(xr.DataArray),"
                f" but got {type(derived_field)}."
            )

        # Add back dropped coordinates
        ds_derived_vars = _return_dropped_coordinates(
            ds_derived_vars, ds_input, required_coordinates, chunks
        )

    return ds_derived_vars


def _chunk_dataset(ds, chunks):
    """
    Check the chunk size and chunk dataset.

    Parameters
    ----------
    ds: xr.Dataset
        Dataset to be chunked
    chunks: Dict[str, int]
        Dictionary with keys as dimensions to be chunked and
        chunk sizes as the values

    Returns
    -------
    ds: xr.Dataset
        Dataset with chunking applied
    """
    # Check the chunk size
    check_chunk_size(ds, chunks)

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

    # Get module and function names
    module_name, _, function_name = function_namespace.rpartition(".")

    # Check if the module_name is pointing to here (the calling module or empty "")
    # If it does, then use globals() to get the function otherwise import the
    # correct module and get the correct function
    if module_name in [calling_module, ""]:
        function = globals().get(function_name)
        if not function:
            raise TypeError(
                f"Function '{function_namespace}' was not found in '{calling_module}'."
                f" Check that you have specified the correct function name"
                " and/or that you have defined the full function namespace if you"
                " want to use a function defined outside of of the current module"
                f" '{calling_module}'."
            )
    else:
        # Check if the module is already imported
        if module_name in sys.modules:
            module = module_name
        else:
            module = importlib.import_module(module_name)

        # Get the function from the module
        function = getattr(module, function_name)

    return function


def _check_for_required_attributes(field, expected_attributes):
    """
    Check the attributes of the derived variable.

    Parameters
    ----------
    field: xr.DataArray
        The derived field
    expected_attributes: Dict[str, str]
        Dictionary with expected attributes for the derived variables.
        Defined in the config file.

    Returns
    -------
    field: xr.DataArray
        The derived field
    """
    for attribute in REQUIRED_FIELD_ATTRIBUTES:
        if attribute not in field.attrs or field.attrs[attribute] is None:
            if attribute in expected_attributes.keys():
                field.attrs[attribute] = expected_attributes[attribute]
            else:
                # The expected attributes are empty and the attributes have not been
                # set during the calculation of the derived variable
                raise KeyError(
                    f'The attribute "{attribute}" has not been set for the derived'
                    f' variable "{field.name}". This is most likely because you are'
                    " using a function external to `mlllam-data-prep` to derive the field,"
                    f" in which the required attributes ({', '.join(REQUIRED_FIELD_ATTRIBUTES)})"
                    " are not set. If they are not set in the function call when deriving the field,"
                    ' they can be set in the config file by adding an "attrs" section under the'
                    f' "{field.name}" derived variable section. For example, if the required attributes'
                    f" ({', '.join(REQUIRED_FIELD_ATTRIBUTES)}) are not set for a derived variable named"
                    f' "toa_radiation" they can be set by adding the following to the config file:'
                    ' {"attrs": {"units": "W*m**-2", "long_name": "top-of-atmosphere incoming radiation"}}.'
                )
        elif attribute in expected_attributes.keys():
            logger.warning(
                f"The attribute '{attribute}' of the derived field"
                f" {field.name} is being overwritten from"
                f" '{field.attrs[attribute]}' to"
                f" '{expected_attributes[attribute]}' according"
                " to the specification in the config file."
            )
            field.attrs[attribute] = expected_attributes[attribute]
        else:
            # Attributes are set in the funciton and nothing has been defined in the config file
            pass

    return field


def _return_dropped_coordinates(
    ds_derived_vars, ds_input, required_coordinates, chunks
):
    """
    Return the coordinates that have been reset.

    Parameters
    ----------
    ds_derived_vars: xr.Dataset
        Dataset with derived variables
    ds_input: xr.Dataset
        Input dataset for deriving variables
    required_coordinates: List[str]
        List of coordinates required for the derived variable
    chunks: Dict[str, int]
        Dictionary with keys as dimensions to be chunked and
        chunk sizes as the values

    Returns
    -------
    ds_derived_vars: xr.Dataset
        Dataset with derived variables, now also with dropped coordinates returned
    """
    for req_coord in required_coordinates:
        if req_coord in chunks:
            ds_derived_vars.coords[req_coord] = ds_input[req_coord]

    return ds_derived_vars


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


def calculate_hour_of_day(time):
    """
    Function for calculating hour of day features with a cyclic encoding

    Parameters
    ----------
    time : Union[xr.DataArray, datetime.datetime]
        Time

    Returns
    -------
    hour_of_day_cos: Union[xr.DataArray, float]
        cosine of the hour of day
    hour_of_day_sin: Union[xr.DataArray, float]
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
    time : Union[xr.DataArray, datetime.datetime]
        Time

    Returns
    -------
    day_of_year_cos: Union[xr.DataArray, float]
        cosine of the day of year
    day_of_year_sin: Union[xr.DataArray, float]
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
    data : Union[xr.DataArray, float, int]
        Data that should be cyclically encoded
    data_max: Union[int, float]
        Maximum possible value of input data. Should be greater than 0.

    Returns
    -------
    data_cos: Union[xr.DataArray, float, int]
        Cosine part of cyclically encoded input data
    data_sin: Union[xr.DataArray, float, int]
        Sine part of cyclically encoded input data
    """

    data_sin = np.sin((data / data_max) * 2 * np.pi)
    data_cos = np.cos((data / data_max) * 2 * np.pi)

    return data_cos, data_sin


def get_latlon_coords_for_input(ds_input):
    """Dummy function for getting lat and lon."""
    return ds_input[["lat", "lon"]].chunk(-1, -1)
