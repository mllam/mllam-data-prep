"""
Handle deriving new variables (xr.DataArrays) from an individual input dataset
that has been loaded. This makes it possible to for example add fields that can
be derived from analytical expressions and are functions of coordinate values
(e.g. top-of-atmosphere incoming radiation is a function of time and lat/lon location),
but also of other physical fields (wind-speed is a function of both meridional
and zonal wind components).
"""
import importlib
import sys

import numpy as np
import xarray as xr
from loguru import logger

from ..chunking import chunk_dataset

REQUIRED_FIELD_ATTRIBUTES = ["units", "long_name"]


def derive_variables(ds, ds_input, derived_variables, chunking):
    """
    Load the dataset, and derive the specified variables

    Parameters
    ---------
    ds : xr.Dataset
        Output dataset
    ds_input : xr.Dataset
        Input/source dataset
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

    target_dims = list(ds_input.sizes.keys())

    for _, derived_variable in derived_variables.items():
        required_kwargs = derived_variable.kwargs
        function_namespace = derived_variable.function
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

        # Get subset of input dataset for calculating derived variables
        ds_subset = ds_input[required_kwargs.keys()]

        # Any coordinates needed for the derivation, for which chunking should be performed,
        # should be converted to variables since it is not possible for *indexed* coordinates
        # to be chunked dask arrays
        chunks = {
            dim: chunking.get(dim, int(ds_subset[dim].count()))
            for dim in ds_subset.dims
        }
        required_coordinates = [
            req_var for req_var in required_kwargs.keys() if req_var in ds_subset.coords
        ]
        ds_subset = ds_subset.drop_indexes(required_coordinates, errors="ignore")
        for req_coord in required_coordinates:
            if req_coord in chunks:
                ds_subset = ds_subset.reset_coords(req_coord)

        # Chunk the dataset
        ds_subset = chunk_dataset(ds_subset, chunks)

        # Add function arguments to kwargs
        kwargs = {}
        if len(latlon_coords_to_include):
            latlon = get_latlon_coords_for_input(ds_input)
            for key, val in latlon_coords_to_include.items():
                kwargs[val] = latlon[key]
        kwargs.update({val: ds_subset[key] for key, val in required_kwargs.items()})
        func = _get_derived_variable_function(function_namespace)
        # Calculate the derived variable
        derived_field = func(**kwargs)

        # Check that the derived field has the necessary attributes (REQUIRED_FIELD_ATTRIBUTES)
        # set, return any dropped/reset coordinates, align it to the output dataset dimensions
        # (if necessary) and add it to the dataset
        if isinstance(derived_field, xr.DataArray):
            derived_field = _check_for_required_attributes(
                derived_field, expected_field_attributes
            )
            derived_field = _return_dropped_coordinates(
                derived_field, ds_subset, required_coordinates, chunks
            )
            derived_field = _align_derived_variable(
                derived_field, ds_input, target_dims
            )
            ds[derived_field.name] = derived_field
        elif isinstance(derived_field, tuple) and all(
            isinstance(field, xr.DataArray) for field in derived_field
        ):
            for field in derived_field:
                field = _check_for_required_attributes(field, expected_field_attributes)
                field = _return_dropped_coordinates(
                    field, ds_subset, required_coordinates, chunks
                )
                field = _align_derived_variable(field, ds_input, target_dims)
                ds[field.name] = field
        else:
            raise TypeError(
                "Expected an instance of xr.DataArray or tuple(xr.DataArray),"
                f" but got {type(derived_field)}."
            )

    return ds


def _get_derived_variable_function(function_namespace):
    """
    Function for getting the function for deriving
    the specified variable.

    Parameters
    ----------
    function_namespace: str
        The full function namespace

    Returns
    -------
    function: object
        Function for deriving the specified variable
    """
    # Get module and function names
    module_name, _, function_name = function_namespace.rpartition(".")

    # Import the module (if necessary)
    if module_name in sys.modules:
        module = sys.modules[module_name]
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


def _return_dropped_coordinates(derived_field, ds, required_coordinates, chunks):
    """
    Return the coordinates that have been dropped/reset.

    Parameters
    ----------
    derived_field: xr.Dataset
        Derived variable
    ds: xr.Dataset
        Dataset with required coordinatwes
    required_coordinates: List[str]
        List of coordinates required for the derived variable
    chunks: Dict[str, int]
        Dictionary with keys as dimensions to be chunked and
        chunk sizes as the values

    Returns
    -------
    derived_field: xr.Dataset
        Derived variable, now also with dropped coordinates returned
    """
    for req_coord in required_coordinates:
        if req_coord in chunks:
            derived_field.coords[req_coord] = ds[req_coord]

    return derived_field


def _align_derived_variable(field, ds, target_dims):
    """
    Align a derived variable to the target dimensions (ignoring non-dimension coordinates).

    Parameters
    ----------
    field: xr.DataArray
        Derived field to align
    ds: xr.Dataset
        Target dataset
    target_dims: List[str]
        Dimensions to align to (e.g. 'time', 'y', 'x')

    Returns
    -------
    field: xr.DataArray
        The derived field aligned to the target dimensions
    """
    # Ensure that dimensions are ordered correctly
    field = field.transpose(
        *[dim for dim in target_dims if dim in field.dims], missing_dims="ignore"
    )

    # Add missing dimensions explicitly
    for dim in target_dims:
        if dim not in field.dims:
            field = field.expand_dims({dim: ds.sizes[dim]})

    # Broadcast to match only the target dimensions
    broadcast_shape = {dim: ds[dim] for dim in target_dims if dim in ds.dims}
    field = field.broadcast_like(xr.Dataset(coords=broadcast_shape))

    return field


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


def get_latlon_coords_for_input(ds):
    """Dummy function for getting lat and lon."""
    return ds[["lat", "lon"]].chunk(-1, -1)
