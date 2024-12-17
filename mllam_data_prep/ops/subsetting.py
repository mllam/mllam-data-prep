import xarray as xr


def subset_dataset(ds, variables, chunking):
    """
    Select specific variables from the provided the dataset, subset the
    variables along the specified coordinates and check coordinate units

    Parameters
    ----------
    ds : xr.Dataset
        Source dataset
    variables : dict
        Dictionary with the variables to subset
        with keys as the variable names and values with entries for each
        coordinate and coordinate values to extract
    chunking: dict
        Dictionary with keys as the dimensions to chunk along and values
        with the chunk size
    """

    ds_subset = xr.Dataset()
    ds_subset.attrs.update(ds.attrs)
    if isinstance(variables, dict):
        for var, coords_to_sample in variables.items():
            da = ds[var]
            for coord, sampling in coords_to_sample.items():
                coord_values = sampling.values
                try:
                    da = da.sel(**{coord: coord_values})
                except KeyError as ex:
                    raise KeyError(
                        f"Could not find the all coordinate values `{coord_values}` in "
                        f"coordinate `{coord}` in the dataset"
                    ) from ex
                expected_units = sampling.units
                coord_units = da[coord].attrs.get("units", None)
                if coord_units is not None and coord_units != expected_units:
                    raise ValueError(
                        f"Expected units {expected_units} for coordinate {coord}"
                        f" in variable {var} but got {coord_units}"
                    )
            ds_subset[var] = da
    elif isinstance(variables, list):
        try:
            ds_subset = ds[variables]
        except KeyError as ex:
            raise KeyError(
                f"Could not find the all variables `{variables}` in the dataset. "
                f"The available variables are {list(ds.data_vars)}"
            ) from ex
    else:
        raise ValueError("The `variables` argument should be a list or a dictionary")

    chunks = {
        dim: chunking.get(dim, int(ds_subset[dim].count())) for dim in ds_subset.dims
    }
    ds_subset = ds_subset.chunk(chunks)

    return ds_subset
