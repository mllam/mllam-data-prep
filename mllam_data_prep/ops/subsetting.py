def subset_dataset(ds_subset, ds_input, variables):
    """
    Select specific variables from the provided the dataset, subset the
    variables along the specified coordinates and check coordinate units

    Parameters
    ----------
    ds_subset : xr.Dataset
        Subset of ds_input
    ds_input : xr.Dataset
        Input/source dataset
    variables : dict
        Dictionary with the variables to subset
        with keys as the variable names and values with entries for each
        coordinate and coordinate values to extract
    """

    if isinstance(variables, dict):
        for var, coords_to_sample in variables.items():
            da = ds_input[var]
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
            ds_subset = ds_input[variables]
        except KeyError as ex:
            raise KeyError(
                f"Could not find the all variables `{variables}` in the dataset. "
                f"The available variables are {list(ds_input.data_vars)}"
            ) from ex
    else:
        raise ValueError("The `variables` argument should be a list or a dictionary")

    return ds_subset
