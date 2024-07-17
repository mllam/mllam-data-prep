import xarray as xr


def stack_variables_as_coord_values(ds, name_format, combined_dim_name):
    """
    combine all variables in an xr.Dataset into a single xr.DataArray
    by stacking the variables along a new coordinate with the name given
    by `name_format` (which should include the variable name, `var_name`)


    Parameters
    ----------
    ds : xr.Dataset
        source dataset with variables to stack
    name_format : str
        format string to construct the new coordinate values for the
        stacked variables, e.g. "{var_name}_level"
    combined_dim_name : str
        name of the new dimension to create for the stacked variables, for
        example "forcing_feature"

    Returns
    -------
    da_combined : xr.DataArray
        The combined dataset with all variables stacked along the new
        coordinate
    """
    if "{var_name}" not in name_format:
        raise ValueError(
            "The name_format should include the variable name as"
            " {var_name} to construct the new coordinate values"
        )
    dataarrays = []
    for var_name in list(ds.data_vars):
        da = ds[var_name].expand_dims(combined_dim_name)
        da.coords[combined_dim_name] = [name_format.format(var_name=var_name)]

        # add extra coordinates (spanning along `combined_dim_name`) for
        # keeping track of `units` and `long_name` attributes
        for attr in ["units", "long_name"]:
            da_attr = xr.DataArray(
                [ds[var_name].attrs.get(attr, "")],
                dims=[combined_dim_name],
                coords={combined_dim_name: da.coords[combined_dim_name]},
            )
            da.coords[f"{combined_dim_name}_{attr}"] = da_attr
        dataarrays.append(da)
    da_combined = xr.concat(dataarrays, dim=combined_dim_name)

    return da_combined


def stack_variables_by_coord_values(ds, coord, name_format, combined_dim_name):
    """
    combine all variables in an xr.Dataset on all coordinate values of `coord`
    into a single xr.DataArray

    for example for a set of variables in a dataset, e.g. [u, v, t], on a set
    of "levels" in a coordinate [50, 100] the output will combine all variables
    into a single xr.DataArray with coordinate values given by the name_format
    e.g. [u_l50, u_l100, v_l50, v_l100, t_l50, t_l100] if the format was
    "{var_name}_l{level}"

    This is implemented by:
    1. iterating over all variables in the dataset
    2. for each variable, we create a new set of coordinate values which
       include the variable name and the coordinate values, and rename the
       coordinate to the `combined_dim_name`
    3. stack all the variables along the `combined_dim_name` dimension to
       produce a single xr.DataArray

    In addition to the stacked variables, we also add extra coordinates for
    keeping track of `units` and `long_name` attributes for each variable in
    `{combined_dim_name}_units` and `{combined_dim_name}_long_name`
    respectively.

    Parameters
    ----------
    ds : xr.Dataset
        dataset with variables as data_vars and `level_dim` as a coordinate
    coord : str
        name of the coordinate that should mapped over
    name_format : str
        format string to construct the new coordinate values for the
        stacked levels
    combined_dim_name : str
        name of the new dimension to create for the stacked variables

    Returns
    -------
    da_combined : xr.DataArray
        The combined dataset with the stacked variables along the `coord`
    """
    if "{var_name}" not in name_format:
        raise ValueError(
            "The name_format should include the variable name as"
            " {var_name} to construct the new coordinate values"
        )
    if f"{{{coord}}}" not in name_format:
        raise ValueError(
            "The name_format should include the coordinate name as"
            f" {{{coord}}} to construct the new coordinate values"
        )
    if coord not in ds.coords:
        raise ValueError(
            f"The coordinate {coord} is not in the dataset, found coords: {list(ds.coords)}"
        )

    datasets = []
    for var_name in list(ds.data_vars):
        da = ds[var_name]
        coord_values = da.coords[coord].values
        new_coord_values = [
            name_format.format(var_name=var_name, **{coord: val})
            for val in coord_values
        ]
        da = da.assign_coords({coord: new_coord_values}).rename(
            {coord: combined_dim_name}
        )

        # add extra coordinates for keeping track of `units` and `long_name` attributes
        for attr in ["units", "long_name"]:
            da_attr = xr.DataArray(
                [ds[var_name].attrs.get(attr, "")] * len(coord_values),
                dims=[combined_dim_name],
            )
            da.coords[f"{combined_dim_name}_{attr}"] = da_attr
        datasets.append(da)

    da_combined = xr.concat(datasets, dim=combined_dim_name)

    return da_combined
