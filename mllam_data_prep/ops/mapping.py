from .stacking import stack_variables_as_coord_values, stack_variables_by_coord_values


def _check_for_malformed_list_arg(s):
    if isinstance(s, str) and "," in s:
        raise Exception(
            "Rather than writing `{s}` to define a list you would `[{s}]` in the config file."
        )


def map_dims_and_variables(ds, dim_mapping, expected_input_var_dims):
    """
    Map the input dimensions to the architecture dimensions
    using the `dim_mapping` dictionary. Each key in the `dim_mapping`
    describes the name of the architecture dimension to map to and the values
    describe what to map from (through a `dict` named `input_dim_map`).
    Finally, the function checks that each variable has the dimensions of
    `expected_input_var_dims`.

    Each `input_dim_map` `dict` defines how to map the input dimensions with the following
    entries:

      - 'dims': The list of dimensions in the input dataset to map to the
        architecture dimension
      - 'method': The method to use for mapping the variables to the
        architecture dimension, with the following options:
        - 'stack_variables_by_var_name':
            map variables to coordinate values by stacking the variables along
            the architecture dimension, the 'name' key should be the string
            format to construct the new coordinate values for the architecture
            dimension. Exactly one of this type of mapping should be used.
        - 'stack':
            used to map variables to the architecture dimension by stacking
            the along the `dims` provided.
        - 'rename' (or if the input_dim_map is a string naming the dimension to rename):
            rename the provided dimension to the architecture dimension (only one
            dimension must be given by `dims`)
      - 'name_format': The string format to construct the new coordinate values
        for the architecture dimension (only used for method 'stack_variables_by_var_name')

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to map the dimensions and variables
    dim_mapping : dict
        The mapping of the input dimensions to the architecture
        dimensions.
    arch_dim : str
        The name of the architecture dimension to map to
    expected_input_var_dims : list
        The list of dimensions that each variable in the input dataset
        should have

    Returns
    -------
    da: xr.DataArray
        The dataset mapped to a single data-array with coordinates given by the keys
        of the `dim_mapping` dictionary
    """

    # check that there is only one mapping defined going from the input variables
    # store it so we do that last
    dim_mapping = dim_mapping.copy()
    variable_dim_mappings = {}
    for arch_dim in list(dim_mapping.keys()):
        if dim_mapping[arch_dim].method == "stack_variables_by_var_name":
            variable_dim_mappings[arch_dim] = dim_mapping.pop(arch_dim)
    if len(variable_dim_mappings) > 1:
        raise ValueError(
            "Only one mapping which requires stacking variables"
            " into a single dataarray is allowed, found ones targeting"
            f" the following arch dimensions: {list(variable_dim_mappings.keys())}"
        )
    elif len(variable_dim_mappings) == 0:
        raise Exception(
            "At least one mapping should be defined for stacking variables, i.e. uses"
            f" the method `stack_variables_by_var_name`. Current mapping is: {dim_mapping}"
        )

    # check that none of the variables have dims that are not in the expected_input_var_dims
    for var_name in ds.data_vars:
        if not set(ds[var_name].dims).issubset(expected_input_var_dims):
            extra_dims = set(ds[var_name].dims) - set(expected_input_var_dims)
            raise ValueError(
                f"The variable {var_name} has dimensions {ds[var_name].dims} however the"
                f" dimensions `{extra_dims}` are not in "
                f" the `dims` defined for this input dataset: {expected_input_var_dims}"
            )

    # handle those mappings that involve just renaming or stacking dimensions
    for arch_dim, input_dim_map in dim_mapping.items():
        method = input_dim_map.method

        if method == "rename":
            source_dim = input_dim_map.dim
            ds = ds.rename({source_dim: arch_dim})
        elif method == "stack":
            source_dims = input_dim_map.dims
            # when stacking we assume that the input_dims is a list of dimensions
            # in the input dataset that we want to stack to create the architecture
            # dimension, this is for example used for flatting the spatial dimensions
            # into a single dimension representing the grid index
            ds = ds.stack({arch_dim: source_dims}).reset_index(arch_dim)
        else:
            raise NotImplementedError(method)

    # Finally, we handle the stacking of variables to coordinate values. We
    # might want to deal with variables that exist on multiple coordinate
    # values that we want to stack over too. The dimensions to map from are
    # expected to be given explicitly in the 'dims' key and the new string
    # format to construct the new coordinate values is given in the 'name' key.
    try:
        arch_dim, variable_dim_map = variable_dim_mappings.popitem()
        dims = variable_dim_map.dims or []
        _check_for_malformed_list_arg(dims)
        name_format = variable_dim_map.name_format
        if len(dims) == 0:
            da = stack_variables_as_coord_values(
                ds=ds, name_format=name_format, combined_dim_name=arch_dim
            )
        elif len(dims) == 1:
            da = stack_variables_by_coord_values(
                ds=ds,
                coord=dims[0],
                name_format=name_format,
                combined_dim_name=arch_dim,
            )
        else:
            # TODO: this will have to involved xrarrays MultiIndex, but lets leave
            # this until we need it
            raise NotImplementedError(len(dims))
        # set a flag we can use later to identify which coordinate the variables
        # were mapped into
        da.attrs["variables_mapping_dim"] = arch_dim
    except ValueError as ex:
        raise Exception(
            f"There was an issue handling the following mapping:\n{variable_dim_map}"
            f"\n from variables {list(ds.data_vars)} and dims {list(ds.dims)}"
        ) from ex

    return da
