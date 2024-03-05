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
    describe what to map from (the `input_dim_map`). Finally, the function checks
    that each variable has the dimensions of `expected_input_var_dims`.

    The method of mapping is determined by the type of `input_dim_map`:

    - A string: The name of the dimension in the input dataset to map to
    - A list: The list of dimensions in the input dataset to stack to
      create the architecture dimension
    - A dictionary: This is used mapping individual dataarrays into a single
      dataarray by stacking along a coordinate. There should be exactly one of
      these maps in the `dim_mapping` dictionary. The dictionary should have
      the following keys:
      - 'dims': The list of dimensions in the input dataset to map to the
        architecture dimension
      - 'name': The string format to construct the new coordinate values
        for the architecture dimension
      - 'map_variables_to_var_name': True, the this is kept explicit so that it
        is clear in the data config into which target dimension the variables
        are mapped

    There

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to map the dimensions and variables
    input_dim_map : str, list, dict
        The mapping of the input dimensions to the architecture
        dimensions. The method of mapping is determined by the type
        of the `input_dim_map` variable.
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
        if isinstance(dim_mapping[arch_dim], dict):
            variable_dim_mappings[arch_dim] = dim_mapping.pop(arch_dim)
    if len(variable_dim_mappings) > 1:
        raise ValueError(
            "Only one mapping which requires stacking variables"
            " into a single dataarray is allowed, found ones targeting"
            " the following arch dimensions: {list(variable_dim_mappings.keys())}"
        )
    elif len(variable_dim_mappings) == 0:
        raise Exception("At least one mapping should be defined for stacking variables")

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
        if isinstance(input_dim_map, str):
            _check_for_malformed_list_arg(input_dim_map)
            # if the input_dims is a string, we assume that it is the
            # name of the dimension in the input dataset and we rename
            # it to the architecture dimension
            ds = ds.rename({input_dim_map: arch_dim})
        elif isinstance(input_dim_map, list):
            # if the input_dims is a list, we assume that it is a list of
            # dimensions in the input dataset that we want to stack to create the
            # architecture dimension, this is for example used for flatting the
            # spatial dimensions into a single dimension representing the grid
            # index
            ds = ds.stack({arch_dim: input_dim_map}).reset_index(arch_dim)

    # Finally, we handle the stacking of variables to coordinate values. We
    # might want to deal with variables that exist on multiple coordinate
    # values that we want to stack over too. The dimensions to map from are
    # expected to be given explicitly in the 'dims' key and the new string
    # format to construct the new coordinate values is given in the 'name' key.
    try:
        arch_dim, variable_dim_map = variable_dim_mappings.popitem()
        dims = variable_dim_map.get("dims", [])
        _check_for_malformed_list_arg(dims)
        name_format = variable_dim_map["name"]
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
