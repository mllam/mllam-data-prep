from collections import defaultdict

import xarray as xr
import yaml


def _load_and_subset_dataset(fp, variables):
    """
    Load the dataset, subset the variables along the specified coordinates and
    check coordinate units

    Parameters
    ----------
    fp : str
        Filepath to the zarr dataset
    variables : dict
        Dictionary with the variables to subset
        with keys as the variable names and values with entries for each
        coordinate and coordinate values to extract
    """

    ds = xr.open_zarr(fp)
    ds_subset = xr.Dataset()
    ds_subset.attrs.update(ds.attrs)
    if isinstance(variables, dict):
        for var, coords_to_sample in variables.items():
            da = ds[var]
            for coord, sampling in coords_to_sample.items():
                coord_values = sampling["sel"]
                try:
                    da = da.sel(**{coord: coord_values})
                except KeyError as ex:
                    raise KeyError(
                        f"Could not find the all coordinate values `{coord_values}` in "
                        f"coordinate `{coord}` in the dataset"
                    ) from ex
                expected_units = sampling.get("units", None)
                coord_units = da[coord].attrs.get("units", None)
                if coord_units is not None and coord_units != expected_units:
                    raise ValueError(
                        f"Expected units {expected_units} for coordinate {coord}"
                        f" in variable {var} but got {coord_units}"
                    )
            ds_subset[var] = da
    elif isinstance(variables, list):
        ds_subset = ds[variables]
    else:
        raise ValueError("The `variables` argument should be a list or a dictionary")
    return ds_subset


def _stack_variables_as_coord_values(ds, name_format, combined_dim_name):
    """
    combine all variables in an xr.Dataset into a single xr.DataArray
    by stacking the variables along a new coordinate with the name given
    by `name_format` (which should include the variable name, `var_name`)
    """
    if "{var_name}" not in name_format:
        raise ValueError(
            "The name_format should include the variable name as"
            " {var_name} to construct the new coordinate values"
        )
    dataarrays = []
    for var_name in list(ds.data_vars):
        da = ds[var_name]
        da.coords[combined_dim_name] = name_format.format(var_name=var_name)
        dataarrays.append(da)
    da_combined = xr.concat(dataarrays, dim=combined_dim_name)
    return da_combined


def _stack_variables_by_coord_values(ds, coord, name_format, combined_dim_name):
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
    for var in list(ds.data_vars):
        da = ds[var]
        coord_values = da.coords[coord].values
        new_coord_values = [
            name_format.format(var_name=var, **{coord: val}) for val in coord_values
        ]
        da = da.assign_coords({coord: new_coord_values}).rename(
            {coord: combined_dim_name}
        )
        datasets.append(da)

    da_combined = xr.concat(datasets, dim=combined_dim_name)

    return da_combined


def _check_for_malformed_list_arg(s):
    if isinstance(s, str) and "," in s:
        raise Exception(
            "Rather than writing `{s}` to define a list you would `[{s}]` in the config file."
        )


def map_dims_and_variables(ds, dim_mapping):
    """
    Map the input dimensions to the architecture dimensions
    using the `dim_mapping` dictionary. Each key in the `dim_mapping`
    describes the name of the architecture dimension to map to and the values
    describe what to map from (the `input_dim_map`).

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
            ds = ds.stack({arch_dim: input_dim_map})

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
            da = _stack_variables_as_coord_values(
                ds=ds, name_format=name_format, combined_dim_name=arch_dim
            )
        elif len(dims) == 1:
            da = _stack_variables_by_coord_values(
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


def _check_dataset_attributes(ds, expected_attributes, dataset_name):
    # check that the dataset has the expected attributes with the expected values
    missing_attributes = set(expected_attributes.keys()) - set(ds.attrs.keys())
    if len(missing_attributes) > 0:
        raise ValueError(
            f"Dataset {dataset_name} is missing the following attributes: {missing_attributes}"
        )

    # check for attributes having the wrong value
    incorrect_attributes = {
        k: v for k, v in expected_attributes.items() if ds.attrs[k] != v
    }
    if len(incorrect_attributes) > 0:
        s_list = "\n".join(
            [f"{k}: {v} != {ds.attrs[k]}" for k, v in incorrect_attributes.items()]
        )
        raise ValueError(
            f"Dataset {dataset_name} has the following incorrect attributes: {s_list}"
        )


def main(fp_config):
    with open(fp_config, "r") as fh:
        config = yaml.load(fh, Loader=yaml.FullLoader)

    architecture_config = config["architecture"]

    dataarrays_by_target = defaultdict(list)

    for dataset_config in config["inputs"]:
        path = dataset_config["path"]
        variables = dataset_config["variables"]
        dataset_name = dataset_config["name"]
        target_arch_var = dataset_config["target"]
        expected_input_attributes = dataset_config.get("attributes", {})

        arch_dims = architecture_config["input_variables"][target_arch_var]

        try:
            ds = _load_and_subset_dataset(fp=path, variables=variables)
        except Exception as ex:
            raise Exception(f"Error loading dataset {dataset_name} from {path}") from ex
        _check_dataset_attributes(
            ds=ds,
            expected_attributes=expected_input_attributes,
            dataset_name=dataset_name,
        )

        dim_mapping = dataset_config["dim_mapping"]

        # check that there is an entry for each arch dimension
        # in the dim_mapping so that we know how to construct the
        # final dataset
        missing_dims = set(arch_dims) - set(dim_mapping.keys())
        if missing_dims:
            raise ValueError(
                f"Missing dimension mapping for {missing_dims}"
                f" for input dataset {dataset_name}, please provide"
                " a mapping for all architecture dimensions in"
                " using the 'dim_mapping' key in the input dataset"
            )

        da_target = map_dims_and_variables(ds=ds, dim_mapping=dim_mapping)
        dataarrays_by_target[target_arch_var].append(da_target)

    dataarrays = {}
    for target, das in dataarrays_by_target.items():
        concat_dim = None
        for da in das:
            d = da.attrs.get("variables_mapping_dim", None)
            if d is None:
                raise ValueError(
                    f"Dataarray for target {target} does not have the 'variables_mapping_dim' attribute"
                )
            if concat_dim is not None and d != concat_dim:
                raise ValueError(
                    f"Dataarrays for target {target} have different 'variables_mapping_dim' attributes: {d} != {concat_dim}"
                )
            concat_dim = d
        dataarrays[target] = xr.concat(das, dim=concat_dim)

    ds = xr.Dataset(dataarrays)
    print(ds)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to the config file")
    args = parser.parse_args()

    main(fp_config=args.config)


EXAMPLE_CONFIG = """
schema_version: v0.1.0
dataset_version: v0.1.0

arch:
  sampling_dim: time
  input_dims: [time, grid_index, feature]
  input_range:
    time:
      start: 2000-01-01T00:00
      end: 2001-01-01T00:00
      step: PT1H
  input_variables:
    static: [grid_index, feature]
    state: [time, grid_index, feature]
    forcing: [time, grid_index, feature]

inputs:
- name: danra_height_levels
  path: /data/danra/height_levels.zarr
  attributes:
    version: v0.3.0
  dims: [analysis_time, x, y, altitude]
  variables:
    u:
      altitude:
        values: [50, 100, 200, 300, 500, 700, 850, 1000]
        units: m
    v:
      altitude:
        values: [50, 100, 200, 300, 500, 700, 850, 1000]
        units: m
    t:
      altitude:
        values: [50, 100, 200, 300, 500, 700, 850, 1000]
        units: m
  dim_mapping:
    time: analysis_time
    feature:
      map_variables_to_var_name: True
      dims: [altitude]
      name: f"{var_name}_{altitude}"
    grid_index: x, y
  target: state

- name: danra_pressure_levels
  path: /data/danra/pressure_levels.zarr
  attributes:
    version: v0.3.0
  dims: [analysis_time, x, y, pressure]
  variables:
    u:
      pressure:
        values: [1000, 850, 700, 500, 300, 200, 100]
        units: hPa
  dim_mapping:
    time: analysis_time
    feature:
      map_variables_to_var_name: True
      dims: [pressure]
      name: f"{var_name}_{pressure}"
    grid_index: x, y
  target: state

- name: danra_single_levels
  path: /data/danra/single_levels.zarr
  attributes:
    version: v0.3.0
  dims: [analysis_time, x, y]
  variables: u10m, v10m, t2m
  dim_mapping:
    time: analysis_time
    feature:
      map_variables_to_var_name: True
      name: f"{var_name}"
    grid_index: x, y
  target: state

- name: danra_single_levels_forcings
  path: /data/danra/single_levels.zarr
  attributes:
    version: v0.3.0
  dims: [analysis_time, x, y]
  variables: nswlr
  dim_mapping:
    time: analysis_time
    feature:
      map_variables_to_var_name: True
      name: f"{var_name}"
    grid_index: x, y
  target: forcing

- name: danra_static2d
  path: /data/danra/static2d.zarr
  attributes:
    version: v0.3.0
  dims: [x, y]
  variables: [topography_height, land_area_fraction]
  dim_mapping:
    grid_index: x, y
  target: static

- name: meps_ensemble_forecasts
  path: /data/meps/ensemble_forecasts.zarr
  variables: [u, v, t]
  dims: [analysis_time, forecast_time, ensemble_member, x, y]
  dim_mapping:
    time: forecast_time
    grid_index: x, y
  sub_sampling:
    analysis_time:
      time: 0
    ensemble_member: "random"
  target: state


- name: dini_forecast
  path: /data/dini_forecasts_2000_2010.zarr
  variables: [u, v, t]
  dims: [analysis_time, forecast_time, x, y]
  dim_mapping:
    time: forecast_time
    grid_index: x, y
  sub_sampling:
    analysis_time:
      time: 0
  target: state
"""
