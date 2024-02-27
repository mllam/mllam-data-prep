import xarray as xr
import yaml

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
    for var, coords_to_sample in variables.items():
        da = ds[var]
        for coord, sampling in coords_to_sample.items():
            coord_values = sampling["sel"]
            da = da[var].sel(**{coord: coord_values})
            expected_units = sampling.get("units", None)
            coord_units = da[coord].attrs.get("units", None)
            if coord_units is not None and coord_units != expected_units:
                raise ValueError(
                    f"Expected units {expected_units} for coordinate {coord}"
                    f" in variable {var} but got {coord_units}"
                )
        ds_subset[var] = da
    return ds_subset


def _stack_variables_by_coord_values(ds, level_dim, name_format):
    """
    combine all levels of all variables into a single dataset

    for a set of variables in a file, e.g. [u, v, t], and a set of levels [50, 100]
    the output will combine all variables into a single xr.DataArray with
    coordinate values given by the name_format e.g. [u_l50, u_l100, v_l50,
    v_l100, t_l50, t_l100] if the format was "{var_name}_l{level}"

    Parameters
    ----------
    ds : xr.Dataset
        dataset with variables as data_vars and `level_dim` as a coordinate
    level_dim : str
        name of the coordinate that represents the levels
    name_format : str
        format string to construct the new coordinate values for the
        stacked levels

    Returns
    -------
    da_combined : xr.DataArray
        The combined dataset with the stacked levels
    """

    datasets = []
    for var in list(ds.data_vars):
        da = ds[var]
        coord_values = da.coords[level_dim].values
        new_coord_values = [
            name_format.format(var_name=var, **{level_dim: val}) for val in coord_values
        ]
        da = da.assign_coords({level_dim: new_coord_values})
        datasets.append(da)

    da_combined = xr.concat(datasets, dim=level_dim)

    return da_combined


def map_dims_and_variables(ds, input_dim_map, arch_dim):
    """
    Map the input dimensions to the architecture dimensions
    using the `input_dim_map`. The method of mapping is determined
    by the type of the `input_dim_map` variable.

    The mapping method can be one of the following:
    - A string: The name of the dimension in the input dataset to map to
    - A list: The list of dimensions in the input dataset to stack to
      create the architecture dimension
    - A dictionary: The dictionary with the mapping of the input dimensions
      to the architecture dimensions. The dictionary should have the following
      keys:
      - 'dims': The list of dimensions in the input dataset to map to the
        architecture dimension
      - 'name': The string format to construct the new coordinate values
        for the architecture dimension
      - 'map_variables_to_var_name': Optional, if True, the variables are
        mapped to the new coordinate values

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
    ds : xr.Dataset
        The dataset with the dimensions and variables mapped
    """
    if isinstance(input_dim_map, str):
        # if the input_dims is a string, we assume that it is the
        # name of the dimension in the input dataset and we rename
        # it to the architecture dimension
        ds = ds.rename({input_dim_map: arch_dim})
    elif isinstance(input_dim_map, list):
        # if the input_dims is a list, we assume that it is a list
        # of dimensions in the input dataset that we want to stack
        # to create the architecture dimension
        ds = ds.stack({arch_dim: input_dim_map})
    elif isinstance(input_dim_map, dict):
        # if the mapping is a dictionary, then the dimensions to map from
        # are expected to be given explicitly in the 'dims' key and the
        # new string format to construct the new coordinate values is
        # given in the 'name' key
        # optionally we can also set the 'map_variables_to_var_name' key
        # to True to map the variables to the new coordinate values
        dims = input_dim_map.get("dims", None)
        if dims is None:
            if len(dims) > 1:
                ds = ds.stack({arch_dim: dims})

        name_format = input_dim_map["name"]
        dim = dims[0]
        for value in ds[dim].values:
            new_coord = name_format.format(var_name=dim, **{dim: value})
            da_coord_value = ds[dim].sel({dim: value})
            assert da_coord_value is not None
            ds = ds.assign_coords({arch_dim: new_coord})
            if input_dim_map.get("map_variables_to_var_name", False):
                ds = ds.rename({dim: new_coord})


def main(fp_config):
    config = yaml.load(fp_config)

    arch_dims = config["arch_dims"]

    datasets = []

    for dataset_config in config["inputs"]:
        path = dataset_config["path"]
        variables = dataset_config["variables"]
        dataset_name = dataset_config["name"]
        ds = _load_and_subset_dataset(path=path, variables=variables)
        assert ds is not None

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

        # Do the dimension and variable mapping
        for arch_dim, input_dim_map in dim_mapping.items():
            datasets.append(None)
            assert arch_dim in arch_dims
            assert input_dim_map
