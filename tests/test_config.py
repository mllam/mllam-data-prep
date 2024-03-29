import pytest
import yaml

from mllam_data_prep.config import (
    ConfigDict,
    InvalidConfigVariableException,
    MissingConfigVariableException,
)

EXAMPLE_CONFIG_YAML = """
schema_version: v0.1.0
dataset_version: v0.1.0

architecture:
  sampling_dim: time
  input_variables:
    static: [grid_index, feature]
    state: [time, grid_index, state_feature]
    forcing: [time, grid_index, forcing_feature]
  input_range:
    time:
      start: 1990-09-03T00:00
      end: 1990-09-04T00:00
      step: PT3H

inputs:
  danra_height_levels:
    path: ~/Desktop/mldev/height_levels.zarr
    dims: [time, x, y, altitude]
    variables:
      u:
        altitude:
          sel: [100, ]
          units: m
      v:
        altitude:
          sel: [100, ]
          units: m
    dim_mapping:
      time: time
      state_feature:
        method: stack_variables_by_var_name
        dims: [altitude]
        name: f"{var_name}{altitude}m"
      grid_index:
        method: flatten
        dims: [x, y]
    target_architecture_variable: state

  danra_surface:
    path: ~/Desktop/mldev/single_levels.zarr
    dims: [time, x, y]
    variables:
      - pres_seasurface
    dim_mapping:
      time: time
      grid_index:
        method: flatten
        dims: [x, y]
      forcing_feature:
        method: stack_variables_by_var_name
        name: f"{var_name}"
    target_architecture_variable: forcing
"""


def test_get_config():
    config = ConfigDict(dict(schema_version="v0.1.0", foobar=42))

    assert config["schema_version"] is not None
    # raise `InvalidConfig` because `dataset_version` is in the spec, but isn't defined in the config
    with pytest.raises(MissingConfigVariableException):
        config["dataset_version"]

    # raise `InvalidConfig` because although `foobar` is defined in the config, it shouldn't be as it isn't part of the spec
    with pytest.raises(InvalidConfigVariableException):
        config["foobar"]


def test_get_config_nested():
    config = ConfigDict(yaml.load(EXAMPLE_CONFIG_YAML, Loader=yaml.FullLoader))

    for dataset_name, input_config in config["inputs"].items():
        assert input_config["path"] is not None
        assert input_config["variables"] is not None
        assert input_config["target_architecture_variable"] is not None
        with pytest.raises(KeyError):
            # `name` is given by the key, so isn't expected to be its own field
            input_config["name"]
