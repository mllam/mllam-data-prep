import datetime

import pytest
from dataclass_wizard.errors import MissingFields, UnknownJSONKey

import mllam_data_prep as mdp
from mllam_data_prep import config

INVALID_EXTRA_FIELDS_CONFIG_YAML = """
schema_version: v0.1.0
dataset_version: v0.1.0

output:
  variables:
    static: [grid_index, feature]
  coord_ranges:
    time:
      start: 1990-09-03T00:00
      end: 1990-09-04T00:00
      step: PT3H

inputs: {}
foobar: 42
"""

MISSING_FIELDS_CONFIG_YAML = """
schema_version: v0.1.0
dataset_version: v0.1.0
"""


def test_get_config_issues():
    """Test that the Config class raises the correct exceptions when the YAML file is invalid."""
    with pytest.raises(UnknownJSONKey):
        mdp.Config.from_yaml(INVALID_EXTRA_FIELDS_CONFIG_YAML)

    with pytest.raises(MissingFields):
        mdp.Config.from_yaml(MISSING_FIELDS_CONFIG_YAML)


VALID_EXAMPLE_CONFIG_YAML = """
schema_version: v0.1.0
dataset_version: v0.1.0

output:
  variables:
    static: [grid_index, feature]
    state: [time, grid_index, state_feature]
    forcing: [time, grid_index, forcing_feature]
  coord_ranges:
    time:
      start: 1990-09-03T00:00
      end: 1990-09-04T00:00
      step: PT3H
  splitting:
    dim: time
    splits:
      train:
        start: 1990-09-03T00:00
        end: 1990-09-06T00:00
        compute_statistics:
          ops: [mean, std]
          dims: [grid_index, time]
      validation:
        start: 1990-09-06T00:00
        end: 1990-09-07T00:00
      test:
        start: 1990-09-07T00:00
        end: 1990-09-09T00:00

inputs:
  danra_height_levels:
    path: ~/Desktop/mldev/height_levels.zarr
    dims: [time, x, y, altitude]
    variables:
      u:
        altitude:
          values: [100, ]
          units: m
      v:
        altitude:
          values: [100, ]
          units: m
    dim_mapping:
      time:
        method: rename
        dim: time
      state_feature:
        method: stack_variables_by_var_name
        dims: [altitude]
        name_format: f"{var_name}{altitude}m"
      grid_index:
        method: flatten
        dims: [x, y]
    target_output_variable: state

  danra_surface:
    path: ~/Desktop/mldev/single_levels.zarr
    dims: [time, x, y]
    variables:
      - pres_seasurface
    dim_mapping:
      time:
        method: rename
        dim: time
      grid_index:
        method: flatten
        dims: [x, y]
      forcing_feature:
        method: stack_variables_by_var_name
        name_format: f"{var_name}"
    target_output_variable: forcing
"""


def test_can_load_config_with_datetime_object_in_time_range():
    fp = "tests/resources/sliced_example.danra.yaml"
    mdp.Config.from_yaml_file(fp)


def test_can_load_config_with_datetime_string_in_time_range():
    fp = "tests/resources/sliced_example_with_datetime_strings.danra.yaml"
    mdp.Config.from_yaml_file(fp)


def test_get_config_nested():
    config = mdp.Config.from_yaml(VALID_EXAMPLE_CONFIG_YAML)

    for dataset_name, input_config in config.inputs.items():
        assert input_config.path is not None
        assert input_config.variables is not None
        assert input_config.target_output_variable is not None
        with pytest.raises(AttributeError):
            input_config.foobarfield


def test_that_range_accepts_datetime():
    start = datetime.datetime(1990, 9, 3, 0, 0)
    end = datetime.datetime(1990, 9, 4, 0, 0)
    step = "PT3H"

    range_ = config.Range(start=start, end=end, step=step)
