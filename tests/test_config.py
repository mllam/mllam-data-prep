import pytest
import yaml
from dataclass_wizard.errors import MissingFields, UnknownJSONKey

import mllam_data_prep as mdp

with open("example.danra.yaml", "r") as file:
    BASE_CONFIG = file.read()

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


def test_get_config_nested():
    config = mdp.Config.from_yaml(VALID_EXAMPLE_CONFIG_YAML)

    for dataset_name, input_config in config.inputs.items():
        assert input_config.path is not None
        assert input_config.variables is not None
        assert input_config.target_output_variable is not None
        with pytest.raises(AttributeError):
            input_config.foobarfield


def test_config_roundtrip():
    original_config = mdp.Config.from_yaml(VALID_EXAMPLE_CONFIG_YAML)
    roundtrip_config_dict = mdp.Config.from_dict(original_config.to_dict())
    roundtrip_config_yaml = mdp.Config.from_yaml(original_config.to_yaml())
    roundtrip_config_json = mdp.Config.from_json(original_config.to_json())
    assert original_config == roundtrip_config_dict
    assert original_config == roundtrip_config_yaml
    assert original_config == roundtrip_config_json


INVALID_MISSING_VARIABLES_YAML = """
inputs:
  danra_height_levels:
    path: https://object-store.os-api.cci1.ecmwf.int/mllam-testdata/danra_cropped/v0.2.0/single_levels.zarr
    dims: [time, x, y]
    dim_mapping:
      time:
        method: rename
        dim: time
      state_feature:
        method: stack_variables_by_var_name
        name_format: "{var_name}"
      grid_index:
        method: stack
        dims: [x, y]
    target_output_variable: state
"""

INVALID_OVERLAPPING_VARIABLE_NAMES_YAML = """
inputs:
  danra_height_levels:
    path: https://object-store.os-api.cci1.ecmwf.int/mllam-testdata/danra_cropped/v0.2.0/single_levels.zarr
    dims: [time, x, y]
    variables:
      - swavr0m
    derived_variables:
      swavr0m:
        kwargs:
          time: ds_input.time
          lat: ds_input.lat
          lon: ds_input.lon
        function: mllam_data_prep.ops.derive_variable.physical_field.calculate_toa_radiation
    dim_mapping:
      time:
        method: rename
        dim: time
      state_feature:
        method: stack_variables_by_var_name
        name_format: "{var_name}"
      grid_index:
        method: stack
        dims: [x, y]
    target_output_variable: state
"""

INVALID_INPUTS_COORDS_SELECTION_YAML = """
inputs:
  danra_height_levels:
    path: https://object-store.os-api.cci1.ecmwf.int/mllam-testdata/danra_cropped/v0.2.0/height_levels.zarr
    dims: [time, x, y, altitude]
    variables:
      u:
        altitude:
          values: [100, ]
          units: m
      v:
        altitude:
          values: [50, ]
          units: m
    dim_mapping:
      time:
        method: rename
        dim: time
      state_feature:
        method: stack_variables_by_var_name
        dims: [altitude]
        name_format: "{var_name}{altitude}m"
      grid_index:
        method: stack
        dims: [x, y]
    target_output_variable: state
"""


def update_config(config: str, update: str):
    """
    Update provided config

    Parameters
    ----------
    config: str
        String with config in yaml format
    update: dict
        Dictionary with the updated config

    Returns
    -------
    config: Config
        Updated config
    """
    original_config = mdp.Config.from_yaml(config)
    update = yaml.safe_load(update)
    modified_config = original_config.to_dict()
    modified_config.update(update)

    return modified_config


@pytest.mark.parametrize(
    "base_config, invalid_inputs_section, expected_exception, expected_message",
    [
        (
            # Invalid inputs section with missing `variables` and/or `derived_variables`
            BASE_CONFIG,
            INVALID_MISSING_VARIABLES_YAML,  # invalid config yaml example
            mdp.InvalidConfigException,  # expected exception
            "Missing `variables` and/or `derived_variables`",  # expected error message from
        ),
        (
            # Invalid inputs section with overlapping variable names between `variables`
            # and `derived_variables`
            BASE_CONFIG,
            INVALID_OVERLAPPING_VARIABLE_NAMES_YAML,
            mdp.InvalidConfigException,
            "Overlapping variable names in `variables` and `derived_variables`",
        ),
        (
            # Invalid inputs section selection of variables from different coords levels,
            # which is currently not implemented
            BASE_CONFIG,
            INVALID_INPUTS_COORDS_SELECTION_YAML,
            NotImplementedError,
            "Selection of variables from different coord levels is currently not supported",
        ),
    ],
)
def test_config_validation_of_inputs_section(
    base_config, invalid_inputs_section, expected_exception, expected_message
):
    """
    Test that `validate_config` raises the coerrect exceptions when an inputs dataset
    1. is missing `variables` and/or `derived_variables`
    2. has overlapping variable names between `variables` and `derived_variables`
    3. includes a selection of variables from different coord levels, since this is not yet implemented
    """
    invalid_config_dict = update_config(base_config, invalid_inputs_section)

    with pytest.raises(expected_exception, match=expected_message):
        mdp.Config.from_dict(invalid_config_dict)
