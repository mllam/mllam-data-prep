import pytest

import mllam_data_prep as mdp


def test_range_with_datetime():
    mdp.Config.from_yaml_file("tests/resources/valid_config.yaml")
