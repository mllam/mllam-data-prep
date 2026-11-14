import numpy as np
import xarray as xr

from mllam_data_prep.config import Config
from mllam_data_prep.create_dataset import create_dataset

def test_variable_selection_by_independent_coords():
    """
    Test reproducing Issue #61: selecting variables by different coordinates.
    Ensure that we don't get NaN-filled cartesian product variables.
    """
    # Create mock dataset
    altitudes = [30, 50, 75, 100]
    time = [1, 2]
    x = [0, 1]
    y = [0, 1]

    shape = (len(time), len(x), len(y), len(altitudes))
    coords = {"time": time, "x": x, "y": y, "altitude": altitudes}

    ds_mock = xr.Dataset(
        data_vars={
            "u": (["time", "x", "y", "altitude"], np.ones(shape)),
            "v": (["time", "x", "y", "altitude"], np.ones(shape) * 2),
            "t": (["time", "x", "y", "altitude"], np.ones(shape) * 3),
        },
        coords=coords,
    )
    # Add expected units to coordinates for extraction check
    ds_mock.altitude.attrs["units"] = "m"

    # Save mock dataset to disk or pass it to config somehow
    # By default load_input_dataset reads from path.
    # To bypass, we can mock load_input_dataset, or just save it to a temp zarr.
    import tempfile
    import pathlib

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = pathlib.Path(tmpdir) / "height_levels.zarr"
        ds_mock.to_zarr(tmp_path)

        # Config exactly resembling issue 61
        config_dict = {
            "schema_version": "v0.6.0",
            "dataset_version": "v1.0",
            "inputs": {
                "danra_height_levels": {
                    "path": str(tmp_path),
                    "dims": ["time", "x", "y", "altitude"],
                    "variables": {
                        "u": {"altitude": {"values": [100, 50], "units": "m"}},
                        "v": {"altitude": {"values": [100, 75], "units": "m"}},
                        "t": {"altitude": {"values": [30], "units": "m"}},
                    },
                    "dim_mapping": {
                        "time": {"method": "rename", "dim": "time"},
                        "state_feature": {
                            "method": "stack_variables_by_var_name",
                            "dims": ["altitude"],
                            "name_format": "{var_name}{altitude}m",
                        },
                        "grid_index": {"method": "stack", "dims": ["x", "y"]},
                    },
                    "target_output_variable": "state",
                }
            },
            "output": {
                "variables": {
                    "state": ["time", "grid_index", "state_feature"]
                }
            },
        }

        import yaml
        config_path = tmp_path.parent / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_dict, f)

        config = Config.from_yaml_file(config_path)

        # Execute
        ds_out = create_dataset(config)

        # Check results
        expected_vars = {"u100m", "u50m", "v100m", "v75m", "t30m"}
        
        # ds_out has data variables mapped into `state`. But wait!
        # The variables are mapped into `state_feature` coordinate in the `state` data_var, NOT as `data_vars`!
        # Let's verify the mllam-data-prep behavior.
        # "target_output_variable": "state" means it creates a dataset with `ds_out["state"]`
        # and coordinate `state_feature` containing `['u100m', 'u50m', 'v100m', 'v75m', 't30m']`.

        assert "state" in ds_out.data_vars
        state_features = ds_out.coords["state_feature"].values.tolist()

        assert set(state_features) == expected_vars, f"Expected {expected_vars}, got {state_features}"

        for feature in expected_vars:
            da_feature = ds_out["state"].sel(state_feature=feature)
            # Assert it's not entirely NaNs
            assert not da_feature.isnull().all(), f"Feature {feature} is entirely NaNs!"
