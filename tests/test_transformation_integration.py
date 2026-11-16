import tempfile
from pathlib import Path

import numpy as np
import xarray as xr
import yaml

import mllam_data_prep as mdp


def _write_dummy_zarr(fp, n_time=4, n_x=3, n_y=3):
    time = np.arange(n_time)
    x = np.arange(n_x)
    y = np.arange(n_y)
    rng = np.random.default_rng(seed=0)

    # precip: non-negative, right-skewed, suitable for log1p
    precip = rng.exponential(scale=2.0, size=(n_time, n_x, n_y))

    ds = xr.Dataset(
        {
            "precip": (
                ["time", "x", "y"],
                precip,
                {"units": "mm", "long_name": "precipitation"},
            ),
        },
        coords={"time": time, "x": x, "y": y},
    )
    ds.to_zarr(fp, mode="w")
    return ds


def test_log1p_transformation_end_to_end():
    with tempfile.TemporaryDirectory() as tmpdir:
        fp_input_zarr = Path(tmpdir) / "input.zarr"
        ds_input = _write_dummy_zarr(fp_input_zarr)

        config_dict = dict(
            schema_version="v0.7.0",
            dataset_version="v0.1.0",
            output=dict(
                variables=dict(
                    forcing=["time", "grid_index", "forcing_feature"],
                ),
            ),
            inputs=dict(
                dummy_input=dict(
                    path=str(fp_input_zarr),
                    dims=["time", "x", "y"],
                    variables=dict(
                        precip=dict(
                            transformation=dict(method="log1p"),
                        ),
                    ),
                    dim_mapping=dict(
                        time=dict(method="rename", dim="time"),
                        grid_index=dict(method="stack", dims=["x", "y"]),
                        forcing_feature=dict(
                            method="stack_variables_by_var_name",
                            name_format="{var_name}",
                        ),
                    ),
                    target_output_variable="forcing",
                ),
            ),
        )

        fp_config = Path(tmpdir) / "config.yaml"
        with open(fp_config, "w") as f:
            yaml.dump(config_dict, f)

        ds_out = mdp.create_dataset(mdp.Config.from_yaml_file(fp_config))

        # the output should have the forcing variable, with a forcing_feature
        # dimension of length 1 (just "precip")
        assert "forcing" in ds_out.data_vars
        assert "forcing_feature" in ds_out.coords
        assert list(ds_out.forcing_feature.values) == ["precip"]

        # the values should be log1p of the original precip values, flattened
        # over grid_index in the same (x, y) stacking order
        expected = np.log1p(ds_input["precip"].values)
        expected_flat = expected.reshape(expected.shape[0], -1)  # (time, grid_index)

        np.testing.assert_allclose(
            ds_out["forcing"].sel(forcing_feature="precip").values,
            expected_flat,
        )
