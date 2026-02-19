import itertools

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import mllam_data_prep as mdp
from mllam_data_prep.config import DimMapping, InputDataset, Output, Range


def _write_zarr(ds: xr.Dataset, path):
    ds.to_zarr(path, mode="w")


@pytest.mark.parametrize(
    "input_order",
    list(itertools.permutations(["state", "static", "forcing"])),
)
def test_output_coord_ranges_not_dropped_between_inputs(tmp_path, input_order):
    """
    Ensure output coord range slicing is applied per-input without being
    affected by the order of inputs. This guards against mutating the shared
    output_coord_ranges dict when an input lacks a dimension (e.g. `static`
    without `time`), which would otherwise remove slicing for later inputs.

    See https://github.com/mllam/mllam-data-prep/issues/81 for bug report.
    """
    time = pd.date_range("2000-01-01", "2000-01-05", freq="1D")
    x = np.arange(2)

    state_ds = xr.Dataset(
        {"s": (("time", "x"), np.zeros((len(time), len(x))))},
        coords={"time": time, "x": x},
    )
    forcing_ds = xr.Dataset(
        {"f": (("time", "x"), np.ones((len(time), len(x))))},
        coords={"time": time, "x": x},
    )
    static_ds = xr.Dataset(
        {"static_feature": (("x",), np.array([10.0, 20.0]))},
        coords={"x": x},
    )

    state_path = tmp_path / "state.zarr"
    forcing_path = tmp_path / "forcing.zarr"
    static_path = tmp_path / "static.zarr"

    _write_zarr(state_ds, state_path)
    _write_zarr(forcing_ds, forcing_path)
    _write_zarr(static_ds, static_path)

    inputs_by_name = {
        "state": InputDataset(
            path=str(state_path),
            dims=["time", "x"],
            variables=["s"],
            target_output_variable="state",
            dim_mapping={
                "time": DimMapping(method="rename", dim="time"),
                "grid_index": DimMapping(method="stack", dims=["x"]),
                "state_feature": DimMapping(
                    method="stack_variables_by_var_name",
                    name_format="{var_name}",
                ),
            },
        ),
        "static": InputDataset(
            path=str(static_path),
            dims=["x"],
            variables=["static_feature"],
            target_output_variable="static",
            dim_mapping={
                "grid_index": DimMapping(method="stack", dims=["x"]),
                "static_feature": DimMapping(
                    method="stack_variables_by_var_name",
                    name_format="{var_name}",
                ),
            },
        ),
        "forcing": InputDataset(
            path=str(forcing_path),
            dims=["time", "x"],
            variables=["f"],
            target_output_variable="forcing",
            dim_mapping={
                "time": DimMapping(method="rename", dim="time"),
                "grid_index": DimMapping(method="stack", dims=["x"]),
                "forcing_feature": DimMapping(
                    method="stack_variables_by_var_name",
                    name_format="{var_name}",
                ),
            },
        ),
    }

    ordered_inputs = {name: inputs_by_name[name] for name in input_order}

    config = mdp.Config(
        schema_version="v0.6.0",
        dataset_version="v0.0.0",
        output=Output(
            variables={
                "state": ["time", "grid_index", "state_feature"],
                "forcing": ["time", "grid_index", "forcing_feature"],
                "static": ["grid_index", "static_feature"],
            },
            coord_ranges={
                "time": Range(
                    start="2000-01-01T00:00",
                    end="2000-01-03T00:00",
                    step="PT24H",
                )
            },
        ),
        inputs=ordered_inputs,
    )

    ds = mdp.create_dataset(config=config)

    expected_len = 3
    assert ds["state"].sizes["time"] == expected_len
    assert ds["forcing"].sizes["time"] == expected_len
    assert "time" not in ds["static"].dims
