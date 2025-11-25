import cf_xarray as cfxr
import numpy as np
import xarray as xr

from mllam_data_prep.config import Config, DimMapping
from mllam_data_prep.create_dataset import create_dataset
from mllam_data_prep.ops import mapping as mdp_mapping
from mllam_data_prep.ops import stacking as mdp_stacking


def test_stack_variables_along_coord():
    """
    Test the stacking of variables along a coordinate

    i.e. from variables [var1, var2] with levels [1, 2, 3]
    to a single variable with levels [var1_l1, var1_l2, var1_l3, var2_l1, var2_l2, var2_l3]
    """
    name_format = "{var_name}_l{level}"
    nx, ny, nz = 10, 6, 3
    dims = ["x", "y", "level"]
    ds = xr.Dataset(
        {
            "var1": xr.DataArray(np.random.random((nx, ny, nz)), dims=dims),
            "var2": xr.DataArray(np.random.random((nx, ny, nz)), dims=dims),
        },
        coords={"level": np.arange(nz)},
    )

    combined_dim_name = "feature"
    da_stacked = mdp_stacking.stack_variables_by_coord_values(
        ds=ds,
        coord="level",
        name_format=name_format,
        combined_dim_name=combined_dim_name,
    )
    expected_coord_values = [
        name_format.format(var_name=v, level=level)
        for v in ["var1", "var2"]
        for level in range(nz)
    ]

    assert da_stacked.dims == ("x", "y", "feature")
    assert da_stacked.coords[combined_dim_name].values.tolist() == expected_coord_values
    for v in expected_coord_values:
        assert da_stacked.sel({combined_dim_name: v}).shape == (nx, ny)

    # check that the values are the same
    for v in ["var1", "var2"]:
        for level in [1, 2]:
            expected_values = ds[v].sel(level=level).values
            actual_values = da_stacked.sel(
                {combined_dim_name: name_format.format(var_name=v, level=level)}
            ).values
            assert np.all(expected_values == actual_values)


def test_stack_xy_coords():
    """
    Test stacking two (or more) coordinates to create a single coordinate, for
    example (x, y) grid coordinates to a single grid_index coordinate
    """
    nx, ny, nz = 10, 6, 3
    dims = ["x", "y", "level"]

    ds = xr.Dataset(
        {
            "var1": xr.DataArray(np.random.random((nx, ny, nz)), dims=dims),
            "var2": xr.DataArray(np.random.random((nx, ny, nz)), dims=dims),
        },
        coords={"level": np.arange(nz)},
    )
    dim_mapping = dict(
        grid_index=DimMapping(
            method="stack",
            dims=["x", "y"],
        ),
        feature=DimMapping(
            method="stack_variables_by_var_name",
            dims=["level"],
            name_format="{level}_{var_name}",
        ),
    )

    da_stacked = mdp_mapping.map_dims_and_variables(
        ds=ds, dim_mapping=dim_mapping, expected_input_var_dims=dims
    )

    assert set(da_stacked.dims) == set(("grid_index", "feature"))
    assert da_stacked.coords["grid_index"].shape == (nx * ny,)


STACKING_EXAMPLE_YAML = """
schema_version: v0.6.0
dataset_version: v0.1.0

output:
  variables:
    static: [grid_index, static_feature]
  coord_ranges:
    time:
      start: 1990-09-03T00:00
      end: 1990-09-06T00:00
  chunking:
    time: 1

inputs:
  danra_height_levels:
    path: https://mllam-test-data.s3.eu-north-1.amazonaws.com/single_levels.zarr
    dims: [time, x, y]
    variables: [t2m]
    dim_mapping:
      static_feature:
        method: stack_variables_by_var_name
        name_format: "{var_name}"
      grid_index:
        method: stack
        dims: [x, y]

    target_output_variable: static
"""


def test_unstack_on_processed_dataset():
    config = Config.from_yaml(STACKING_EXAMPLE_YAML)

    ds_stacked = create_dataset(config)

    # check that we can get back the original x, y values from the stacked
    # coord. This is possible because we use the CF-compliant "gather
    # compression" for encoding the stacked coordinate MultiIndex
    # (https://cf-xarray.readthedocs.io/en/latest/coding.html)
    ds_multi_index = cfxr.decode_compress_to_multi_index(
        ds_stacked, idxnames="grid_index"
    )
    # check we can get out the names of the stacked coords
    stacked_coords = ds_multi_index.grid_index.to_index().names
    assert set(stacked_coords) == set(["x", "y"])

    ds_unstacked = ds_multi_index.unstack("grid_index")

    # compare unstacked coords to original dataset coords

    ds_orig = xr.open_zarr(config.inputs["danra_height_levels"].path, chunks={})

    for coord in ["x", "y"]:
        xr.testing.assert_equal(ds_unstacked[coord], ds_orig[coord])
