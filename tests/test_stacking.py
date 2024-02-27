import xarray as xr
import numpy as np
import mllam_data_prep.main as mdp


def test_stack_levels():
    """
    Test the stacking of levels in the dataset
    
    i.e. from variables [var1, var2] with levels [1, 2, 3]
    to a single variable with levels [var1_l1, var1_l2, var1_l3, var2_l1, var2_l2, var2_l3]
    """
    name_format = "{var_name}_l{level}"
    nx, ny, nz = 10, 6, 3
    ds = xr.Dataset(
        {
            "var1": xr.DataArray(np.random.random((nx, ny, nz)), dims=("x", "y", "level",)),
            "var2": xr.DataArray(np.random.random((nx, ny, nz)), dims=("x", "y", "level",)),
        },
        coords={"level": np.arange(nz)},
    )
    
    da_stacked = mdp._stack_variables_by_coord_values(ds=ds, level_dim="level", name_format=name_format)
    expected_coords = [name_format.format(var_name=v, level=l) for v in ["var1", "var2"] for l in range(nz)]
    
    assert da_stacked.dims == ("x", "y", "level")
    assert da_stacked.coords["level"].values.tolist() == expected_coords
    for v in expected_coords:
        assert da_stacked.sel(level=v).shape == (nx, ny)
        
    # check that the values are the same
    for v in ["var1", "var2"]:
        for l in [1, 2]:
            expected_values = ds[v].sel(level=l).values
            actual_values = da_stacked.sel(level=name_format.format(var_name=v, level=l)).values
            assert np.all(expected_values == actual_values)