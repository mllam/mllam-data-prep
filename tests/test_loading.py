"""
Unit tests for ops.loading module.
"""
import pytest
import xarray as xr

from mllam_data_prep.ops.loading import load_input_dataset


@pytest.fixture
def sample_dataset():
    """Create a simple test dataset."""
    return xr.Dataset(
        {"var": (["x"], [1, 2, 3])},
        coords={"x": [0, 1, 2]},
    )


def test_load_input_dataset_zarr(sample_dataset, tmp_path):
    """Test load_input_dataset with zarr format."""
    zarr_path = tmp_path / "test.zarr"
    sample_dataset.to_zarr(zarr_path, mode="w")
    
    loaded = load_input_dataset(str(zarr_path))
    assert isinstance(loaded, xr.Dataset)
    assert "var" in loaded.data_vars
    assert list(loaded.x.values) == [0, 1, 2]


def test_load_input_dataset_netcdf(sample_dataset, tmp_path):
    """Test load_input_dataset with netCDF format."""
    # Skip if NetCDF engine is not available
    pytest.importorskip("netCDF4")
    
    nc_path = tmp_path / "test.nc"
    sample_dataset.to_netcdf(nc_path, engine="netcdf4")
    
    loaded = load_input_dataset(str(nc_path))
    assert isinstance(loaded, xr.Dataset)
    assert "var" in loaded.data_vars
    assert list(loaded.x.values) == [0, 1, 2]


def test_load_input_dataset_nonexistent():
    """Test load_input_dataset with non-existent file."""
    with pytest.raises((OSError, FileNotFoundError)):
        load_input_dataset("/nonexistent/path/to/file.zarr")
