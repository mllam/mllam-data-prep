import pytest
import xarray as xr

import mllam_data_prep as mdp


@pytest.fixture
def ds():
    """
    Load the height_levels.zarr dataset
    """
    fp = "https://mllam-test-data.s3.eu-north-1.amazonaws.com/height_levels.zarr"
    return xr.open_zarr(fp)


def test_range_slice_within_range(ds):
    """
    test if the slice is within the specified range
    """
    x_start = -50000
    x_end = -40000
    y_start = -600000
    y_end = -590000
    coord_ranges = {
        "x": mdp.config.Range(start=x_start, end=x_end),
        "y": mdp.config.Range(start=y_start, end=y_end),
    }

    ds = mdp.ops.selection.select_by_kwargs(ds, **coord_ranges)
    assert ds.x.min() >= x_start
    assert ds.x.max() <= x_end
    assert ds.y.min() >= y_start
    assert ds.y.max() <= y_end

    ds


@pytest.mark.parametrize("x_start, x_end", ([-2, -1], [10, 11]))
def test_error_on_empty_range(x_start, x_end):
    """
    Test if an error is thrown if the chosen range is empty
    """
    ds = xr.Dataset(coords={"x": [0, 1, 2], "y": [0, 1, 2]})
    y_start = 0
    y_end = 2
    coord_ranges = {
        "x": mdp.config.Range(start=x_start, end=x_end),
        "y": mdp.config.Range(start=y_start, end=y_end),
    }

    with pytest.raises(ValueError, match="empty range"):
        ds = mdp.ops.selection.select_by_kwargs(ds, **coord_ranges)


def test_error_on_equal_range_bounds():
    ds = xr.Dataset(coords={"x": [0, 1, 2]})
    coord_ranges = {"x": mdp.config.Range(start=1, end=1)}

    with pytest.raises(ValueError, match="Start and end cannot be the same"):
        mdp.ops.selection.select_by_kwargs(ds, **coord_ranges)
