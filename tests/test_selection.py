import isodate
import numpy as np
import pytest
import xarray as xr

import mllam_data_prep as mdp


@pytest.fixture
def ds():
    """
    Load the height_levels.zarr dataset
    """
    fp = "https://mllam-test-data.s3.eu-north-1.amazonaws.com/height_levels.zarr"
    ds = xr.open_zarr(fp)
    return ds


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


@pytest.mark.parametrize("x_start, x_end", ([-50000, -51000], [0, 500000]))
def test_error_on_empty_range(ds, x_start, x_end):
    """
    Test if an error is thrown if the chosen range is empty
    """
    y_start = -600000
    y_end = -590000
    coord_ranges = {
        "x": mdp.config.Range(start=x_start, end=x_end),
        "y": mdp.config.Range(start=y_start, end=y_end),
    }

    with pytest.raises(AssertionError):
        ds = mdp.ops.selection.select_by_kwargs(ds, **coord_ranges)


def test_slice_time(ds):
    start = "1990-09-01T00:00"
    end = "1990-09-09T00:00"
    coord_ranges = {
        "time": mdp.config.Range(start=start, end=end),
    }

    ds = mdp.ops.selection.select_by_kwargs(ds, **coord_ranges)


@pytest.mark.parametrize("step", ["PT6H", "PT3H"])
def test_timestep_matches_output(ds, step):
    start = "1990-09-01T00:00"
    end = "1990-09-09T00:00"
    coord_ranges = {
        "time": mdp.config.Range(start=start, end=end, step=step),
    }

    ds = mdp.ops.selection.select_by_kwargs(ds, **coord_ranges)

    td = isodate.parse_duration(step)
    timestep_chosen_in_slice = np.timedelta64(int(td.total_seconds()), "s")
    timestep_in_dataset = np.diff(ds.time)[0]

    assert timestep_chosen_in_slice == timestep_in_dataset


def test_raises_if_time_step_is_not_multiple_of_dataset_frequency(ds):
    step = "PT5H"
    start = "1990-09-01T03:00"
    end = "1990-09-09T00:00"
    coord_ranges = {
        "time": mdp.config.Range(start=start, end=end, step=step),
    }

    with pytest.raises(ValueError):
        ds = mdp.ops.selection.select_by_kwargs(ds, **coord_ranges)
