"""
Unit tests for helper functions in ops.selection module.
"""
import pandas as pd
import pytest
import xarray as xr

from mllam_data_prep.ops.selection import check_point_in_dataset, check_step


@pytest.fixture
def simple_time_dataset():
    """Create a simple dataset with time coordinate."""
    time_values = pd.date_range("2020-01-01", periods=5, freq="3H")
    return xr.Dataset(
        {"var": (["time"], range(5))},
        coords={"time": time_values},
    )


def test_check_point_in_dataset_point_exists(simple_time_dataset):
    """Test check_point_in_dataset when point exists in coordinate."""
    point = simple_time_dataset.time.values[2]
    # Should not raise
    check_point_in_dataset("time", point, simple_time_dataset)


def test_check_point_in_dataset_point_not_exists(simple_time_dataset):
    """Test check_point_in_dataset when point does not exist in coordinate."""
    point = pd.Timestamp("2020-01-02T12:00")
    with pytest.raises(ValueError, match="Provided value for coordinate time"):
        check_point_in_dataset("time", point, simple_time_dataset)


def test_check_point_in_dataset_none_point(simple_time_dataset):
    """Test check_point_in_dataset when point is None (should not raise)."""
    # Should not raise when point is None
    check_point_in_dataset("time", None, simple_time_dataset)


def test_check_step_constant_step_matches(simple_time_dataset):
    """Test check_step when step is constant and matches requested step."""
    requested_step = pd.Timedelta(hours=3)
    # Should not raise
    check_step(requested_step, "time", simple_time_dataset)


def test_check_step_constant_step_mismatch(simple_time_dataset):
    """Test check_step when step is constant but doesn't match requested step."""
    requested_step = pd.Timedelta(hours=6)
    with pytest.raises(ValueError, match="Step size for coordinate time"):
        check_step(requested_step, "time", simple_time_dataset)


def test_check_step_non_constant_step():
    """Test check_step when step size is not constant."""
    # Create dataset with non-constant time steps
    time_values = pd.to_datetime(
        ["2020-01-01T00:00", "2020-01-01T03:00", "2020-01-01T10:00", "2020-01-01T13:00"]
    )
    ds = xr.Dataset(
        {"var": (["time"], range(4))},
        coords={"time": time_values},
    )
    requested_step = pd.Timedelta(hours=3)
    with pytest.raises(ValueError, match="Step size for coordinate time is not constant"):
        check_step(requested_step, "time", ds)


def test_check_step_single_point_coordinate():
    """Test check_step with single point coordinate (should raise descriptive ValueError)."""
    # Create dataset with single time point
    time_values = pd.date_range("2020-01-01", periods=1, freq="3H")
    ds = xr.Dataset(
        {"var": (["time"], [1])},
        coords={"time": time_values},
    )
    requested_step = pd.Timedelta(hours=3)
    with pytest.raises(ValueError, match="Cannot compute step size.*fewer than 2 points"):
        check_step(requested_step, "time", ds)
