"""
Test that calc_stats does not mutate the input dataset across loop iterations.

Regression test for https://github.com/mllam/mllam-data-prep/issues/102
"""

import numpy as np
import xarray as xr

from mllam_data_prep.config import Statistics
from mllam_data_prep.ops.statistics import calc_stats


def test_calc_stats_no_mutation_with_multiple_diff_ops():
    """
    When ops=[mean, std, diff_mean, diff_std], diff_std should compute
    std(diff(ds)) not std(diff(diff(ds))). The original dataset must not
    be mutated between iterations.
    """
    data = np.array([1.0, 3.0, 6.0, 10.0, 15.0])
    ds = xr.Dataset({"temp": xr.DataArray(data, dims=["time"])})

    stats_config = Statistics(
        ops=["mean", "std", "diff_mean", "diff_std"], dims=["time"]
    )
    result = calc_stats(ds, stats_config, splitting_dim="time")

    # expected from the original dataset
    expected_mean = float(ds["temp"].mean())
    expected_std = float(ds["temp"].std())

    # expected from a single diff of the original dataset
    ds_diff = ds.diff(dim="time")  # [2, 3, 4, 5]
    expected_diff_mean = float(ds_diff["temp"].mean())
    expected_diff_std = float(ds_diff["temp"].std())

    assert np.isclose(float(result["mean"]["temp"]), expected_mean)
    assert np.isclose(float(result["std"]["temp"]), expected_std)
    assert np.isclose(float(result["diff_mean"]["temp"]), expected_diff_mean)
    assert np.isclose(float(result["diff_std"]["temp"]), expected_diff_std)


def test_calc_stats_preserves_original_dataset():
    """
    The original dataset passed to calc_stats must remain unchanged after
    the function returns.
    """
    data = np.array([1.0, 3.0, 6.0, 10.0, 15.0])
    ds = xr.Dataset({"temp": xr.DataArray(data, dims=["time"])})
    ds_copy = ds.copy(deep=True)

    stats_config = Statistics(
        ops=["mean", "std", "diff_mean", "diff_std"], dims=["time"]
    )
    calc_stats(ds, stats_config, splitting_dim="time")

    xr.testing.assert_identical(ds, ds_copy)
