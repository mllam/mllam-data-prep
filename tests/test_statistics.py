import numpy as np
import xarray as xr

from mllam_data_prep.config import Statistics
from mllam_data_prep.ops.statistics import calc_stats


def test_diff_statistics_start_from_original_dataset_for_each_operation():
    ds = xr.Dataset(
        {
            "state": xr.DataArray(
                np.array([[0.0], [1.0], [3.0], [6.0]]),
                dims=("time", "feature"),
                coords={"time": [0, 1, 2, 3], "feature": ["a"]},
            )
        }
    )
    statistics_config = Statistics(
        ops=["diff_mean", "diff_std"],
        dims=["time"],
    )

    stats = calc_stats(
        ds=ds,
        statistics_config=statistics_config,
        splitting_dim="time",
    )

    expected_diffs = ds["state"].diff(dim="time")
    xr.testing.assert_allclose(
        stats["diff_mean"]["state"],
        expected_diffs.mean(dim="time"),
    )
    xr.testing.assert_allclose(
        stats["diff_std"]["state"],
        expected_diffs.std(dim="time"),
    )
