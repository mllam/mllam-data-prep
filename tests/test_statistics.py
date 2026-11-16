import numpy as np
import pytest
import xarray as xr

from mllam_data_prep.config import Statistics
from mllam_data_prep.ops.statistics import calc_stats


@pytest.fixture
def sample_dataset():
    rng = np.random.default_rng(seed=42)
    time = np.arange(10)
    grid_index = np.arange(5)
    data = rng.normal(loc=10.0, scale=2.0, size=(len(time), len(grid_index)))
    ds = xr.Dataset(
        {"temperature": (["time", "grid_index"], data)},
        coords={"time": time, "grid_index": grid_index},
    )
    return ds


def test_calc_stats_mean_std(sample_dataset):
    stats_config = Statistics(ops=["mean", "std"], dims=["time", "grid_index"])
    stats = calc_stats(sample_dataset, stats_config, splitting_dim="time")
    assert set(stats.keys()) == {"mean", "std"}
    assert "temperature" in stats["mean"]
    assert "temperature" in stats["std"]


def test_calc_stats_order_independence(sample_dataset):
    """Regression test for the mutation bug: the order of ops in the config
    must not change the result of non-diff ops."""
    stats_config_a = Statistics(ops=["diff_mean", "mean"], dims=["time", "grid_index"])
    stats_config_b = Statistics(ops=["mean", "diff_mean"], dims=["time", "grid_index"])
    stats_a = calc_stats(sample_dataset, stats_config_a, splitting_dim="time")
    stats_b = calc_stats(sample_dataset, stats_config_b, splitting_dim="time")

    xr.testing.assert_allclose(stats_a["mean"], stats_b["mean"])
    xr.testing.assert_allclose(stats_a["diff_mean"], stats_b["diff_mean"])

    # and the plain mean should match computing it directly on the undiffed data
    expected_mean = sample_dataset.mean(dim=["time", "grid_index"])
    xr.testing.assert_allclose(stats_a["mean"], expected_mean)


def test_calc_stats_diff_ops(sample_dataset):
    stats_config = Statistics(
        ops=["diff_mean", "diff_std"], dims=["time", "grid_index"]
    )
    stats = calc_stats(sample_dataset, stats_config, splitting_dim="time")
    expected = sample_dataset.diff(dim="time").mean(dim=["time", "grid_index"])
    xr.testing.assert_allclose(stats["diff_mean"], expected)


def test_calc_stats_median(sample_dataset):
    stats_config = Statistics(ops=["median"], dims=["time", "grid_index"])
    stats = calc_stats(sample_dataset, stats_config, splitting_dim="time")
    expected = sample_dataset.median(dim=["time", "grid_index"])
    xr.testing.assert_allclose(stats["median"], expected)


def test_calc_stats_min_max(sample_dataset):
    stats_config = Statistics(ops=["min", "max"], dims=["time", "grid_index"])
    stats = calc_stats(sample_dataset, stats_config, splitting_dim="time")
    assert "temperature" in stats["min"]
    assert "temperature" in stats["max"]


def test_calc_stats_quantile(sample_dataset):
    stats_config = Statistics(
        ops=["quantile_0.25", "quantile_0.75"], dims=["time", "grid_index"]
    )
    stats = calc_stats(sample_dataset, stats_config, splitting_dim="time")
    expected_q25 = sample_dataset.quantile(0.25, dim=["time", "grid_index"]).drop_vars(
        "quantile"
    )
    xr.testing.assert_allclose(stats["quantile_0.25"], expected_q25)


def test_calc_stats_iqr(sample_dataset):
    stats_config = Statistics(ops=["iqr"], dims=["time", "grid_index"])
    stats = calc_stats(sample_dataset, stats_config, splitting_dim="time")
    q75 = sample_dataset.quantile(0.75, dim=["time", "grid_index"]).drop_vars(
        "quantile"
    )
    q25 = sample_dataset.quantile(0.25, dim=["time", "grid_index"]).drop_vars(
        "quantile"
    )
    xr.testing.assert_allclose(stats["iqr"], q75 - q25)


def test_calc_stats_unsupported_op_raises(sample_dataset):
    stats_config = Statistics(ops=["not_a_real_op"], dims=["time", "grid_index"])
    with pytest.raises(NotImplementedError):
        calc_stats(sample_dataset, stats_config, splitting_dim="time")
