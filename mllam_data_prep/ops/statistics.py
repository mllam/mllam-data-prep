"""Export functions to calculate statistics for a given Dataset."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict

import xarray as xr

from ..config import Statistic


def calc_stats(
    ds: xr.Dataset, statistic_configs: Dict[str, Statistic], splitting_dim: str
) -> Dict[str, xr.Dataset]:
    """
    Calculate statistics for a given DataArray by applying the operations
    specified in the Statistics object and reducing over the dimensions
    specified in the Statistics object.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to calculate statistics for
    statistic_configs : Statistics
        Configuration object specifying the operations and dimensions to reduce over
    splitting_dim : str
        Dimension along which splits are made, this is used to calculate difference
        operations, for example "DiffMeanOperator" or "DiffStdOperator".
        Only the variables which actually span along the splitting_dim will be included
        in the output.

    Returns
    -------
    stats : Dict[str, xr.Dataset]
        Dictionary with the operation names as keys and the calculated statistics as values
    """
    stats = {}
    for stat_name, statistic in statistic_configs.items():
        if stat_name in globals():
            stat: StatisticOperator = globals()[stat_name](ds, splitting_dim)
            stats[stat.name] = stat.calc_stats(statistic.dims)
        else:
            raise NotImplementedError(stat_name)

    return stats


@dataclass
class StatisticOperator(ABC):
    """Base class to calculate statistics for a given Dataset.

    Attributes:
    -----------
    ds : xr.Dataset
        Dataset to calculate statistics for
    splitting_dim : str
        Dimension along which splits are made, this is used to calculate difference
        operations, for example "DiffMeanOperator" or "DiffStdOperator".
        Only the variables which actually span along the splitting_dim will be included
        in the output.
    """

    ds: xr.Dataset
    splitting_dim: str

    @property
    @abstractmethod
    def name(self):
        """Override property to specify the name of the statistic"""

    @abstractmethod
    def calc_stats(self, dims):
        """Override this method to implement the actual calculation"""


class MeanOperator(StatisticOperator):
    """Calculate the mean along the specified dimensions."""

    name = "mean"

    def calc_stats(self, dims):
        return self.ds.mean(dim=dims)


class StdOperator(StatisticOperator):
    """Calculate the standard deviation along the specified dimensions."""

    name = "std"

    def calc_stats(self, dims):
        return self.ds.std(dim=dims)


class DiffMeanOperator(StatisticOperator):
    """Calculate the mean of the differences along the specified dimensions."""

    name = "diff_mean"

    def calc_stats(self, dims):
        vars_to_keep = [
            v for v in self.ds.data_vars if self.splitting_dim in self.ds[v].dims
        ]
        ds_diff = self.ds[vars_to_keep].diff(dim=self.splitting_dim)
        return ds_diff.mean(dim=dims)


class DiffStdOperator(StatisticOperator):
    """Calculate std of the differences along the specified dimensions."""

    name = "diff_std"

    def calc_stats(self, dims):
        vars_to_keep = [
            v for v in self.ds.data_vars if self.splitting_dim in self.ds[v].dims
        ]
        ds_diff = self.ds[vars_to_keep].diff(dim=self.splitting_dim)
        return ds_diff.std(dim=dims)


class DiffTimeMeanOperator(DiffMeanOperator):
    """Calculate the mean of the differences along the time dimension.

    This is a duplicate of the DiffMeanOperator to allow for averaging over
    other dimensions.
    """

    name = "diff_time_mean"


class DiurnalDiffMeanOperator(StatisticOperator):
    """Calculate the mean of the diurnal differences along the specified dimensions."""

    name = "diurnal_mean"

    def calc_stats(self, dims):
        vars_to_keep = [
            v for v in self.ds.data_vars if self.splitting_dim in self.ds[v].dims
        ]
        ds_diff = self.ds[vars_to_keep].diff(dim=self.splitting_dim)

        # Group by hour and calculate mean
        grouped = ds_diff.groupby("time.hour")
        return grouped.mean(dim=dims)


class DiurnalDiffStdOperator(StatisticOperator):
    """Calculate the std of the diurnal differences along the specified dimensions."""

    name = "diurnal_std"

    def calc_stats(self, dims):
        vars_to_keep = [
            v for v in self.ds.data_vars if self.splitting_dim in self.ds[v].dims
        ]
        ds_diff = self.ds[vars_to_keep].diff(dim=self.splitting_dim)

        # Group by hour and calculate mean
        grouped = ds_diff.groupby("time.hour")
        return grouped.std(dim=dims)
