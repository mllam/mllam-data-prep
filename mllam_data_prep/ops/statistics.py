"""Export functions to calculate statistics for a given Dataset."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List

import xarray as xr

from ..config import Statistic


def calc_stats(
    ds: xr.Dataset, statistic_configs: Dict[str, List[Statistic]], splitting_dim: str
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
    for stat_name, statistics in statistic_configs.items():
        if stat_name in globals():
            # Apply the operation to the dataset (multiple different configurations
            # of the same operator can be applied)
            for statistic in statistics:
                operator: StatisticOperator = globals()[stat_name](
                    ds=ds, splitting_dim=splitting_dim, var_name=statistic.var_name
                )
                stats[statistic.var_name] = operator.calc_stats(statistic.dims)
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
    var_name: str

    @abstractmethod
    def calc_stats(self, dims):
        """Override this method to implement the actual calculation"""


class MeanOperator(StatisticOperator):
    """Calculate the mean along the specified dimensions."""

    def calc_stats(self, dims):
        return self.ds.mean(dim=dims)


class StdOperator(StatisticOperator):
    """Calculate the standard deviation along the specified dimensions."""

    def calc_stats(self, dims):
        return self.ds.std(dim=dims)


class DiffMeanOperator(StatisticOperator):
    """Calculate the mean of the differences along the specified dimensions."""

    def calc_stats(self, dims):
        vars_to_keep = [
            v for v in self.ds.data_vars if self.splitting_dim in self.ds[v].dims
        ]
        ds_diff = self.ds[vars_to_keep].diff(dim=self.splitting_dim)
        return ds_diff.mean(dim=dims)


class DiffStdOperator(StatisticOperator):
    """Calculate std of the differences along the specified dimensions."""

    def calc_stats(self, dims):
        vars_to_keep = [
            v for v in self.ds.data_vars if self.splitting_dim in self.ds[v].dims
        ]
        ds_diff = self.ds[vars_to_keep].diff(dim=self.splitting_dim)
        return ds_diff.std(dim=dims)


class DiurnalDiffMeanOperator(StatisticOperator):
    """Calculate the mean of the diurnal differences along the specified dimensions."""

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

    def calc_stats(self, dims):
        vars_to_keep = [
            v for v in self.ds.data_vars if self.splitting_dim in self.ds[v].dims
        ]
        ds_diff = self.ds[vars_to_keep].diff(dim=self.splitting_dim)

        # Group by hour and calculate mean
        grouped = ds_diff.groupby("time.hour")
        return grouped.std(dim=dims)
