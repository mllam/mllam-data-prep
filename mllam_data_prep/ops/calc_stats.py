from typing import Dict

import xarray as xr

from ..config import Statistics


def calc_stats(ds: xr.Dataset, statistics_config: Statistics) -> Dict[str, xr.Dataset]:
    """
    Calculate statistics for a given DataArray by applying the operations
    specified in the Statistics object and reducing over the dimensions
    specified in the Statistics object.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to calculate statistics for
    statistics_config : Statistics
        Configuration object specifying the operations and dimensions to reduce over

    Returns
    -------
    stats : Dict[str, xr.Dataset]
        Dictionary with the operation names as keys and the calculated statistics as values
    """
    stats = {}
    for op in statistics_config.ops:
        fn = getattr(ds, op)
        stats[op] = fn(dim=statistics_config.dims)

    return stats
