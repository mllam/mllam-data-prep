from typing import Dict

import xarray as xr

from ..config import Statistics


def calc_stats(
    ds: xr.Dataset, statistics_config: Statistics, splitting_dim: str
) -> Dict[str, xr.Dataset]:
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
    splitting_dim : str
        Dimension along which splits are made, this is used to calculate differences
        for operations prefixed with "diff_", for example "diff_mean" or "diff_std".
        Only the variables which actually span along the splitting_dim will be included
        in the output.

    Returns
    -------
    stats : Dict[str, xr.Dataset]
        Dictionary with the operation names as keys and the calculated statistics as values
    """
    stats = {}
    for op_split in statistics_config.ops:
        try:
            pre_op, op = op_split.split("_")
        except ValueError:
            op = op_split
            pre_op = None

        if pre_op is not None:
            if pre_op == "diff":
                # subset to select only the variable which have the splitting_dim
                vars_to_keep = [v for v in ds.data_vars if splitting_dim in ds[v].dims]
                ds = ds[vars_to_keep].diff(dim=splitting_dim)
            else:
                raise NotImplementedError(pre_op)

        fn = getattr(ds, op)
        stats[op_split] = fn(dim=statistics_config.dims)

    return stats
