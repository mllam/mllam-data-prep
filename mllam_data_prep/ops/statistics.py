import re
from typing import Dict

import xarray as xr

from ..config import Statistics

_QUANTILE_PATTERN = re.compile(r"^quantile_(?P<q>\d*\.?\d+)$")


def _apply_op(ds: xr.Dataset, op: str, dims) -> xr.Dataset:
    """Apply a single (non diff-prefixed) statistic to `ds`, reducing over `dims`."""
    if op == "iqr":
        q75 = ds.quantile(0.75, dim=dims)
        q25 = ds.quantile(0.25, dim=dims)
        result = q75 - q25
        if "quantile" in result.coords:
            result = result.drop_vars("quantile")
        return result

    match = _QUANTILE_PATTERN.match(op)
    if match is not None:
        q = float(match.group("q"))
        result = ds.quantile(q, dim=dims)
        if "quantile" in result.coords:
            result = result.drop_vars("quantile")
        return result

    if not hasattr(ds, op):
        raise NotImplementedError(
            f"Statistic operation '{op}' is not supported. It must be a no-argument "
            "method of xarray.Dataset (e.g. 'mean', 'std', 'median', 'min', 'max'), "
            "'iqr', or a quantile given as 'quantile_<q>' (e.g. 'quantile_0.25')."
        )
    return getattr(ds, op)(dim=dims)


def calc_stats(
    ds: xr.Dataset, statistics_config: Statistics, splitting_dim: str
) -> Dict[str, xr.Dataset]:
    """
    Calculate statistics for a given Dataset by applying the operations
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

    # Compute the differenced dataset once, rather than mutating `ds` in the loop
    # (the old code did `ds = ds[...].diff(...)` inside the loop, which silently
    # applied the diff to every op processed afterwards too, regardless of whether
    # it was requested).
    vars_with_splitting_dim = [v for v in ds.data_vars if splitting_dim in ds[v].dims]
    ds_diff = ds[vars_with_splitting_dim].diff(dim=splitting_dim)

    for op_full in statistics_config.ops:
        if op_full.startswith("diff_"):
            op = op_full[len("diff_") :]
            stats[op_full] = _apply_op(ds_diff, op, statistics_config.dims)
        else:
            stats[op_full] = _apply_op(ds, op_full, statistics_config.dims)

    return stats
