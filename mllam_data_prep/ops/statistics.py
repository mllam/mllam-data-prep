"""Export functions to calculate statistics for a given Dataset."""

from typing import Callable, Dict, List, Optional

import xarray as xr


def calc_stats(ds: xr.Dataset, statistic_methods: List[str]) -> Dict[str, xr.Dataset]:
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
    for statistic_method in statistic_methods:
        if statistic_method in globals():
            # Apply the operation to the dataset
            statistic_operator: Callable = globals()[statistic_method]
            stats[statistic_method] = statistic_operator(ds=ds)
        else:
            raise NotImplementedError(statistic_method)

    return stats


def compute_pipeline_statistic(
    ds: xr.Dataset,
    stats_op: str,
    stats_dims: str | List[str] = None,
    diff_dim: Optional[str] = None,
    n_diff_steps: Optional[int] = 1,
    groupby: Optional[str] = None,
):
    """
    Apply a series of oprations to compute a specific compound statistic

    The operations applied in order are:
    1. (If diff_dim != None) Apply diff over the `diff_dim` dimension (default to 1 step diff)
    2. (If groupby != None) Apply grouping of dataset according to the `groupby` index
    3. Apply the stats_op to the dataarray over the `stats_dims` dimensions.
       If no stats_dims are provided, apply operator accross time and grid_index.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to compute the statistic on
    stats_op : str
        Statistic operation to apply, must be a valid xarray operation
    diff_dim : str, optional
        Dimension to apply diff over, by default None
    n_diff_steps : int, optional
        Number of steps to compute the diff over, by default 1
    groupby : str, optional
        Index to group over, by default None
    """
    # Convert string to list to unify type of stats_dims
    if isinstance(stats_dims, str):
        stats_dims = [stats_dims]
    # Build up CF compliant cell-method attribute so that people know what
    # operations were applied
    cell_methods = []

    if diff_dim:
        # Only keep variables that have the diff_dim as a dimension
        vars_to_keep = [v for v in ds.data_vars if diff_dim in ds[v].dims]
        if not vars_to_keep:
            raise ValueError(f"No variables found with dimension {diff_dim}")

        # Apply the diff operation
        ds = ds[vars_to_keep].diff(dim=diff_dim, n=n_diff_steps)
        # Get unit of the diff'ed array
        diff_unit_array: xr.DataArray = ds[diff_dim][1] - ds[diff_dim][0]
        diff_unit = diff_unit_array.values
        if diff_dim == "time":
            # Convert the diff unit to hours
            diff_unit = diff_unit.astype("timedelta64[h]")
        else:
            raise NotImplementedError(
                f"diff_dim of type {type(diff_dim)} not supported"
            )

        # Update the cell_methods with the operation applied
        cell_methods.append(f"{diff_dim}: diff (interval: {diff_unit})")

    if groupby:
        # Apply the groupby operation
        ds = ds.groupby(groupby)
        # Update the cell_methods with the operation applied
        cell_methods.append(f"{groupby}: groupby")

    if stats_op:
        if isinstance(stats_op, str):
            # If no stats_dims are provided, apply operator accross time and grid_index
            if not stats_dims:
                stats_dims = ["grid_index", "time"]

            # Build up the cell_methods attribute
            methods = [f"{dim}:" for dim in stats_dims]
            methods.append(stats_op)

            ds = getattr(ds, stats_op)(dim=stats_dims)

            # Update the cell_methods with the operation applied
            cell_methods.extend(methods)
        else:
            raise NotImplementedError(
                f"stats_op of type {type(stats_op)} not supported"
            )

    cell_methods_str = " ".join(cell_methods)
    # Add cell_methods attribute to all variables
    for var in ds.data_vars:
        ds[var].attrs["cell_methods"] = cell_methods_str

    return ds


def mean(ds: xr.Dataset):
    """Compute the mean across grid_index and time for all variables.

    Args:
        ds (xr.Dataset): Input dataset

    Returns:
        xr.Dataset: Dataset with the computed statistical variables
    """
    return compute_pipeline_statistic(ds, stats_op="mean")


def mean_per_gridpoint(ds: xr.Dataset):
    """Compute the mean across time for all variables.

    Args:
        ds (xr.Dataset): Input dataset

    Returns:
        xr.Dataset: Dataset with the computed statistical variables
    """
    return compute_pipeline_statistic(ds, stats_op="mean", stats_dims="time")


def std(ds: xr.Dataset):
    """Compute the standard deviation across grid_index and time for all variables.

    Args:
        ds (xr.Dataset): Input dataset

    Returns:
        xr.Dataset: Dataset with the computed statistical variables
    """
    return compute_pipeline_statistic(ds, stats_op="std")


def std_per_gridpoint(ds: xr.Dataset):
    """Compute the standard deviation across time for all variables.

    Args:
        ds (xr.Dataset): Input dataset

    Returns:
        xr.Dataset: Dataset with the computed statistical variables
    """
    return compute_pipeline_statistic(ds, stats_op="std", stats_dims="time")


def diff_mean(ds: xr.Dataset):
    """Compute the mean across grid_point and time of the difference in time
    for all variables.

    Args:
        ds (xr.Dataset): Input dataset

    Returns:
        xr.Dataset: Dataset with the computed statistical variables
    """
    return compute_pipeline_statistic(
        ds, stats_op="mean", diff_dim="time", n_diff_steps=1
    )


def diff_mean_per_gridpoint(ds: xr.Dataset):
    """Compute the mean across time of the difference in time for all variables.

    Args:
        ds (xr.Dataset): Input dataset

    Returns:
        xr.Dataset: Dataset with the computed statistical variables
    """
    return compute_pipeline_statistic(
        ds, stats_op="mean", stats_dims="time", diff_dim="time", n_diff_steps=1
    )


def diff_std(ds: xr.Dataset):
    """Compute the std across grid_point and time of the difference in time
    for all variables.

    The difference is computed over 1 time step.

    Args:
        ds (xr.Dataset): Input dataset

    Returns:
        xr.Dataset: Dataset with the computed statistical variables
    """
    return compute_pipeline_statistic(
        ds, stats_op="std", diff_dim="time", n_diff_steps=1
    )


def diff_std_per_gridpoint(ds: xr.Dataset):
    """Compute the std across time of the difference in time for all variables.

    The difference is computed over 1 time step.

    Args:
        ds (xr.Dataset): Input dataset

    Returns:
        xr.Dataset: Dataset with the computed statistical variables
    """
    return compute_pipeline_statistic(
        ds, stats_op="std", stats_dims="time", diff_dim="time", n_diff_steps=1
    )


def diurnal_diff_mean(ds: xr.Dataset):
    """Compute the diurnal mean across grid_index and time of the difference in
    time for all variables.

    The data is grouped by time.hour to make the operator be applied accross
    diurnal cycles.
    The difference in time is computed over 1 time step.

    Args:
        ds (xr.Dataset): Input dataset

    Returns:
        xr.Dataset: Dataset with the computed statistical variables
    """
    return compute_pipeline_statistic(
        ds, groupby="time.hour", stats_op="mean", diff_dim="time", n_diff_steps=1
    )


def diurnal_diff_mean_per_gridpoint(ds: xr.Dataset):
    """Compute the diurnal mean across time of the difference in time for all
    variables.

    The data is grouped by time.hour to make the operator be applied accross
    diurnal cycles.
    The difference in time is computed over 1 time step.

    Args:
        ds (xr.Dataset): Input dataset

    Returns:
        xr.Dataset: Dataset with the computed statistical variables
    """
    return compute_pipeline_statistic(
        ds,
        groupby="time.hour",
        stats_op="mean",
        stats_dims="time",
        diff_dim="time",
        n_diff_steps=1,
    )


def diurnal_diff_std(ds: xr.Dataset):
    """Compute the diurnal std across grid_index and time of the difference in
    time for all variables.

    The data is grouped by the hour to make the operator be applied accross
    diurnal cycles.
    The difference in time is computed over 1 time step.

    Args:
        ds (xr.Dataset): Input dataset

    Returns:
        xr.Dataset: Dataset with the computed statistical variables
    """
    return compute_pipeline_statistic(
        ds, groupby="time.hour", stats_op="std", diff_dim="time", n_diff_steps=1
    )


def diurnal_diff_std_per_gridpoint(ds: xr.Dataset):
    """Compute the diurnal std across time of the difference in time for all
    variables.

    The data is grouped by time.hour to make the operator be applied accross
    diurnal cycles.
    The difference in time is computed over 1 time step.

    Args:
        ds (xr.Dataset): Input dataset

    Returns:
        xr.Dataset: Dataset with the computed statistical variables
    """
    return compute_pipeline_statistic(
        ds,
        stats_dims="time",
        groupby="time.hour",
        stats_op="std",
        diff_dim="time",
        n_diff_steps=1,
    )
