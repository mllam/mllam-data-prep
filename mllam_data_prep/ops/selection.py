import warnings

import pandas as pd

from mllam_data_prep.config import Range


def str_to_datetime(s):
    if isinstance(s, pd.Timestamp):
        return s
    elif isinstance(s, str):
        try:
            return pd.Timestamp(s)
        except ValueError:
            return s
    else:
        return s


def str_to_timedelta(s):
    if isinstance(s, pd.Timedelta):
        return s
    elif isinstance(s, str):
        try:
            return pd.to_timedelta(s)
        except ValueError:
            return s
    else:
        return s


def select_by_kwargs(ds, **coord_ranges):
    """
    Do `xr.Dataset.sel` on `ds` using the `coord_ranges` to select the coordinates, for each
    entry in the dictionary, the key is the coordinate name and the value is the selection
    to make, either given as 1) a list of values or a 2) dictionary with keys "start" and "end".
    This functionally works like `xr.Dataset.sel` but can create slice objects for each
    selection from the dictionary provided and also supports the use of ISO 8601 duration strings. In addition
    the `step` size is used to check that the step size in the data is the same as the requested step size.

    In future time interpolation and subsampling could be done here

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to select from
    coord_ranges : dict
        Dictionary with the coordinate names as keys and the selection to make as values,
        either a list of values or a dictionary with keys "start" and "end"
        (and optionally "step" for the slice object)

    Returns
    -------
    xr.Dataset
        Dataset with the selection made
    """

    for coord, selection in coord_ranges.items():
        sel_start = str_to_datetime(selection.start)
        sel_end = str_to_datetime(selection.end)
        sel_step = str_to_timedelta(selection.step)

        assert sel_start != sel_end, "Start and end cannot be the same"

        if coord == "time" and sel_step is not None:
            warnings.warn(
                "Step size for time coordinate is not supported and is ignored"
            )
            sel_step = None

        check_selection(ds, coord, sel_start, sel_end)
        ds = ds.sel({coord: slice(sel_start, sel_end, sel_step)})

        assert (
            len(ds[coord]) > 0
        ), f"You have selected an empty range {sel_start}:{sel_end} for coordinate {coord}"

    return ds

def check_selection(ds, coord, sel_start, sel_end):
    if ds[coord].values.min() < sel_start or ds[coord].values.max() > sel_end:
        warnings.warn(
            f"Selection points is outside the range of the range of {coord}, the data spans from {ds[coord].values.min()} to {ds[coord].values.max()}"
        )
