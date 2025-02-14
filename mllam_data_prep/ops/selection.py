import warnings

import numpy as np
import pandas as pd


def to_timestamp(s):
    if isinstance(s, str):
        return pd.Timestamp(s)
    return s


def to_timedelta(s):
    if isinstance(s, str):
        return np.timedelta64(pd.to_timedelta(s))
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
        sel_start = selection.start
        sel_end = selection.end
        sel_step = selection.step

        if coord == "time":
            sel_start = to_timestamp(selection.start)
            sel_end = to_timestamp(selection.end)
            sel_step = get_time_step(sel_step, ds)

        assert sel_start != sel_end, "Start and end cannot be the same"

        check_selection(ds, coord, sel_start, sel_end)
        ds = ds.sel({coord: slice(sel_start, sel_end, sel_step)})

        assert (
            len(ds[coord]) > 0
        ), f"You have selected an empty range {sel_start}:{sel_end} for coordinate {coord}"

    return ds


def get_time_step(sel_step, ds):
    if sel_step is None:
        return None

    dataset_timedelta = ds.time[1] - ds.time[0]
    sel_timedelta = to_timedelta(sel_step)
    step = sel_timedelta / dataset_timedelta
    if step % 1 != 0:
        raise ValueError(
            f"The chosen stepsize {sel_step} is not multiple of the stepsize in the dataset {dataset_timedelta}"
        )

    return int(step)


def check_selection(ds, coord, sel_start, sel_end):
    if ds[coord].values.min() < sel_start or ds[coord].values.max() > sel_end:
        warnings.warn(
            f"\nEndpoints are outside the range of {coord}, \nDataset span: [ {ds[coord].values.min()} : {ds[coord].values.max()} ] \nChosen slice: [ {sel_start} : {sel_end} ]\n"
        )
