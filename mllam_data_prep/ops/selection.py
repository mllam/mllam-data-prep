import datetime

import pandas as pd

from ..config import Range


def _normalize_slice_startstop(s):
    if isinstance(s, pd.Timestamp):
        return s
    elif isinstance(s, str):
        try:
            return pd.Timestamp(s)
        except ValueError:
            return s
    else:
        return s


def _normalize_slice_step(s):
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
        if coord not in ds.coords:
            raise ValueError(f"Coordinate {coord} not found in dataset")
        if isinstance(selection, Range):
            if selection.start is None and selection.end is None:
                raise ValueError(
                    f"Selection for coordinate {coord} must have either 'start' and 'end' given"
                )
            sel_start = _normalize_slice_startstop(selection.start)
            sel_end = _normalize_slice_startstop(selection.end)
            sel_step = _normalize_slice_step(selection.step)

            assert sel_start != sel_end, "Start and end cannot be the same"

            # we don't select with the step size for now, but simply check (below) that
            # the step size in the data is the same as the requested step size
            ds = ds.sel({coord: slice(sel_start, sel_end)})

            if coord == "time":
                check_point_in_dataset(coord, sel_start, ds)
                check_point_in_dataset(coord, sel_end, ds)
                if sel_step is not None:
                    check_step(sel_step, coord, ds)

            assert (
                len(ds[coord]) > 0
            ), f"You have selected an empty range {sel_start}:{sel_end} for coordinate {coord}"

        elif isinstance(selection, list):
            ds = ds.sel({coord: selection})
        else:
            raise NotImplementedError(
                f"Selection for coordinate {coord} must be a list or a dict"
            )
    return ds


def check_point_in_dataset(coord, point, ds):
    """
    check that the requested point is in the data.
    """
    if point is not None and point not in ds[coord].values:
        raise ValueError(
            f"Provided value for coordinate {coord} ({point}) is not in the data."
        )


def check_step(sel_step, coord, ds):
    """
    check that the step requested is exactly what the data has
    """
    all_steps = ds[coord].diff(dim=coord).values
    first_step = all_steps[0].astype("timedelta64[s]").astype(datetime.timedelta)

    if not all(all_steps[0] == all_steps):
        raise ValueError(
            f"Step size for coordinate {coord} is not constant: {all_steps}"
        )
    if sel_step != first_step:
        raise ValueError(
            f"Step size for coordinate {coord} is not the same as requested: {first_step} != {sel_step}"
        )
