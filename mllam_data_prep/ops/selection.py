import datetime
import warnings

import pandas as pd

from ..config import Range


def normalize_slice_startstop(s):
    if isinstance(s, pd.Timestamp):
        return s
    elif isinstance(s, str):
        try:
            return pd.Timestamp(s)
        except ValueError:
            return s
    else:
        return s


def normalize_slice_step(s):
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
            sel_start = normalize_slice_startstop(selection.start)
            sel_end = normalize_slice_startstop(selection.end)
            sel_step = normalize_slice_step(selection.step)

            assert sel_start != sel_end, "Start and end cannot be the same"

            # TODO Implement handling of time step size. See issue #69
            if coord == "time" and sel_step is not None:
                warnings.warn(
                    "Step size for time coordinate is not yet supported and is ignored"
                )
                sel_step = None
            ################

            ds = ds.sel({coord: slice(sel_start, sel_end, sel_step)})

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
