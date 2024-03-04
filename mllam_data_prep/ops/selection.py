import datetime

import pandas as pd


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


def select_by_kwargs(ds, **config_dict):
    """
    Do `xr.Dataset.sel` on `ds` using the `config_dict` to select the coordinates, for each
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
    config_dict : dict
        Dictionary with the coordinate names as keys and the selection to make as values,
        either a list of values or a dictionary with keys "start" and "end"
        (and optionally "step" for the slice object)

    Returns
    -------
    xr.Dataset
        Dataset with the selection made
    """
    for coord, selection in config_dict.items():
        if coord not in ds.coords:
            raise ValueError(f"Coordinate {coord} not found in dataset")
        if isinstance(selection, dict):
            if "start" not in selection and "end" not in selection:
                raise ValueError(
                    f"Selection for coordinate {coord} must have either 'start' and 'end' given"
                )
            start = _normalize_slice_startstop(selection.get("start"))
            end = _normalize_slice_startstop(selection.get("end"))
            step = _normalize_slice_step(selection.get("step"))

            ds_orig = ds

            ds = ds.sel({coord: slice(start, end)})

            # check that the start and end are in the data
            coord_minmax = ds_orig[coord].min().values, ds_orig[coord].max().values
            if start is not None and start not in ds[coord].values:
                raise ValueError(
                    f"Provided start value for coordinate {coord} ({start}) is not in the data."
                    f"Coord {coord} spans [{coord_minmax[0]}, {coord_minmax[1]}]"
                )
            if end is not None and end not in ds[coord].values:
                raise ValueError(
                    f"Provided end value for coordinate {coord} ({end}) is not in the data. "
                    f"Coord {coord} spans [{coord_minmax[0]}, {coord_minmax[1]}]"
                )

            if step is not None:
                # check that the step requested is exactly what the data has
                all_steps = ds[coord].diff(dim=coord).values
                first_step = (
                    all_steps[0].astype("timedelta64[s]").astype(datetime.timedelta)
                )

                if not all(all_steps[0] == all_steps):
                    raise ValueError(
                        f"Step size for coordinate {coord} is not constant: {all_steps}"
                    )
                if step != first_step:
                    raise ValueError(
                        f"Step size for coordinate {coord} is not the same as requested: {first_step} != {step}"
                    )

        elif isinstance(selection, list):
            ds = ds.sel({coord: selection})
        else:
            raise NotImplementedError(
                f"Selection for coordinate {coord} must be a list or a dict"
            )
    return ds
