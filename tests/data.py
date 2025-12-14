import uuid

import isodate
import numpy as np
import pandas as pd
import xarray as xr

SCHEMA_VERSION = "v0.5.0"

NX, NY = 10, 8
NT_ANALYSIS, NT_FORECAST = 5, 12
NZ = 3
DT_ANALYSIS = isodate.parse_duration("PT6H")
DT_FORECAST = isodate.parse_duration("PT1H")
T_START = isodate.parse_datetime("2000-01-01T00:00")
T_END_ANALYSIS = T_START + (NT_ANALYSIS - 1) * DT_ANALYSIS
T_END_FORECAST = T_START + (NT_FORECAST - 1) * DT_FORECAST
DEFAULT_FORECAST_VARS = ["u", "v", "t", "precip"]
DEFAULT_ATMOSPHERIC_ANALYSIS_VARS = ["u", "v", "t"]
DEFAULT_SURFACE_ANALYSIS_VARS = [
    "pres_seasurface",
]
DEFAULT_STATIC_VARS = ["topography_height", "land_area_fraction"]
ALL_DATA_KINDS = [
    "surface_forecast",
    "surface_analysis",
    "analysis_on_levels",
    "forecast_on_levels",
    "static",
]


def create_surface_forecast_dataset(
    nt_analysis, nt_forecast, nx, ny, var_names=DEFAULT_FORECAST_VARS
):
    """
    Create a fake forecast dataset with `nt_analysis` analysis times, `nt_forecast`
    forecast times, `nx` grid points in x-direction and `ny` grid points in y-direction.
    """
    ts_analysis = pd.date_range(
        T_START, periods=nt_analysis, freq=DT_ANALYSIS
    ).tz_localize(None)
    ts_forecast = pd.date_range(
        T_START, periods=nt_forecast, freq=DT_FORECAST
    ).tz_localize(None)

    x = np.arange(nx)
    y = np.arange(ny)

    dataarrays = {}
    for var_name in var_names:
        da = xr.DataArray(
            np.random.random((nt_analysis, nt_forecast, nx, ny)),
            dims=["analysis_time", "forecast_time", "x", "y"],
            coords={
                "analysis_time": ts_analysis,
                "forecast_time": ts_forecast,
                "x": x,
                "y": y,
            },
        )
        dataarrays[var_name] = da

    ds = xr.Dataset(dataarrays)
    return ds


def create_surface_analysis_dataset(
    nt_analysis, nx, ny, var_names=DEFAULT_SURFACE_ANALYSIS_VARS
):
    """
    Create a fake analysis dataset with `nt_analysis` analysis times, `nx` grid points
    in x-direction and `ny` grid points in y-direction.
    """
    ts_analysis = pd.date_range(
        T_START, periods=nt_analysis, freq=DT_ANALYSIS
    ).tz_localize(None)

    x = np.arange(nx)
    y = np.arange(ny)

    dataarrays = {}
    for var_name in var_names:
        da = xr.DataArray(
            np.random.random((nt_analysis, nx, ny)),
            dims=["analysis_time", "x", "y"],
            coords={
                "analysis_time": ts_analysis,
                "x": x,
                "y": y,
            },
        )
        dataarrays[var_name] = da

    ds = xr.Dataset(dataarrays)
    return ds


def create_analysis_dataset_on_levels(
    nt_analysis,
    nx,
    ny,
    nz,
    level_dim="altitude",
    var_names=DEFAULT_ATMOSPHERIC_ANALYSIS_VARS,
):
    """
    Create a fake analysis dataset with `nt_analysis` analysis times, `nx` grid points in x-direction,
    `ny` grid points in y-direction and `nz` levels, using level dimension `level_dim`.

    Parameters
    ----------
    nt_analysis : int
        Number of analysis times
    nx : int
        Number of grid points in x-direction
    ny : int
        Number of grid points in y-direction
    nz : int
        Number of levels
    level_dim : str, optional
        Name of the level dimension, by default "altitude"
    """
    ts_analysis = pd.date_range(
        T_START, periods=nt_analysis, freq=DT_ANALYSIS
    ).tz_localize(None)

    x = np.arange(nx)
    y = np.arange(ny)
    z = np.arange(nz)

    dataarrays = {}
    for var_name in var_names:
        da = xr.DataArray(
            np.random.random((nt_analysis, nz, nx, ny)),
            dims=["analysis_time", level_dim, "x", "y"],
            coords={
                "analysis_time": ts_analysis,
                level_dim: z,
                "x": x,
                "y": y,
            },
        )
        dataarrays[var_name] = da

    ds = xr.Dataset(dataarrays)
    return ds


def create_forecast_dataset_on_levels(
    nt_analysis,
    nt_forecast,
    nx,
    ny,
    nz,
    level_dim="altitude",
    var_names=DEFAULT_FORECAST_VARS,
):
    """
    Create a fake forecast dataset with `nt_analysis` analysis times, `nt_forecast`
    forecast times, `nx` grid points in x-direction, `ny` grid points in y-direction
    and `nz` levels, using level dimension `level_dim`.

    Parameters
    ----------
    nt_analysis : int
        Number of analysis times
    nt_forecast : int
        Number of forecast times
    nx : int
        Number of grid points in x-direction
    ny : int
        Number of grid points in y-direction
    nz : int
        Number of levels
    level_dim : str, optional
        Name of the level dimension, by default "altitude"
    """

    ts_analysis = pd.date_range(
        T_START, periods=nt_analysis, freq=DT_ANALYSIS
    ).tz_localize(None)
    ts_forecast = pd.date_range(
        T_START, periods=nt_forecast, freq=DT_FORECAST
    ).tz_localize(None)

    x = np.arange(nx)
    y = np.arange(ny)
    z = np.arange(nz)

    dataarrays = {}
    for var_name in var_names:
        da = xr.DataArray(
            np.random.random((nt_analysis, nt_forecast, nz, nx, ny)),
            dims=["analysis_time", "forecast_time", level_dim, "x", "y"],
            coords={
                "analysis_time": ts_analysis,
                "forecast_time": ts_forecast,
                level_dim: z,
                "x": x,
                "y": y,
            },
        )
        dataarrays[var_name] = da

    ds = xr.Dataset(dataarrays)
    return ds


def create_static_dataset(nx, ny, var_names=DEFAULT_STATIC_VARS):
    """
    Create a fake static dataset with `nx` grid points in x-direction and `ny` grid points in y-direction.
    """
    x = np.arange(nx)
    y = np.arange(ny)

    dataarrays = {}
    for var_name in var_names:
        da = xr.DataArray(
            np.random.random((nx, ny)),
            dims=["x", "y"],
            coords={
                "x": x,
                "y": y,
            },
        )
        dataarrays[var_name] = da

    ds = xr.Dataset(dataarrays)
    return ds


def create_data_collection(data_kinds, fp_root):
    """
    Create a fake data collection with the given `data_kinds` and save it to `fp_root`, with
    each dataset having the `data_kind` name with a unique suffix and saved in `.zarr` format.


    Parameters
    ----------
    data_kinds : list
        List of data kinds to create, e.g. ["surface_forecast", "static"]
    fp_root : str
        Root directory to save the data collection

    Returns
    -------
    dict
        Dictionary of the created datasets with the key being the data_kind
        and value being the path to the saved dataset
    """
    datasets = {}

    # check that non of the data_kinds are repeated
    if len(data_kinds) != len(set(data_kinds)):
        raise ValueError(
            "Data kinds should be unique, you're welcome to call this function twice :)"
        )

    for data_kind in data_kinds:
        if data_kind == "surface_forecast":
            ds = create_surface_forecast_dataset(NT_ANALYSIS, NT_FORECAST, NX, NY)
        elif data_kind == "surface_analysis":
            ds = create_surface_analysis_dataset(NT_ANALYSIS, NX, NY)
        elif data_kind == "analysis_on_levels":
            ds = create_analysis_dataset_on_levels(NT_ANALYSIS, NX, NY, NZ)
        elif data_kind == "forecast_on_levels":
            ds = create_forecast_dataset_on_levels(NT_ANALYSIS, NT_FORECAST, NX, NY, NZ)
        elif data_kind == "static":
            ds = create_static_dataset(NX, NY)
        else:
            raise ValueError(f"Unknown data kind: {data_kind}")

        identifier = str(uuid.uuid4())
        dataset_name = f"{data_kind}_{identifier}"

        fp = f"{fp_root}/{dataset_name}.zarr"
        ds.to_zarr(fp, mode="w")
        datasets[data_kind] = fp

    return datasets
