import uuid
from pathlib import Path
from typing import List

import isodate
import numpy as np
import pandas as pd
import xarray as xr

try:
    import pyproj
except ImportError:
    pyproj = None

import mllam_data_prep as mdp

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

DEFAULT_XLIM = DEFAULT_YLIM = DEFAULT_ZLIM = (0.0, 1.0)


def create_surface_forecast_dataset(
    nt_analysis,
    nt_forecast,
    nx,
    ny,
    var_names=DEFAULT_FORECAST_VARS,
    xlim=DEFAULT_XLIM,
    ylim=DEFAULT_YLIM,
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

    x = np.linspace(*xlim, nx)
    y = np.linspace(*ylim, ny)

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
    nt_analysis,
    nx,
    ny,
    var_names=DEFAULT_SURFACE_ANALYSIS_VARS,
    xlim=DEFAULT_XLIM,
    ylim=DEFAULT_YLIM,
):
    """
    Create a fake analysis dataset with `nt_analysis` analysis times, `nx` grid points
    in x-direction and `ny` grid points in y-direction.
    """
    ts_analysis = pd.date_range(
        T_START, periods=nt_analysis, freq=DT_ANALYSIS
    ).tz_localize(None)

    x = np.linspace(*xlim, nx)
    y = np.linspace(*ylim, ny)

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
    xlim=DEFAULT_XLIM,
    ylim=DEFAULT_YLIM,
    zlim=DEFAULT_ZLIM,
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
    xlim : tuple, optional
        Tuple of the form (xmin, xmax) defining the x-limits, by default DEFAULT_XLIM
    ylim : tuple, optional
        Tuple of the form (ymin, ymax) defining the y-limits, by default DEFAULT_YLIM
    zlim : tuple, optional
        Tuple of the form (zmin, zmax) defining the z-limits, by default DEFAULT_ZLIM

    Returns
    -------
    xarray.Dataset
        The created dataset
    """
    ts_analysis = pd.date_range(
        T_START, periods=nt_analysis, freq=DT_ANALYSIS
    ).tz_localize(None)

    x = np.linspace(*xlim, nx)
    y = np.linspace(*ylim, ny)
    z = np.linspace(*zlim, nz)

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
    xlim=DEFAULT_XLIM,
    ylim=DEFAULT_YLIM,
    zlim=DEFAULT_ZLIM,
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
    xlim : tuple, optional
        Tuple of the form (xmin, xmax) defining the x-limits, by default DEFAULT_XLIM
    ylim : tuple, optional
        Tuple of the form (ymin, ymax) defining the y-limits, by default DEFAULT_YLIM
    zlim : tuple, optional
        Tuple of the form (zmin, zmax) defining the z-limits, by default DEFAULT_ZLIM

    Returns
    -------
    xarray.Dataset
        The created dataset
    """

    ts_analysis = pd.date_range(
        T_START, periods=nt_analysis, freq=DT_ANALYSIS
    ).tz_localize(None)
    ts_forecast = pd.date_range(
        T_START, periods=nt_forecast, freq=DT_FORECAST
    ).tz_localize(None)

    x = np.linspace(*xlim, nx)
    y = np.linspace(*ylim, ny)
    z = np.linspace(*zlim, nz)

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


def create_static_dataset(
    nx, ny, var_names=DEFAULT_STATIC_VARS, xlim=DEFAULT_XLIM, ylim=DEFAULT_YLIM
):
    """
    Create a fake static dataset with `nx` grid points in x-direction and `ny` grid points in y-direction.

    Parameters
    ----------
    nx : int
        Number of grid points in x-direction
    ny : int
        Number of grid points in y-direction
    var_names : list, optional
        List of variable names to create, by default DEFAULT_STATIC_VARS
    xlim : tuple, optional
        Tuple of the form (xmin, xmax) defining the x-limits, by default DEFAULT_XLIM
    ylim : tuple, optional
        Tuple of the form (ymin, ymax) defining the y-limits, by default DEFAULT_YLIM

    Returns
    -------
    xarray.Dataset
        The created dataset
    """
    x = np.linspace(*xlim, nx)
    y = np.linspace(*ylim, ny)

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


def _add_latlon(ds: xr.Dataset) -> xr.Dataset:
    """
    Add latitude and longitude coordinates to the dataset using a local equal
    area projection centered on Denmark.
    """
    if pyproj is None:
        raise ImportError("pyproj is required for this function")

    da_x = ds.coords["x"]
    da_y = ds.coords["y"]
    xs, ys = np.meshgrid(da_x, da_y, indexing="ij")
    proj = pyproj.Proj(proj="laea", lon_0=12.25, lat_0=55.65)
    lon, lat = proj(xs, ys, inverse=True)
    dims = da_x.dims + da_y.dims
    coords = {d: ds.coords[d] for d in dims}
    da_lat = xr.DataArray(lat, coords=coords, dims=dims)
    da_lon = xr.DataArray(lon, coords=coords, dims=dims)

    ds.coords["lon"] = da_lon
    ds.coords["lat"] = da_lat

    return ds


def create_data_collection(
    data_kinds,
    fp_root,
    xlim=DEFAULT_XLIM,
    ylim=DEFAULT_YLIM,
    nx=NX,
    ny=NY,
    nz=NZ,
    nt_analysis=NT_ANALYSIS,
    nt_forecast=NT_FORECAST,
    add_latlon=False,
):
    """
    Create a fake data collection with the given `data_kinds` and save it to `fp_root`, with
    each dataset having the `data_kind` name with a unique suffix and saved in `.zarr` format.


    Parameters
    ----------
    data_kinds : list
        List of data kinds to create, e.g. ["surface_forecast", "static"]
    fp_root : str
        Root directory to save the data collection
    xlim : tuple, optional
        Tuple of the form (xmin, xmax) defining the x-limits, by default DEFAULT_XLIM
    ylim : tuple, optional
        Tuple of the form (ymin, ymax) defining the y-limits, by default DEFAULT_YLIM
    nx : int, optional
        Number of grid points in x-direction, by default NX
    ny : int, optional
        Number of grid points in y-direction, by default NY
    nz : int, optional
        Number of levels, by default NZ
    nt_analysis : int, optional
        Number of analysis times, by default NT_ANALYSIS
    nt_forecast : int, optional
        Number of forecast times, by default NT_FORECAST

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
            ds = create_surface_forecast_dataset(
                nt_analysis=nt_analysis,
                nt_forecast=nt_forecast,
                nx=nx,
                ny=ny,
                xlim=xlim,
                ylim=ylim,
            )
        elif data_kind == "surface_analysis":
            ds = create_surface_analysis_dataset(
                nt_analysis=nt_analysis, nx=nx, ny=ny, xlim=xlim, ylim=ylim
            )
        elif data_kind == "analysis_on_levels":
            ds = create_analysis_dataset_on_levels(
                nt_analysis=nt_analysis, nx=nx, ny=ny, nz=nz, xlim=xlim, ylim=ylim
            )
        elif data_kind == "forecast_on_levels":
            ds = create_forecast_dataset_on_levels(
                nt_analysis=nt_analysis,
                nt_forecast=nt_forecast,
                nx=nx,
                ny=ny,
                nz=nz,
                xlim=xlim,
                ylim=ylim,
            )
        elif data_kind == "static":
            ds = create_static_dataset(nx=nx, ny=ny, xlim=xlim, ylim=ylim)
        else:
            raise ValueError(f"Unknown data kind: {data_kind}")

        identifier = str(uuid.uuid4())
        dataset_name = f"{data_kind}_{identifier}"

        if add_latlon:
            ds = _add_latlon(ds)

        fp = f"{fp_root}/{dataset_name}.zarr"
        ds.to_zarr(fp, mode="w")
        datasets[data_kind] = fp

    return datasets


def create_input_datasets_and_config(
    identifier: str,
    tmpdir: Path,
    data_categories: List[str],
    xlim: List[float] = DEFAULT_XLIM,
    ylim: List[float] = DEFAULT_YLIM,
    nx: int = NX,
    ny: int = NY,
    add_latlon: bool = False,
):
    """
    Create a config and input datasets with test data for it with a given set
    of data categories.

    Parameters
    ----------
    identifier : str
        Named identifier for the data collection
    tmpdir : Path
        Temporary directory to save the data collection
    data_catagories : List[str]
        List of categories of data to create, from state/forcing/static.
    xlim : List[float], optional
        List of the form [xmin, xmax] defining the x-limits, by default DEFAULT_XLIM
    ylim : List[float], optional
        List of the form [ymin, ymax] defining the y-limits, by default DEFAULT_YLIM
    nx : int, optional
        Number of grid points in x-direction, by default NX
    ny : int, optional
        Number of grid points in y-direction, by default NY

    Returns
    -------
    mdp.Config
        The created config
    """

    output_variables = {}
    inputs = {}

    for data_category in data_categories:
        input_dims = []
        output_dims = []
        if data_category in ["state", "forcing"]:
            data_kinds = ["surface_analysis"]
            variable_names = DEFAULT_SURFACE_ANALYSIS_VARS
            input_dims.append("analysis_time")
            output_dims.append("time")
        elif data_category == "static":
            data_kinds = ["static"]
            variable_names = DEFAULT_STATIC_VARS
        else:
            raise NotImplementedError(f"Unknown data category: {data_category}")
        input_dims.extend(["x", "y"])
        output_dims += ["grid_index", f"{data_category}_feature"]

        datasets = create_data_collection(
            data_kinds=data_kinds,
            fp_root=Path(tmpdir.name) / identifier,
            xlim=xlim,
            ylim=ylim,
            nx=nx,
            ny=ny,
            add_latlon=add_latlon,
        )

        assert len(datasets) == 1
        dataset_kind, dataset_path = datasets.popitem()

        output_variables[data_category] = output_dims
        dim_mapping = {}

        for d in output_dims:
            if d == "time":
                dim_mapping[d] = mdp.config.DimMapping(
                    method="rename",
                    dim="analysis_time",
                )
            elif d == "grid_index":
                dim_mapping[d] = mdp.config.DimMapping(
                    method="stack",
                    dims=["x", "y"],
                )
            else:
                dim_mapping[d] = mdp.config.DimMapping(
                    method="stack_variables_by_var_name",
                    name_format="{var_name}",
                )

        inputs[f"{identifier}_{dataset_kind}"] = mdp.config.InputDataset(
            path=dataset_path,
            dims=input_dims,
            variables=variable_names,
            dim_mapping=dim_mapping,
            target_output_variable=data_category,
        )

    config = mdp.Config(
        schema_version=SCHEMA_VERSION,
        dataset_version="v0.1.0",
        output=mdp.config.Output(
            variables=output_variables,
        ),
        inputs=inputs,
    )

    return config
