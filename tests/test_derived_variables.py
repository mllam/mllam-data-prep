import datetime
import random
from unittest.mock import patch

import isodate
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import mllam_data_prep as mdp

NCOORD = 10
NTIME = 10
LAT_MIN = -90
LAT_MAX = 90
LON_MIN = 0
LON_MAX = 360
LATITUDE = [
    55.711,
    xr.DataArray(
        np.random.uniform(LAT_MIN, LAT_MAX, size=(NCOORD, NCOORD)),
        dims=["x", "y"],
        coords={"x": np.arange(NCOORD), "y": np.arange(NCOORD)},
        name="lat",
    ),
]
LONGITUDE = [
    12.564,
    xr.DataArray(
        np.random.uniform(LON_MIN, LON_MAX, size=(NCOORD, NCOORD)),
        dims=["x", "y"],
        coords={"x": np.arange(NCOORD), "y": np.arange(NCOORD)},
        name="lon",
    ),
]
TIME = [
    np.datetime64("2004-06-11T00:00:00"),  # invalid type
    isodate.parse_datetime("1999-03-21T00:00"),
    xr.DataArray(
        pd.date_range(
            start=isodate.parse_datetime("1999-03-21T00:00"),
            periods=NTIME,
            freq=isodate.parse_duration("PT1H"),
        ),
        dims=["time"],
        name="time",
    ),
]


def mock_cyclic_encoding(data, data_max):
    """Mock the `cyclic_encoding` function from mllam_data_prep.ops.derived_variables."""
    if isinstance(data, xr.DataArray):
        data_cos = xr.DataArray(
            random.uniform(-1, 1),
            coords=data.coords,
            dims=data.dims,
        )
        data_sin = xr.DataArray(
            random.uniform(-1, 1),
            coords=data.coords,
            dims=data.dims,
        )
        return data_cos, data_sin
    elif isinstance(data, (float, int)):
        return random.uniform(-1, 1), random.uniform(-1, 1)


@pytest.mark.parametrize("lat", LATITUDE)
@pytest.mark.parametrize("lon", LONGITUDE)
@pytest.mark.parametrize("time", TIME)
def test_toa_radiation(lat, lon, time):
    """
    Test the `calculate_toa_radiation` function from mllam_data_prep.derived_variables
    """
    with patch(
        "mllam_data_prep.ops.derived_variables.cyclic_encoding",
        side_effect=mock_cyclic_encoding,
    ):
        if isinstance(time, (xr.DataArray, datetime.datetime)):
            mdp.ops.derive_variable.physical_field.calculate_toa_radiation(
                lat, lon, time
            )
        else:
            with pytest.raises(TypeError):
                mdp.ops.derive_variable.physical_field.calculate_toa_radiation(
                    lat, lon, time
                )


@pytest.mark.parametrize("time", TIME)
def test_hour_of_day(time):
    """
    Test the `calculate_hour_of_day` function from mllam_data_prep.derived_variables
    """
    with patch(
        "mllam_data_prep.ops.derived_variables.cyclic_encoding",
        side_effect=mock_cyclic_encoding,
    ):
        if isinstance(time, (xr.DataArray, datetime.datetime)):
            mdp.ops.derive_variable.time_components.calculate_hour_of_day(time)
        else:
            with pytest.raises(TypeError):
                mdp.ops.derive_variable.time_components.calculate_hour_of_day(time)


@pytest.mark.parametrize("time", TIME)
def test_day_of_year(time):
    """
    Test the `calculate_day_of_year` function from mllam_data_prep.derived_variables
    """
    with patch(
        "mllam_data_prep.ops.derived_variables.cyclic_encoding",
        side_effect=mock_cyclic_encoding,
    ):
        if isinstance(time, (xr.DataArray, datetime.datetime)):
            mdp.ops.derive_variable.time_components.calculate_day_of_year(time)
        else:
            with pytest.raises(TypeError):
                mdp.ops.derive_variable.time_components.calculate_day_of_year(time)
