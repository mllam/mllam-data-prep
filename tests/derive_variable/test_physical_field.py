"""Unit tests for the `mllam_data_prep.ops.derive_variable.physical_field` module."""

import datetime
from typing import List, Union

import numpy as np
import pytest
import xarray as xr

from mllam_data_prep.ops.derive_variable.physical_field import calculate_toa_radiation


@pytest.fixture(name="lat")
def fixture_lat(request) -> List[Union[float, xr.DataArray]]:
    """Fixture that returns test latitude data

    The fixture has to be indirectly parametrized with the number of coordinates,
    the minimum and maximum latitude values.
    """
    ncoord, lat_min, lat_max = request.param
    return [
        55.711,
        xr.DataArray(
            np.random.uniform(lat_min, lat_max, size=(ncoord, ncoord)),
            dims=["x", "y"],
            coords={"x": np.arange(ncoord), "y": np.arange(ncoord)},
            name="lat",
        ),
    ]


@pytest.fixture(name="lon")
def fixture_lon(request) -> List[Union[float, xr.DataArray]]:
    """Fixture that returns test longitude data

    The fixture has to be indirectly parametrized with the number of coordinates,
    the minimum and maximum longitude values.
    """
    ncoord, lon_min, lon_max = request.param
    return [
        12.564,
        xr.DataArray(
            np.random.uniform(lon_min, lon_max, size=(ncoord, ncoord)),
            dims=["x", "y"],
            coords={"x": np.arange(ncoord), "y": np.arange(ncoord)},
            name="lon",
        ),
    ]


@pytest.mark.parametrize(
    "lat",
    # Format: (ncoord, lat_min, lat_max)
    [(10, -90, 90), (10, -40, 40), (10, 40, -40), (10, -10, 10), (1000, -40, 40)],
    indirect=True,
)
@pytest.mark.parametrize(
    "lon",
    # Format: (ncoord, lon_min, lon_max)
    [(10, 0, 360), (10, -180, 180), (10, -90, 90), (10, 100, 110), (1000, -180, 180)],
    indirect=True,
)
@pytest.mark.parametrize("time", [1, 10, 100], indirect=True)
def test_toa_radiation(
    lat: Union[float, xr.DataArray],
    lon: Union[float, xr.DataArray],
    time: Union[np.datetime64, datetime.datetime, xr.DataArray],
):
    """Test the `calculate_toa_radiation` function.

    Function from mllam_data_prep.ops.derive_variable.physical_field.
    """
    if isinstance(time, (xr.DataArray, datetime.datetime)):
        calculate_toa_radiation(lat, lon, time)
    else:
        with pytest.raises(TypeError):
            calculate_toa_radiation(lat, lon, time)
