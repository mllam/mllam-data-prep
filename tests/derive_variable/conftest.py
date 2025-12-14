"""Fixtures for the derive_variable module tests."""

import datetime
from typing import List, Union

import isodate
import numpy as np
import pandas as pd
import pytest
import xarray as xr


@pytest.fixture(name="time")
def fixture_time(
    request,
) -> List[Union[np.datetime64, datetime.datetime, xr.DataArray]]:
    """Fixture that returns test time data

    The fixture has to be indirectly parametrized with the number of time steps.
    """
    ntime = request.param
    return [
        np.datetime64("2004-06-11T00:00:00"),  # invalid type
        isodate.parse_datetime("1999-03-21T00:00"),
        xr.DataArray(
            pd.date_range(
                start=isodate.parse_datetime("1999-03-21T00:00"),
                periods=ntime,
                freq=isodate.parse_duration("PT1H"),
            ),
            dims=["time"],
            name="time",
        ),
    ]
