"""Unit tests for the `mllam_data_prep.ops.derive_variable.time_components` module."""

import datetime
from typing import Union

import numpy as np
import pytest
import xarray as xr

from mllam_data_prep.ops.derive_variable.time_components import (
    calculate_day_of_year,
    calculate_hour_of_day,
)


@pytest.mark.parametrize("time", [1, 10, 1000], indirect=True)
@pytest.mark.parametrize(
    "component",
    [
        "cos",
        "sin",
    ],
)
def test_hour_of_day(
    time: Union[np.datetime64, datetime.datetime, xr.DataArray], component: str
):
    """Test the `calculate_hour_of_day` function.

    Function from mllam_data_prep.ops.derive_variable.time_components.
    """
    if isinstance(time, (xr.DataArray, datetime.datetime)):
        calculate_hour_of_day(time, component=component)
    else:
        with pytest.raises(TypeError):
            calculate_hour_of_day(time, component=component)


@pytest.mark.parametrize("time", [1, 10, 1000], indirect=True)
@pytest.mark.parametrize(
    "component",
    [
        "cos",
        "sin",
    ],
)
def test_day_of_year(
    time: Union[np.datetime64, datetime.datetime, xr.DataArray], component: str
):
    """Test the `calculate_day_of_year` function.

    Function from mllam_data_prep.ops.derive_variable.time_components.
    """

    if isinstance(time, (xr.DataArray, datetime.datetime)):
        calculate_day_of_year(time, component=component)
    else:
        with pytest.raises(TypeError):
            calculate_day_of_year(time, component=component)
