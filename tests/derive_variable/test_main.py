"""Unit tests for the main module of the derive_variable operations."""

import sys
from types import ModuleType
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest
import xarray as xr

from mllam_data_prep.config import DerivedVariable
from mllam_data_prep.ops.derive_variable.main import (
    _check_and_get_required_attributes,
    _get_derived_variable_function,
    derive_variable,
)


@pytest.fixture(name="mock_import_module")
def fixture_mock_import_module() -> Generator[MagicMock, None, None]:
    """Fixture to mock importlib.import_module."""
    with patch("importlib.import_module") as mock:
        yield mock


@pytest.fixture()
def fixture_mock_sys_modules() -> Generator[None, None, None]:
    """Fixture to mock sys.modules."""
    with patch.dict("sys.modules", {}):
        yield


class TestGetDerivedVariableFunction:
    """Tests for the _get_derived_variable_function."""

    @pytest.mark.usefixtures("fixture_mock_sys_modules")
    def test_function_in_sys_modules(self, mock_import_module: MagicMock) -> None:
        """Test when the function to import is already in sys.modules."""
        # Mock the module and function
        mock_module: ModuleType = MagicMock()
        mock_function: MagicMock = MagicMock()
        sys.modules["mock_module"] = mock_module
        mock_module.mock_function = mock_function

        # Call the function
        result = _get_derived_variable_function("mock_module.mock_function")

        # Assert the function is returned correctly
        assert result == mock_function

        # Assert the module was not imported
        mock_import_module.assert_not_called()

    def test_function_not_in_sys_modules(self, mock_import_module: MagicMock) -> None:
        """Test when the function to import is not in sys.modules."""
        # Mock the module and function
        mock_module: ModuleType = MagicMock()
        mock_function: MagicMock = MagicMock()
        mock_import_module.return_value = mock_module
        mock_module.mock_function = mock_function

        # Call the function
        result = _get_derived_variable_function("mock_module.mock_function")

        # Assert the function is returned correctly
        assert result == mock_function


@patch(
    "mllam_data_prep.ops.derive_variable.main.REQUIRED_FIELD_ATTRIBUTES",
    ["units", "long_name"],
)
class TestCheckAndGetRequiredAttributes:
    """Tests for the _check_and_get_required_attributes function."""

    @pytest.mark.parametrize(
        ["field_attrs", "expected_attributes", "expected_result"],
        [
            [
                {"units": "m", "long_name": "test"},
                {"units": "m", "long_name": "test"},
                {"units": "m", "long_name": "test"},
            ],
            [
                {"units": "m", "long_name": "test"},
                {},
                {"units": "m", "long_name": "test"},
            ],
            [
                {"units": "m"},
                {"units": "m", "long_name": "test"},
                {"units": "m", "long_name": "test"},
            ],
            [
                {"units": "m", "long_name": "old_name"},
                {"units": "m", "long_name": "new_name"},
                {"units": "m", "long_name": "new_name"},
            ],
        ],
    )
    def test_valid_input(
        self, field_attrs, expected_attributes, expected_result
    ) -> None:
        """Test that the function returns the correct attributes with valid input."""
        field = xr.DataArray([1, 2, 3], attrs=field_attrs)

        result = _check_and_get_required_attributes(field, expected_attributes)

        assert result == expected_result

    def test_missing_attributes_raises_key_error(self) -> None:
        """Test when required attributes are missing and not in expected attributes."""
        field = xr.DataArray([1, 2, 3], attrs={"units": "m"})
        expected_attributes = {"units": "m"}

        with pytest.raises(KeyError):
            _check_and_get_required_attributes(field, expected_attributes)


def test_derive_variable_chunks_curvilinear_coordinates() -> None:
    """Chunk required coordinates by their dimensions and restore them."""
    ds = xr.Dataset(
        coords={
            "lat": (("x", "y"), [[50.0, 50.1, 50.2], [51.0, 51.1, 51.2]]),
            "lon": (("x", "y"), [[10.0, 11.0, 12.0], [10.1, 11.1, 12.1]]),
        }
    )
    config = DerivedVariable(
        kwargs={"lat": "ds_input.lat", "lon": "ds_input.lon"},
        function="unused",
        attrs={"units": "degrees", "long_name": "coordinate sum"},
    )
    expected_chunks = ((1, 1), (2, 1))

    def chunk_curvilinear_coordinates(dataset, chunks):
        assert {"lat", "lon"} <= set(dataset.data_vars)
        chunked = dataset.chunk(chunks)
        assert chunked["lat"].chunks == expected_chunks
        assert chunked["lon"].chunks == expected_chunks
        return chunked

    with (
        patch(
            "mllam_data_prep.ops.derive_variable.main.chunk_dataset",
            side_effect=chunk_curvilinear_coordinates,
        ),
        patch(
            "mllam_data_prep.ops.derive_variable.main._get_derived_variable_function",
            return_value=lambda lat, lon: lat + lon,
        ),
    ):
        result = derive_variable(
            ds,
            config,
            chunking={"x": 1, "y": 2},
            target_dims=["x", "y"],
        )

    assert {"lat", "lon"} <= set(result.coords)
    assert result.coords["lat"].chunks == expected_chunks
    assert result.coords["lon"].chunks == expected_chunks
