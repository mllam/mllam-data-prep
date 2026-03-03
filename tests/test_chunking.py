"""
Unit tests for ops.chunking module.
"""
import numpy as np
import pytest
import xarray as xr
from loguru import logger

from mllam_data_prep.ops.chunking import check_chunk_size, chunk_dataset


@pytest.fixture
def small_dataset():
    """Create a small test dataset."""
    return xr.Dataset(
        {
            "var1": (["x", "y"], np.random.random((10, 10))),
            "var2": (["x", "y"], np.random.random((10, 10))),
        },
        coords={"x": range(10), "y": range(10)},
    )


def test_check_chunk_size_small_chunks(small_dataset):
    """Test check_chunk_size with small chunks (should not warn)."""
    chunks = {"x": 5, "y": 5}
    # Should not raise or warn
    check_chunk_size(small_dataset, chunks)


def test_check_chunk_size_large_chunks(small_dataset):
    """Test check_chunk_size with large chunks (should warn)."""
    # Use chunk sizes that exceed 1GB threshold
    # For float64 (8 bytes), need chunks product > 1GB / 8 = 134217728
    # Using chunks of 12000 x 12000 = 144000000 elements > 134217728
    chunks = {"x": 12000, "y": 12000}
    
    # Capture loguru logs using a handler
    from io import StringIO
    
    log_capture = StringIO()
    handler_id = logger.add(log_capture, format="{message}")
    
    try:
        check_chunk_size(small_dataset, chunks)
        log_output = log_capture.getvalue()
        assert "exceeds" in log_output.lower()
    finally:
        logger.remove(handler_id)


def test_check_chunk_size_missing_dimension(small_dataset):
    """Test check_chunk_size when dimension doesn't exist in variable."""
    chunks = {"x": 5, "z": 10}  # z doesn't exist
    # Should not raise, just skip the missing dimension
    check_chunk_size(small_dataset, chunks)


def test_chunk_dataset_success(small_dataset):
    """Test chunk_dataset successfully chunks a dataset."""
    chunks = {"x": 5, "y": 5}
    chunked = chunk_dataset(small_dataset, chunks)
    assert isinstance(chunked, xr.Dataset)
    # Check that chunking was applied
    assert chunked["var1"].chunks is not None


def test_chunk_dataset_invalid_chunks(small_dataset):
    """Test chunk_dataset with invalid chunk specification."""
    chunks = {"x": -1}  # Invalid chunk size
    with pytest.raises(Exception, match="Error chunking dataset"):
        chunk_dataset(small_dataset, chunks)
