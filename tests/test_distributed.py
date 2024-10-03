import importlib
import tempfile

import pytest
import xarray as xr

from mllam_data_prep.cli import call


def call_wrapper(args):
    with tempfile.TemporaryDirectory(suffix=".zarr") as tmpdir:
        args.extend(["--output", tmpdir])
        call(args)
        _ = xr.open_zarr(tmpdir)


def distributed():
    """Check if dask.distributed is installed"""
    try:
        importlib.import_module("dask.distributed")

        return True
    except ModuleNotFoundError:
        return False


@pytest.mark.parametrize(
    "args", [["example.danra.yaml", "--dask-distributed-local-core-fraction", "0.1"]]
)
def test_run_distributed(args):
    if distributed():
        call_wrapper(args)
    else:
        pytest.raises(
            ModuleNotFoundError,
            call_wrapper(args),
            match="Currently dask.distrubuted is not installed",
        )
