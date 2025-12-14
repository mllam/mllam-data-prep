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
    except (ModuleNotFoundError, ImportError):
        return False


@pytest.mark.parametrize(
    "args",
    [
        ["example.danra.yaml", "--dask-distributed-local-core-fraction", "1.0"],
        ["example.danra.yaml", "--dask-distributed-local-core-fraction", "0.0"],
        ["example.danra.yaml"],
    ],
)
def test_run_distributed(args):
    if distributed():
        call_wrapper(args)
    elif not distributed() and "--dask-distributed-local-core-fraction" in args:
        index = args.index("--dask-distributed-local-core-fraction")
        core_fraction = float(args[index + 1])
        if core_fraction > 0:
            pytest.raises(
                ModuleNotFoundError,
                call_wrapper,
                args=args,
            )
        else:
            call_wrapper(args)
    else:
        call_wrapper(args)
