import tempfile

import pytest
import xarray as xr

from mllam_data_prep.cli import call


@pytest.mark.parametrize("args", [["example.danra.yaml"]])
def test_call(args):
    with tempfile.TemporaryDirectory(suffix=".zarr") as tmpdir:
        args.extend(["--output", tmpdir])
        call(args)
        _ = xr.open_zarr(tmpdir)
