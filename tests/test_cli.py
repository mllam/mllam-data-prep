import subprocess
import tempfile

import pytest
import xarray as xr

from mllam_data_prep.cli import call


@pytest.mark.slow
@pytest.mark.parametrize("args", [["example.danra.yaml"]])
def test_call(args):
    with tempfile.TemporaryDirectory(suffix=".zarr") as tmpdir:
        args.extend(["--output", tmpdir])
        call(args)
        _ = xr.open_zarr(tmpdir)


def test_cli_entrypoint_help():
    """Test the actual installed CLI entry point works as a subprocess."""
    result = subprocess.run(
        ["mllam_data_prep", "--help"],
        capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "config" in result.stdout.lower()
