import numpy as np
import pytest
import xarray as xr

from mllam_data_prep.config import Transformation
from mllam_data_prep.ops.transform import apply_transformation


@pytest.fixture
def sample_da():
    return xr.DataArray(
        np.array([0.0, 0.5, 2.0, 50.0]),
        dims=["x"],
        attrs={"units": "mm", "long_name": "precipitation"},
    )


def test_log1p_transformation_values(sample_da):
    t = Transformation(method="log1p")
    da_t = apply_transformation(sample_da, t)
    np.testing.assert_allclose(da_t.values, np.log1p(sample_da.values))


def test_log1p_transformation_attrs_auto_inverse(sample_da):
    t = Transformation(method="log1p")
    da_t = apply_transformation(sample_da, t)
    assert da_t.attrs["transformation"] == "log1p"
    assert da_t.attrs["inverse"] == "expm1"
    # original attrs preserved
    assert da_t.attrs["units"] == "mm"
    assert da_t.attrs["long_name"] == "precipitation"


def test_explicit_inverse_override(sample_da):
    t = Transformation(method="log1p", inverse="custom_inverse_fn")
    da_t = apply_transformation(sample_da, t)
    assert da_t.attrs["inverse"] == "custom_inverse_fn"


def test_round_trip(sample_da):
    t = Transformation(method="log1p")
    da_t = apply_transformation(sample_da, t)
    inv_fn = getattr(np, da_t.attrs["inverse"])
    np.testing.assert_allclose(inv_fn(da_t.values), sample_da.values)


def test_unsupported_method_raises(sample_da):
    t = Transformation(method="not_a_real_transform")
    with pytest.raises(NotImplementedError):
        apply_transformation(sample_da, t)
