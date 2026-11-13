import numpy as np
import xarray as xr

from ..config import Transformation

TRANSFORM_REGISTRY = {
    "log1p": (np.log1p, "expm1"),
}


def apply_transformation(
    da: xr.DataArray, transformation: Transformation
) -> xr.DataArray:
    """
    Apply a transformation to a DataArray's values, and record the forward
    and inverse transformation names in the DataArray's attributes so that
    downstream consumers (e.g. neural-lam) know how to invert the transform
    when converting predictions back to physical values.

    Parameters
    ----------
    da : xr.DataArray
        The variable to transform.
    transformation : Transformation
        The transformation to apply.

    Returns
    -------
    xr.DataArray
        The transformed variable, with `transformation` and `inverse`
        attributes set.
    """
    method = transformation.method
    if method not in TRANSFORM_REGISTRY:
        raise NotImplementedError(
            f"Transformation method '{method}' is not supported. Supported "
            f"methods are: {list(TRANSFORM_REGISTRY.keys())}"
        )
    fn, default_inverse = TRANSFORM_REGISTRY[method]
    inverse = transformation.inverse or default_inverse

    attrs = da.attrs
    da_transformed = xr.apply_ufunc(fn, da, dask="allowed")
    da_transformed.attrs = dict(attrs)
    da_transformed.attrs["transformation"] = method
    da_transformed.attrs["inverse"] = inverse

    return da_transformed
