import numpy as np
import pyproj
import spherical_geometry as sg
import xarray as xr
from loguru import logger
from spherical_geometry.polygon import SphericalPolygon


def _get_latlon_coords(da):
    """
    Get the latlon coordinates of a DataArray.

    Parameters
    ----------
    da : xarray.DataArray
        The data.

    Returns
    -------
    tuple (xarray.DataArray, xarray.DataArray)
        The latitude and longitude coordinates.
    """
    if "latitude" in da.coords and "longitude" in da.coords:
        return (da.latitude, da.longitude)
    elif "lat" in da.coords and "lon" in da.coords:
        return (da.lat, da.lon)
    else:
        raise Exception()


def _add_latlon(ds):
    logger.warning(
        "Could not find lat/lon coordinates. Assuming equal area projection "
        "centered on (lat, lon) = (0, 0) for now. This should be replaced!"
    )

    # create a local equal area projection centered on Denmark (Høje Taastrup)
    # for now
    da_x = ds.coords["x"]
    da_y = ds.coords["y"]
    proj = pyproj.Proj(proj="laea", lon_0=12.25, lat_0=55.65)
    lon, lat = proj(da_x.values, da_y.values, inverse=True)
    da_lat = xr.DataArray(lat, coords=da_x.coords, dims=da_x.dims)
    da_lon = xr.DataArray(lon, coords=da_y.coords, dims=da_y.dims)

    ds.coords["lon"] = da_lon
    ds.coords["lat"] = da_lat

    return ds


def create_convex_hull_mask(ds, ds_reference):
    """
    Create a grid-point mask for lat/lon coordinates in `da` indicating which
    points are interior to the convex hull of the lat/lon coordinates of
    `da_ref`.
    """
    # TODO: this should be replaced with a function to get the true lat/lon coordinates
    _add_latlon(ds)
    _add_latlon(ds_reference)

    da_lat, da_lon = _get_latlon_coords(ds)
    da_lat_ref, da_lon_ref = _get_latlon_coords(ds_reference)

    assert da_lat.dims == da_lon.dims == da_lat_ref.dims == da_lon_ref.dims

    # latlon to (x, y, z) on unit sphere
    pts_ref_xyz = np.array(sg.vector.lonlat_to_vector(da_lon_ref, da_lat_ref)).T

    chull_lam = SphericalPolygon.convex_hull(pts_ref_xyz)

    da_interior_mask = xr.apply_ufunc(
        chull_lam.contains_lonlat, da_lon, da_lat, vectorize=True
    )
    da_interior_mask.attrs[
        "long_name"
    ] = "contained in convex hull of source dataset (da_ref)"

    return da_interior_mask


def boundary_crop_gridpoints_with_latlon_convex_hull(
    ds, ds_reference, grid_index_dim="grid_index", max_dist=0.2
):
    raise NotImplementedError()
    # da_interior_mask = create_convex_hull_mask(ds=ds, ds_reference=ds_reference)

    # points = _get_latlon_coords(ds)
    # points_lam = _get_latlon_coords(ds_reference)

    # latlon to (x, y, z) on unit sphere
    # pts_xyz = np.array(sg.vector.lonlat_to_vector(points[:, 0], points[:, 1])).T
    # pts_lam_xyz = np.array(
    #     sg.vector.lonlat_to_vector(points_lam[:, 0], points_lam[:, 1])
    # ).T

    # da_xyz = xr.DataArray(pts_xyz, coords=ds.coords, dims=[grid_index_dim, "xyz"])

    # chull_lam = SphericalPolygon.convex_hull(pts_lam_xyz)

    # # Finding the boundary points inside the convec hull is what takes time. Possibly speed up by restricting to a subset of points_ that are within some distance from area center or similar?
    # msk_in = np.array([chull_lam.contains_lonlat(x[0], x[1]) for x in points])
    # da_interior_mask = xr.DataArray(msk_in, coords=ds.coords, dims=[grid_index_dim])
    # ds_reference = ds.where(da_interior_mask, drop=True)

    # Distances from all boundary points outside the region to the closest point inside
    # d = np.array(
    #     [np.min(np.arccos(np.dot(pts_lam_xyz, pt_xyz))) for pt_xyz in pts_xyz[~msk_in]]
    # )

    # Define boundary distance and mask boundary points outside that are closer
    # This is now in radians but should be defined as meters based on Earth radius
    # ii = d < max_dist
