from typing import Tuple, Union

import numpy as np
import spherical_geometry as sg
import xarray as xr
from spherical_geometry.polygon import SphericalPolygon


def _get_latlon_coords(da: xr.DataArray) -> tuple:
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
        return (da.longitude, da.latitude)
    elif "lat" in da.coords and "lon" in da.coords:
        return (da.lon, da.lat)
    else:
        raise Exception("Could not find lat/lon coordinates in DataArray.")


def create_convex_hull_mask(ds: xr.Dataset, ds_reference: xr.Dataset) -> xr.DataArray:
    """
    Create a grid-point mask for lat/lon coordinates in `da` indicating which
    points are interior to the convex hull of the lat/lon coordinates of
    `da_ref`.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset for which to create the mask.
    ds_reference : xarray.Dataset
        The reference dataset from which to create the convex hull of the coordinates.

    Returns
    -------
    xarray.DataArray
        A boolean mask indicating which points in `ds_reference` are interior
        to the convex hull.
    xarray.Dataset
        A dataset containing lat lon coordinates for points in `ds` making up
        the convex hull.
    """
    da_lon, da_lat = _get_latlon_coords(ds)
    da_lon_ref, da_lat_ref = _get_latlon_coords(ds_reference)

    assert da_lat.dims == da_lon.dims
    assert da_lat_ref.dims == da_lon_ref.dims

    # latlon to (x, y, z) on unit sphere
    da_ref_xyz = _latlon_to_unit_sphere_xyz(da_lat=da_lat_ref, da_lon=da_lon_ref)

    chull_lam = SphericalPolygon.convex_hull(da_ref_xyz.values)

    # call .load() to avoid using dask arrays in the following apply_ufunc
    da_interior_mask = xr.apply_ufunc(
        chull_lam.contains_lonlat, da_lon.load(), da_lat.load(), vectorize=True
    ).astype(bool)
    da_interior_mask.attrs[
        "long_name"
    ] = "contained in convex hull of source dataset (da_ref)"

    # Get points at edge of convex hull
    chull_lam_lon, chull_lam_lat = list(chull_lam.to_lonlat())[0]
    chull_lat_lons = xr.Dataset(
        coords={
            "lon": (["grid_index_ref"], chull_lam_lon),
            "lat": (["grid_index_ref"], chull_lam_lat),
        }
    )

    return da_interior_mask, chull_lat_lons


def _latlon_to_unit_sphere_xyz(
    da_lat: xr.DataArray, da_lon: xr.DataArray
) -> xr.DataArray:
    """
    Convert lat/lon coordinates to (x, y, z) on the unit sphere.

    Parameters
    ----------
    da_lat : xarray.DataArray
        Latitude coordinates.
    da_lon : xarray.DataArray
        Longitude coordinates.

    Returns
    -------
    xr.DataArray
        The (x, y, z) coordinates on the unit sphere as an xarray.DataArray
        with dimensions (grid_index, component).
    """
    pts_xyz = np.array(sg.vector.lonlat_to_vector(da_lon, da_lat)).T
    da_xyz = xr.DataArray(
        pts_xyz, coords=da_lat.coords, dims=list(da_lat.dims) + ["xyz"]
    )
    return da_xyz


def shortest_distance_to_arc(
    point_cartesian: np.ndarray,
    arc_start_cartesian: np.ndarray,
    arc_end_cartesian: np.ndarray,
) -> np.ndarray:
    """
    Compute shortest haversine distance from a set of points to an arc on the
    surface of the sphere. All points are assumed to be on the surface of a
    sphere of the same radius (e.g. the unit sphere) given in Cartesian (x, y,
    z) coordinates.

    Parameters
    ----------
    point_cartesian : np.ndarray, shape (num_points, 3)
        Points to measure distance from
    arc_start_cartesian : np.ndarray, shape (3,)
        Start point of arc
    arc_end_cartesian : np.ndarray, shape (3,)
        End point of arc

    Returns
    -------
    np.ndarray, shape (num_points,)
        The distances in radians
    """
    # Calculate normal vector to the plane of the great circle
    normal_vector = np.cross(arc_start_cartesian, arc_end_cartesian)
    normal_vector = normal_vector / np.linalg.norm(normal_vector)  # Normalize

    # Project point onto the plane
    point_projection = (
        point_cartesian
        - np.dot(point_cartesian, normal_vector)[:, np.newaxis] * normal_vector
    )

    # Normalize to get the projected point on the sphere's surface
    projected_point = (
        point_projection / np.linalg.norm(point_projection, axis=1)[:, np.newaxis]
    )

    # Calculate the angle between the original point and the projected point
    angle_point_to_projection = np.arccos(
        np.clip(np.sum(point_cartesian * projected_point, axis=1), -1, 1)
    )

    # Check if the projected point is between the start and end points of the arc
    is_between_arc = (
        np.dot(np.cross(arc_start_cartesian, projected_point), normal_vector) >= 0
    ) & (np.dot(np.cross(projected_point, arc_end_cartesian), normal_vector) >= 0)

    # Calculate distances from the point to the start and end points of the arc
    distance_to_start = np.arccos(
        np.clip(np.dot(point_cartesian, arc_start_cartesian), -1, 1)
    )
    distance_to_end = np.arccos(
        np.clip(np.dot(point_cartesian, arc_end_cartesian), -1, 1)
    )

    # Choose the appropriate distance
    distances = np.where(
        is_between_arc,
        angle_point_to_projection,
        np.minimum(distance_to_start, distance_to_end),
    )

    # Distance returned in radians
    return distances


def distance_to_convex_hull_boundary(
    ds: xr.Dataset,
    ds_reference: xr.Dataset,
    grid_index_dim: str = "grid_index",
    include_convex_hull_mask: bool = False,
) -> Union[xr.DataArray, Tuple[xr.DataArray, xr.DataArray]]:
    """
    For all points in `ds` that are external to the convex hull of the points in
    `ds_reference`, calculate the minimum distance to the convex hull boundary.

    The method goes through the following steps:
    1. Create a mask for the points in `ds` from the convex hull of the points
       in `ds_reference`.
    2. Find the points in `ds` that are external to the convex hull.
    3. For each point in `ds` external to the convex hull, calculate the
       minimum distance to the convex hull boundary. The distance is calculated
       as the shortest distance to any of the arcs making up the convex hull
       boundary.


    Parameters
    ----------
    ds : xarray.Dataset
        The dataset for which to calculate the distance.
    ds_reference : xarray.Dataset
        The reference dataset from which to calculate the convex hull boundary.
    grid_index_dim : str, optional
        The name of the grid index dimension in `ds` and `ds_reference`.
    include_convex_hull_mask : bool, optional
        Whether to include the convex hull mask in the output.

    Returns
    -------
    da_mindist_to_ref : xarray.DataArray
        The minimum distance to the convex hull boundary.
    da_ch_mask : xarray.DataArray
        The convex hull mask (only if `include_convex_hull_mask` is True).

    """
    # rename the grid index dimension in ds_reference to avoid conflicts (since
    # the grid index dimension otherwise has the same name in both datasets,
    # and later we will want to find the minimum distance for each point in ds
    # to all points in ds_reference)
    ds_reference_separate_gridindex = ds_reference.rename(
        {grid_index_dim: "grid_index_ref"}
    )

    # create a mask from the convex hull of ds_reference for the grid points in ds
    da_ch_mask, ds_chull_lat_lons = create_convex_hull_mask(
        ds=ds, ds_reference=ds_reference_separate_gridindex
    )

    # only consider points that are external to the convex hull
    ds_exterior = ds.where(~da_ch_mask, drop=True)
    ds_exterior_lon, ds_exterior_lat = _get_latlon_coords(ds_exterior)

    da_xyz = _latlon_to_unit_sphere_xyz(ds_exterior_lon, ds_exterior_lat)

    da_xyz_chull = _latlon_to_unit_sphere_xyz(*_get_latlon_coords(ds_chull_lat_lons))

    # Collect arcs making up chull
    chull_arcs = list(zip(da_xyz_chull[:-1], da_xyz_chull[1:])) + [
        (da_xyz_chull[-1], da_xyz_chull[0])
    ]  # Add arc from last to first point

    # Calculate minimum distance to each arc and take the minimum
    # distance over all arcs
    mindist_to_ref = np.stack(
        [
            shortest_distance_to_arc(da_xyz, arc_start, arc_end)
            for arc_start, arc_end in chull_arcs
        ],
        axis=0,
    ).min(axis=0)

    da_mindist_to_ref = xr.DataArray(
        mindist_to_ref, coords=ds_exterior_lat.coords, dims=ds_exterior_lat.dims
    )
    da_mindist_to_ref.attrs[
        "long_name"
    ] = "minimum distance to convex hull boundary of reference dataset"
    da_mindist_to_ref.attrs["units"] = "radians"

    if include_convex_hull_mask:
        return da_mindist_to_ref, da_ch_mask

    return da_mindist_to_ref


def _mask_with_common_dim(da_mask, ds):
    """
    Apply mask to all variables in `ds` that share dimension(s) with `da_mask`.

    Parameters
    ----------
    da_mask : xarray.DataArray
        The mask.
    ds : xarray.Dataset
        The dataset to mask.

    Returns
    -------
    xarray.Dataset
        The masked dataset including the variables that don't share
        dimension(s) with the mask (these are simply copied over).
    """
    mask_dims = da_mask.dims
    vars_with_dims = [
        v for v in ds.data_vars if all(d in ds[v].dims for d in mask_dims)
    ]
    vars_without_dims = [v for v in ds.data_vars if v not in vars_with_dims]

    ds_masked = ds.drop_vars(vars_without_dims).where(da_mask, drop=True)
    ds_masked = xr.merge([ds_masked, ds[vars_without_dims]])
    return ds_masked


def crop_with_convex_hull(
    ds: xr.Dataset,
    ds_reference: xr.Dataset,
    grid_index_dim: str = "grid_index",
    margin_thickness: float = 2.0,
    include_interior_points: bool = True,
    return_mask=False,
) -> xr.Dataset:
    """
    Crop grid points (with coordinates given in lat/lon) in `ds` that are
    within a certain distance (within the margin of a given width) of the
    convex hull boundary of the points in `ds_reference`. The margin is
    measured in degrees.

    ┌──────────────────────────────────────┐
    │ Margin                               │
    │     ┌────── Convex hull ───────┐     │
    │     │                          │     │
    │     │ included if              │     │
    │     │ include_interior == True │     │
    │     │                          │     │
    │<--->│                          │     │
    │  :  └──────────────────────────┘     │
    │  :... margin width                   │
    └──────────────────────────────────────┘

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset to crop.
    ds_reference : xarray.Dataset
        The reference dataset from which to calculate the convex hull boundary.
    grid_index_dim : str, optional
        The name of the grid index dimension in `ds` and `ds_reference`.
    margin_thickness : float, optional
        The thickness of the margin to apply to the convex hull boundary in
        degrees. Points within this margin will be included in the output.
    """
    if margin_thickness == 0.0:
        if not include_interior_points:
            raise Exception(
                "With no margin and exclude_interior=False, all points would be excluded."
            )
        da_mask = create_convex_hull_mask(ds=ds, ds_reference=ds_reference)
    else:
        da_min_dist_to_ref, da_ch_mask = distance_to_convex_hull_boundary(
            ds,
            ds_reference,
            grid_index_dim=grid_index_dim,
            include_convex_hull_mask=True,
        )

        max_dist_radians = margin_thickness * np.pi / 180.0
        da_boundary_mask = da_min_dist_to_ref < max_dist_radians

        if not include_interior_points:
            da_mask = da_boundary_mask
        else:
            da_interior_points = da_ch_mask.where(da_ch_mask, drop=True)
            da_mask = xr.concat(
                [da_interior_points, da_boundary_mask], dim="grid_index"
            )

    # it is unclear if there is a bug in xr.Dataset.where(), but its default
    # behaviour seems to be broadcast (i.e. add) the dimensions of the mask to
    # any data variables that don't have those dimensions already. We only want
    # to mask the variables that share dimension(s) with the mask (i.e. have
    # the `grid_index` dimension), so we drop the other variables before
    # applying the mask.
    ds_cropped = _mask_with_common_dim(da_mask=da_mask, ds=ds)

    if return_mask:
        return ds_cropped, da_mask

    return ds_cropped
