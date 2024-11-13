"""Collection of operators handling projection information."""

from typing import Any, Dict, List, Tuple, Union

import pyproj
import xarray as xr
from pyproj import CRS


class ProjectionInconsistencyWarning(Warning):
    pass


def _extract_grid_mapping_names(grid_mapping_attr: str) -> List[str]:
    """Extract grid mapping names from the grid_mapping attribute.

    Parameters
    ----------
    grid_mapping_attr : str
        The grid_mapping attribute.

    Returns
    -------
    list
        List of grid mapping names.

    Examples
    --------
    >>> m1 = "crsOSGB: x y crsWGS84: lat lon"
    >>> _extract_grid_mapping_names(m1)
    ['crsOSGB', 'crsWGS84']
    >>> m2 = "crsOSGB"
    >>> _extract_grid_mapping_names(m2)
    ['crsOSGB']

    Note: merge of https://github.com/pydata/xarray/pull/9765 of will allow xarray to handle this directly.
    """
    if ":" not in grid_mapping_attr:
        return grid_mapping_attr.split(" ")
    else:
        return [
            key.split(":")[0].strip()
            for key in grid_mapping_attr.split(" ")
            if ":" in key
        ]


def _get_projection_mappings(dataset: xr.Dataset) -> Dict[str, Any]:
    """Get projections referenced by variables.

    This function extracts the projection variables from a dataset
    by evaluating the grid_mapping attribute of each variable following
    the CF-Conventions.

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset to get the projection information from

    Returns
    -------
    Dict[str, str]
        Dictionary with the variable names as keys and the projection
        variable names as values
    """
    projections = {}
    for var in dataset.variables:
        if "grid_mapping" in dataset[var].attrs:
            gm = _extract_grid_mapping_names(dataset[var].attrs["grid_mapping"])
            projections[var] = gm
    return projections


def validate_projection_consistency(
    projections: List[Dict[str, Union[str, dict]]]
) -> None:
    """Validate the consistency of the projection information.

    Examples
    --------
    >>> crs_wkt = (
    ...     'GEOGCRS["WGS 84",ENSEMBLE["World Geodetic System 1984 ensemble",'
    ...     'MEMBER["World Geodetic System 1984 (Transit)"],'
    ...     'MEMBER["World Geodetic System 1984 (G730)"],'
    ...     'MEMBER["World Geodetic System 1984 (G873)"],'
    ...     'MEMBER["World Geodetic System 1984 (G1150)"],'
    ...     'MEMBER["World Geodetic System 1984 (G1674)"],'
    ...     'MEMBER["World Geodetic System 1984 (G1762)"],'
    ...     'MEMBER["World Geodetic System 1984 (G2139)"],'
    ...     'MEMBER["World Geodetic System 1984 (G2296)"],'
    ...     'ELLIPSOID["WGS 84",6378137,298.257223563,'
    ...     'LENGTHUNIT["metre",1]],ENSEMBLEACCURACY[2.0]],'
    ...     'PRIMEM["Greenwich",0,ANGLEUNIT["degree",0.0174532925199433]],'
    ...     'CS[ellipsoidal,2],AXIS["geodetic latitude (Lat)",north,'
    ...     'ORDER[1],ANGLEUNIT["degree",0.0174532925199433]],'
    ...     'AXIS["geodetic longitude (Lon)",east,ORDER[2],'
    ...     'ANGLEUNIT["degree",0.0174532925199433]],'
    ...     'USAGE[SCOPE["Horizontal component of 3D system."],'
    ...     'AREA["World."],BBOX[-90,-180,90,180]],ID["EPSG",4326]]'
    ... )
    >>> proj1 = {'crs_wkt': crs_wkt}
    >>> proj2 = {"crs_wkt": "EPSG:4326"}
    >>> proj3 = {
    ...     "crs_wkt": crs_wkt,
    ...     'semi_major_axis': 6378137.0,
    ...     'semi_minor_axis': 6356752.314245179,
    ...     'inverse_flattening': 298.257223563,
    ...     'reference_ellipsoid_name': 'WGS 84',
    ...     'longitude_of_prime_meridian': 0.0,
    ...     'prime_meridian_name': 'Greenwich',
    ...     'geographic_crs_name': 'WGS 84',
    ...     'horizontal_datum_name': 'World Geodetic System 1984 ensemble',
    ...     'grid_mapping_name': 'latitude_longitude'
    ... }
    >>> proj4 = {"crs_wkt": "AGD84"}
    >>> validate_projection_consistency([proj1, proj2, proj3])
    Traceback (most recent call last):
        ...
    mllam_data_prep.ops.projection.ProjectionInconsistencyWarning: ["Key 'semi_major_axis' is missing ...
    >>> validate_projection_consistency([proj2, proj4])
    Traceback (most recent call last):
        ...
    ValueError: Multiple projections found in the dataset.Currently only one projection is supported.
    """
    proj_obs = [pyproj.CRS.from_cf(proj) for proj in projections]

    # Check that all projections are the same
    if len(set(proj_obs)) > 1:
        raise ValueError(
            "Multiple projections found in the dataset."
            "Currently only one projection is supported."
        )

    # Check that conversion to CF is consistent with input
    # and all projection information is given
    cf_output = set(proj_obs).pop().to_cf()
    issues = []
    for proj in projections:
        cf_wkt_set = "crs_wkt" in cf_output
        cf_indv_attrs_set = set(cf_output.keys()) - {"crs_wkt"}
        for key, value in cf_output.items():
            if key == "crs_wkt" and cf_wkt_set and value != proj["crs_wkt"]:
                issues.append(f"'{key}' differs: {proj[key]} != {value}")
            elif key != "crs_wkt" and cf_indv_attrs_set and key not in proj:
                issues.append(f"Key '{key}' is missing in the projection information.")
            elif key == "crs_wkt" and cf_indv_attrs_set and value != proj[key]:
                issues.append(f"Value for key '{key}' differs: {proj[key]} != {value}")
        for key in proj:
            if key not in cf_output:
                issues.append(
                    f"Key '{key}' is not expected in the projection information."
                )
    if issues:
        raise ProjectionInconsistencyWarning(issues)

    return


def get_projection_crs(ds: xr.Dataset) -> Dict[str, Any]:
    """Get the projection information from the dataset.

    Parameters
    ----------
    ds : xr.Dataset
        CF-conform dataset to extract projection information from.
        `grid_mapping` attribute must be set for each variable that
        references a projection variable.
        Projection variables must have either a `crs_wkt` attribute
        or CF-conform single-value attributes with individual projection
        information.

    Returns
    -------
    Dict[str, Any]
        Projection information as a dictionary with the projection
        variable names as keys and the projection objects as values
    """
    vars_w_proj = _get_projection_mappings(ds)
    proj_vars = set([proj for proj in vars_w_proj.values()])
    vars_wo_proj = set(ds.data_vars) - set(vars_w_proj.keys()) - proj_vars

    if len(proj_vars) > 1:
        raise ValueError(
            f"Multiple referenced projections found in the dataset {ds.encoding.get('source',None)}: {proj_vars}. "
            "Currently only one projection is supported."
        )

    if len(vars_wo_proj) > 0:
        raise ValueError(
            f"Variables {vars_wo_proj} do not have a projection defined in the dataset {ds.encoding.get('source',None)}"
        )

    crss = {}
    for crs in crss:
        crss[crs] = ds[crs].attrs  # pyproj.CRS.from_cf(ds[proj].attrs)

    return crss


def get_latitude_longitude_from_projection(
    proj: CRS, coords: Tuple[Any, Any], output_proj: CRS = None
) -> Tuple[Any, Any]:
    """Get the latitude and longitude from a projection object.

    Parameters
    ----------
    proj : CRS
        Projection object.
    coords : Tuple[array-like, array-like]
        Coordinates to convert.

    Returns
    -------
    Tuple[array-like, array-like]
        Latitude and longitude in degrees.

    Examples
    --------
    >>> proj = pyproj.CRS.from_cf({"crs_wkt":"EPSG:3857"})
    >>> coords = (500000, 4649776)
    >>> get_latitude_longitude_from_projection(proj, coords)
    (38.496303121059576, 4.491576420597607)
    >>> coords = ([400000, 500000], [3649776, 4649776])
    >>> get_latitude_longitude_from_projection(proj, coords)
    ([31.13101769562677, 38.496303121059576], [3.5932611364780858, 4.491576420597607])
    """
    if output_proj is None:
        output_proj = pyproj.CRS("EPSG:4326")

    transformer = pyproj.Transformer.from_crs(proj, output_proj)
    return transformer.transform(coords[0], coords[1])
