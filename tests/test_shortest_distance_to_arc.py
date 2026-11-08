"""
Tests for `shortest_distance_to_arc` with degenerate arcs, see
https://github.com/mllam/mllam-data-prep/issues/113

The arc and query-point cases below are (lat, lon) in degrees.
"""
import numpy as np
import pytest
import spherical_geometry as sg
import xarray as xr

from mllam_data_prep.ops import cropping

# fail if any of the computations emit a numpy RuntimeWarning (e.g. from
# division by zero when normalizing a zero-length vector)
pytestmark = pytest.mark.filterwarnings("error::RuntimeWarning")


def _xyz(lat, lon):
    """(lat, lon) in degrees to (x, y, z) on the unit sphere"""
    return np.array(sg.vector.lonlat_to_vector(lon, lat), dtype=float)


def _great_circle_distance(points, other):
    """great-circle distance (in radians) from each of `points` to `other`"""
    return np.arccos(np.clip(points @ other, -1.0, 1.0))


def _query_points(arc_start_latlon):
    """an arc endpoint, a point on the equator and a point near a pole"""
    return np.stack([_xyz(*arc_start_latlon), _xyz(0.0, 100.0), _xyz(89.9, 0.0)])


def test_identical_endpoints_is_zero_length_arc():
    arc_start = arc_end = (10.0, 20.0)
    points = _query_points(arc_start)

    distances = cropping.shortest_distance_to_arc(
        points, _xyz(*arc_start), _xyz(*arc_end)
    )

    assert np.all(np.isfinite(distances))
    np.testing.assert_allclose(
        distances, _great_circle_distance(points, _xyz(*arc_start)), atol=1e-12
    )
    # an arc endpoint is at zero distance from the arc
    assert distances[0] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize(
    "arc_start, arc_end",
    [
        pytest.param((0.0, 0.0), (0.0, 180.0), id="antipodal-on-equator"),
        pytest.param((90.0, 0.0), (-90.0, 0.0), id="pole-to-pole"),
        pytest.param((30.0, 45.0), (-30.0, -135.0), id="generic-antipodal"),
    ],
)
def test_antipodal_endpoints_raise(arc_start, arc_end):
    # after converting to cartesian coordinates the cross product of these
    # endpoints is not exactly zero (e.g. cos(90deg) ~ 6e-17), so this also
    # checks that a tolerance is used rather than an exact comparison
    with pytest.raises(ValueError, match="antipodal"):
        cropping.shortest_distance_to_arc(
            _query_points(arc_start), _xyz(*arc_start), _xyz(*arc_end)
        )


def test_near_identical_endpoints_is_zero_length_arc():
    arc_start = (10.0, 20.0)
    arc_end = (10.0 + 1e-12, 20.0 + 1e-12)
    points = _query_points(arc_start)

    distances = cropping.shortest_distance_to_arc(
        points, _xyz(*arc_start), _xyz(*arc_end)
    )

    assert np.all(np.isfinite(distances))
    np.testing.assert_allclose(
        distances, _great_circle_distance(points, _xyz(*arc_start)), atol=1e-9
    )


@pytest.mark.parametrize(
    "arc_start, arc_end",
    [
        pytest.param((0.0, 0.0), (0.0, 180.0 - 1e-12), id="near-antipodal-on-equator"),
        pytest.param(
            (30.0, 45.0), (-30.0 + 1e-12, -135.0), id="near-generic-antipodal"
        ),
    ],
)
def test_near_antipodal_endpoints_raise(arc_start, arc_end):
    with pytest.raises(ValueError, match="antipodal"):
        cropping.shortest_distance_to_arc(
            _query_points(arc_start), _xyz(*arc_start), _xyz(*arc_end)
        )


@pytest.mark.parametrize(
    "point",
    [
        pytest.param([0.0, 0.0, 1.0], id="north-pole-of-great-circle"),
        pytest.param([0.0, 0.0, -1.0], id="south-pole-of-great-circle"),
    ],
)
def test_point_at_pole_of_great_circle(point):
    # the projection of a point at the pole of the arc's great circle onto the
    # great-circle plane is the zero vector. Such a point is a quarter circle
    # away from every point of the great circle, and so from the arc as well
    arc_start, arc_end = _xyz(0.0, 0.0), _xyz(0.0, 90.0)

    distances = cropping.shortest_distance_to_arc(np.array([point]), arc_start, arc_end)

    np.testing.assert_allclose(distances, [np.pi / 2.0], atol=1e-12)


def test_regular_arc_distances_unchanged():
    # arc along the equator from lon=0 to lon=90
    arc_start, arc_end = _xyz(0.0, 0.0), _xyz(0.0, 90.0)
    points = np.stack(
        [
            _xyz(10.0, 45.0),  # above the middle of the arc
            _xyz(-20.0, 30.0),  # below the arc
            _xyz(10.0, 120.0),  # beyond the arc end
            _xyz(0.0, 0.0),  # on the arc start
        ]
    )

    distances = cropping.shortest_distance_to_arc(points, arc_start, arc_end)

    expected = [
        np.radians(10.0),
        np.radians(20.0),
        _great_circle_distance(points[2:3], arc_end)[0],
        0.0,
    ]
    np.testing.assert_allclose(distances, expected, atol=1e-12)


def test_distance_to_convex_hull_boundary_has_no_degenerate_arc_warning():
    # the convex hull returned by spherical_geometry is a closed ring (its
    # first point is repeated as the last point), so the arc closing the ring
    # has identical endpoints. This must not produce NaNs or RuntimeWarnings
    lat_ref, lon_ref = np.meshgrid(np.linspace(50, 60, 6), np.linspace(0, 10, 6))
    ds_ref = xr.Dataset(
        coords=dict(
            lat=("grid_index", lat_ref.ravel()), lon=("grid_index", lon_ref.ravel())
        )
    )
    lat, lon = np.meshgrid(np.linspace(40, 70, 16), np.linspace(-10, 20, 16))
    ds = xr.Dataset(
        coords=dict(lat=("grid_index", lat.ravel()), lon=("grid_index", lon.ravel()))
    )

    da_dist = cropping.distance_to_convex_hull_boundary(ds, ds_ref)

    assert da_dist.size > 0
    assert np.all(np.isfinite(da_dist.values))
    assert np.all(da_dist.values > 0.0)
