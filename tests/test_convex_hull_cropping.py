import tempfile
from pathlib import Path

import numpy as np
import pytest

import mllam_data_prep as mdp
import tests.data as testdata
from mllam_data_prep.ops import cropping


def test_create_convex_hull_mask():
    tmpdir = tempfile.TemporaryDirectory()
    domain_size = 500 * 1.0e3  # length and width of domain in meters
    N = 200
    config_lam = testdata.create_input_datasets_and_config(
        identifier="lam",
        data_categories=["state"],
        tmpdir=tmpdir,
        xlim=[-domain_size / 2.0, domain_size / 2.0],
        ylim=[-domain_size / 2.0, domain_size / 2.0],
        nx=N,
        ny=N,
        add_latlon=True,
    )
    # make the global domain twice as large as the LAM domain so that the lam
    # domain is contained within the global domain
    config_global = testdata.create_input_datasets_and_config(
        identifier="global",
        data_categories=["state"],
        tmpdir=tmpdir,
        xlim=[-domain_size, domain_size],
        ylim=[-domain_size, domain_size],
        add_latlon=True,
        nx=N // 2,
        ny=N // 2,
    )

    ds_lam = mdp.create_dataset(config=config_lam)
    ds_global = mdp.create_dataset(config=config_global)

    da_ch_mask, ds_ch_latlons = cropping.create_convex_hull_mask(
        ds=ds_global, ds_reference=ds_lam
    )

    # just check that some of the points make up the convex hull for now
    # (this doesn't check that the convex hull is correct of course...)
    assert 0 < ds_ch_latlons.grid_index_ref.size < ds_lam.grid_index.size

    # Given that the outer domain is 4x larger than the inner domain and the
    # inner domain sits completely within the outer domain, then approximately
    # 1/4 of the outer domain points should be within the convex hull of the
    # inner domain, and 3/4 outside.
    n_inside = da_ch_mask.where(da_ch_mask).count().values
    n_outside = da_ch_mask.where(~da_ch_mask).count().values
    np.testing.assert_almost_equal(n_inside / n_outside, 1.0 / 3.0, decimal=2)

    da_dist = cropping.distance_to_convex_hull_boundary(
        ds=ds_global, ds_reference=ds_lam
    )
    da_dist_unstacked = da_dist.set_index(grid_index=["x", "y"]).unstack("grid_index")

    # check that the distance decreases towards the middle of the domain in
    # both x and y directions
    da_dist_x = da_dist_unstacked.sel(y=0, method="nearest")
    da_dist_change = da_dist_x.diff("x").dropna("x")
    da_dist_x = da_dist_x.sel(x=da_dist_change.x)
    np.testing.assert_array_equal(np.sign(da_dist_change), np.sign(da_dist_x.x))

    da_dist_y = da_dist_unstacked.sel(x=0, method="nearest")
    da_dist_change = da_dist_y.diff("y").dropna("y")
    da_dist_y = da_dist_y.sel(y=da_dist_change.y)
    np.testing.assert_array_equal(np.sign(da_dist_change), np.sign(da_dist_y.y))

    da_convex_hull_margin_crop = cropping.crop_with_convex_hull(
        ds=ds_global,
        ds_reference=ds_lam,
        margin_thickness=2.0,
        include_interior_points=False,
    )

    # check that there are fewer points in this margin region
    n_points_margin_region = da_convex_hull_margin_crop.count()
    assert n_points_margin_region < n_outside


@pytest.mark.parametrize("include_interior_points", [True, False])
def test_create_cropped_dataset(include_interior_points):

    tmpdir = tempfile.TemporaryDirectory()
    d_len = 500 * 1.0e3  # length and width of domain in meters
    N = 50  # number of grid points in each direction
    config_lam = testdata.create_input_datasets_and_config(
        identifier="lam",
        data_categories=["state"],
        tmpdir=tmpdir,
        xlim=[-d_len / 2.0, d_len / 2.0],
        ylim=[-d_len / 2.0, d_len / 2.0],
        nx=N,
        ny=N,
        add_latlon=True,
    )
    # make the global domain twice as large as the LAM domain so that the lam
    # domain is contained within the global domain, but half the number of grid
    # points in each direction
    config_global = testdata.create_input_datasets_and_config(
        identifier="global",
        data_categories=["state"],
        tmpdir=tmpdir,
        xlim=[-d_len, d_len],
        ylim=[-d_len, d_len],
        nx=N // 2,
        ny=N // 2,
        add_latlon=True,
    )

    fp_lam_config = str(Path(tmpdir.name) / "lam_config.yml")
    config_lam.to_yaml_file(fp_lam_config)

    config_global.output.domain_cropping = mdp.config.ConvexHullCropping(
        margin_width_degrees=2.0,
        include_interior_points=include_interior_points,
        interior_dataset_config_path=fp_lam_config,
    )
    mdp.create_dataset(config=config_global)
