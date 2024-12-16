import tempfile

import numpy as np

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

    da_ch_mask = cropping.create_convex_hull_mask(ds=ds_global, ds_reference=ds_lam)

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

    da_convex_hull_margin_crop = cropping.crop_to_within_convex_hull_margin(
        ds=ds_global, ds_reference=ds_lam, max_dist=2.0
    )

    # check that there are fewer points in this margin region
    n_points_margin_region = da_convex_hull_margin_crop.count()
    assert n_points_margin_region < n_outside
