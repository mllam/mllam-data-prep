import copy
import tempfile
from pathlib import Path

import numpy as np
import pytest

import mllam_data_prep as mdp
import mllam_data_prep.config as mdp_config
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


def test_crop_era5_with_generated_lam_domain():
    """
    Test cropping an ERA5 dataset with a generated LAM domain to ensure that
    coordinates of ERA5 domain have been carried over into resulting dataset,
    i.e. the same variables should be present whether the domain is cropped or
    not.
    """

    era5_config = mdp_config.Config(
        schema_version="v0.5.0",
        dataset_version="v1.0.0",
        output=mdp_config.Output(
            variables=dict(
                state=["time", "grid_index", "state_feature"],
            ),
            coord_ranges=dict(
                time=mdp_config.Range(
                    start="1990-09-03T00:00", end="1990-09-09T00:00", step="PT6H"
                )
            ),
            chunking=dict(time=1),
        ),
        inputs={
            "era_height_levels": mdp_config.InputDataset(
                path="simplecache::gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr",
                dims=["time", "longitude", "latitude", "level"],
                variables={
                    "u_component_of_wind": dict(
                        level=mdp_config.ValueSelection(values=[1000], units="hPa")
                    )
                },
                dim_mapping={
                    "time": mdp_config.DimMapping(method="rename", dim="time"),
                    "state_feature": mdp_config.DimMapping(
                        method="stack_variables_by_var_name",
                        dims=["level"],
                        name_format="{var_name}{level}hPa",
                    ),
                    "grid_index": mdp_config.DimMapping(
                        method="stack", dims=["longitude", "latitude"]
                    ),
                },
                target_output_variable="state",
            )
        },
    )

    # create uncropped dataset
    ds_uncropped = mdp.create_dataset(config=era5_config)

    tmpdir = tempfile.TemporaryDirectory()

    lam_config = testdata.create_input_datasets_and_config(
        identifier="danra_lam",
        data_categories=["state"],
        tmpdir=tmpdir,
        xlim=[-2.0, 2.0],
        ylim=[-2.0, 2.0],
        nx=50,
        ny=50,
        add_latlon=True,
    )
    # save the LAM config to a temporary file
    lam_config_path = Path(tmpdir.name) / "danra_lam_config.yaml"
    lam_config.to_yaml_file(lam_config_path)

    # create cropped dataset
    era5_config_cropped = copy.deepcopy(era5_config)
    era5_config_cropped.output.domain_cropping = mdp_config.ConvexHullCropping(
        margin_width_degrees=10,
        interior_dataset_config_path=lam_config_path.as_posix(),
    )

    ds_cropped = mdp.create_dataset(config=era5_config_cropped)

    # check that the cropped dataset has the same variables and coordinates as
    # the uncropped one, allowing for fewer grid points
    for var in ds_uncropped.data_vars:
        assert var in ds_cropped.data_vars
        for coord in ds_uncropped[var].coords:
            assert coord in ds_cropped[var].coords
