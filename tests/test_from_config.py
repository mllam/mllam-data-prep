import tempfile
from pathlib import Path
from typing import Dict

import isodate
import pyproj
import pytest
import yaml

import mllam_data_prep as mdp
import tests.data as testdata
import tests.test_config as testconfig


def test_gen_data():
    tmpdir = tempfile.TemporaryDirectory()
    testdata.create_data_collection(
        data_kinds=testdata.ALL_DATA_KINDS, fp_root=tmpdir.name
    )


def test_merging_static_and_surface_analysis():
    tmpdir = tempfile.TemporaryDirectory()
    datasets = testdata.create_data_collection(
        data_kinds=["surface_analysis", "static"], fp_root=tmpdir.name
    )

    # use 80% for training and 20% for testing
    t_train_start = testdata.T_START
    t_train_end = testdata.T_START + 0.8 * (testdata.T_END_ANALYSIS - testdata.T_START)
    t_test_start = t_train_end + testdata.DT_ANALYSIS
    t_test_end = testdata.T_END_ANALYSIS

    config = dict(
        schema_version="v0.2.0",
        dataset_version="v0.1.0",
        output=dict(
            variables=dict(
                static=["grid_index", "static_feature"],
                state=["time", "grid_index", "state_feature"],
                forcing=["time", "grid_index", "forcing_feature"],
            ),
            coord_ranges=dict(
                time=dict(
                    start=testdata.T_START.isoformat(),
                    end=testdata.T_END_ANALYSIS.isoformat(),
                    step=isodate.duration_isoformat(testdata.DT_ANALYSIS),
                )
            ),
            splitting=dict(
                dim="time",
                splits=dict(
                    train=dict(
                        start=t_train_start.isoformat(),
                        end=t_train_end.isoformat(),
                        compute_statistics=dict(
                            ops=["mean", "std"],
                            dims=["time", "grid_index"],
                        ),
                    ),
                    test=dict(
                        start=t_test_start.isoformat(),
                        end=t_test_end.isoformat(),
                    ),
                ),
            ),
        ),
        inputs=dict(
            danra_surface=dict(
                path=datasets["surface_analysis"],
                dims=["analysis_time", "x", "y"],
                variables=testdata.DEFAULT_SURFACE_ANALYSIS_VARS,
                dim_mapping=dict(
                    time=dict(
                        method="rename",
                        dim="analysis_time",
                    ),
                    grid_index=dict(
                        method="stack",
                        dims=["x", "y"],
                    ),
                    forcing_feature=dict(
                        method="stack_variables_by_var_name",
                        name_format="{var_name}",
                    ),
                ),
                target_output_variable="forcing",
            ),
            danra_static=dict(
                path=datasets["static"],
                dims=["x", "y"],
                variables=testdata.DEFAULT_STATIC_VARS,
                dim_mapping=dict(
                    grid_index=dict(
                        method="stack",
                        dims=["x", "y"],
                    ),
                    static_feature=dict(
                        method="stack_variables_by_var_name",
                        name_format="{var_name}",
                    ),
                ),
                target_output_variable="static",
            ),
        ),
    )

    # write yaml config to file
    fn_config = "config.yaml"
    fp_config = Path(tmpdir.name) / fn_config
    with open(fp_config, "w") as f:
        yaml.dump(config, f)

    mdp.create_dataset_zarr(fp_config=fp_config)


@pytest.mark.parametrize("source_data_contains_time_range", [True, False])
@pytest.mark.parametrize(
    "time_stepsize",
    [testdata.DT_ANALYSIS, testdata.DT_ANALYSIS * 2, testdata.DT_ANALYSIS / 2],
)
def test_time_selection(source_data_contains_time_range, time_stepsize):
    """
    Check that time selection works as expected, so that when source
    data doesn't contain the time range specified in the config and exception
    is raised, and otherwise that the correct timesteps are in the output
    """

    tmpdir = tempfile.TemporaryDirectory()
    datasets = testdata.create_data_collection(
        data_kinds=["surface_analysis", "static"], fp_root=tmpdir.name
    )

    t_start_dataset = testdata.T_START
    t_end_dataset = t_start_dataset + (testdata.NT_ANALYSIS - 1) * testdata.DT_ANALYSIS

    if source_data_contains_time_range:
        t_start_config = t_start_dataset
        t_end_config = t_end_dataset
    else:
        t_start_config = t_start_dataset - testdata.DT_ANALYSIS
        t_end_config = t_end_dataset + testdata.DT_ANALYSIS

    config = dict(
        schema_version="v0.2.0",
        dataset_version="v0.1.0",
        output=dict(
            variables=dict(
                static=["grid_index", "feature"],
                state=["time", "grid_index", "feature"],
                forcing=["time", "grid_index", "feature"],
            ),
            coord_ranges=dict(
                time=dict(
                    start=t_start_config.isoformat(),
                    end=t_end_config.isoformat(),
                    step=isodate.duration_isoformat(time_stepsize),
                )
            ),
        ),
        inputs=dict(
            danra_surface=dict(
                path=datasets["surface_analysis"],
                dims=["analysis_time", "x", "y"],
                variables=testdata.DEFAULT_SURFACE_ANALYSIS_VARS,
                dim_mapping=dict(
                    time=dict(
                        method="rename",
                        dim="analysis_time",
                    ),
                    grid_index=dict(
                        method="stack",
                        dims=["x", "y"],
                    ),
                    feature=dict(
                        method="stack_variables_by_var_name",
                        name_format="{var_name}",
                    ),
                ),
                target_output_variable="forcing",
            ),
        ),
    )

    # write yaml config to file
    fn_config = "config.yaml"
    fp_config = Path(tmpdir.name) / fn_config
    with open(fp_config, "w") as f:
        yaml.dump(config, f)

    # run the main function
    if source_data_contains_time_range and time_stepsize == testdata.DT_ANALYSIS:
        mdp.create_dataset_zarr(fp_config=fp_config)
    else:
        print(
            f"Expecting ValueError for source_data_contains_time_range={source_data_contains_time_range} and time_stepsize={time_stepsize}"
        )
        with pytest.raises(ValueError):
            mdp.create_dataset_zarr(fp_config=fp_config)


@pytest.mark.parametrize("use_common_feature_var_name", [True, False])
def test_feature_collision(use_common_feature_var_name):
    """
    Use to arch target_output_variable variables which have a different number of features and
    therefore need a unique feature dimension for each target_output_variable. This should raise
    a ValueError if the feature coordinates have the same name
    """
    tmpdir = tempfile.TemporaryDirectory()
    datasets = testdata.create_data_collection(
        data_kinds=["surface_analysis", "static"], fp_root=tmpdir.name
    )

    if use_common_feature_var_name:
        static_feature_var_name = state_feature_var_name = "feature"
    else:
        static_feature_var_name = "static_feature"
        state_feature_var_name = "state_feature"

    config = dict(
        schema_version="v0.2.0",
        dataset_version="v0.1.0",
        output=dict(
            variables=dict(
                static=["grid_index", static_feature_var_name],
                state=["time", "grid_index", state_feature_var_name],
            ),
        ),
        inputs=dict(
            danra_surface=dict(
                path=datasets["surface_analysis"],
                dims=["analysis_time", "x", "y"],
                variables=testdata.DEFAULT_SURFACE_ANALYSIS_VARS,
                dim_mapping={
                    "time": dict(
                        method="rename",
                        dim="analysis_time",
                    ),
                    "grid_index": dict(
                        method="stack",
                        dims=["x", "y"],
                    ),
                    state_feature_var_name: dict(
                        method="stack_variables_by_var_name",
                        name_format="{var_name}",
                    ),
                },
                target_output_variable="state",
            ),
            danra_static=dict(
                path=datasets["static"],
                dims=["x", "y"],
                variables=testdata.DEFAULT_STATIC_VARS,
                dim_mapping={
                    "grid_index": dict(
                        dims=["x", "y"],
                        method="stack",
                    ),
                    static_feature_var_name: dict(
                        method="stack_variables_by_var_name",
                        name_format="{var_name}",
                    ),
                },
                target_output_variable="static",
            ),
        ),
    )

    # write yaml config to file
    fn_config = "config.yaml"
    fp_config = Path(tmpdir.name) / fn_config
    with open(fp_config, "w") as f:
        yaml.dump(config, f)

    if use_common_feature_var_name:
        with pytest.raises(mdp.InvalidConfigException):
            mdp.create_dataset_zarr(fp_config=fp_config)
    else:
        mdp.create_dataset_zarr(fp_config=fp_config)


@pytest.mark.parametrize(
    "projection",
    [
        {
            "crs": {
                "dims": "[x y]",
                "attributes": {
                    "crs_wkt": 'GEOGCRS["WGS 84",DATUM["World Geodetic System 1984",ELLIPSOID["WGS 84",6378137,298.257223563,LENGTHUNIT["metre",1]]],CS[ellipsoidal,2],AXIS["latitude",north,ORDER[1],ANGLEUNIT["degree",0.0174532925199433]],AXIS["longitude",east,ORDER[2],ANGLEUNIT["degree",0.0174532925199433]],ID["EPSG",4326]]'
                },
            }
        },
        {
            "crs": {
                "dims": "[x y]",
                "attributes": {
                    "grid_mapping_name": "rotated_latitude_longitude",
                    "grid_north_pole_latitude": 10,
                    "grid_north_pole_longitude": 50,
                },
            }
        },
    ],
)
def test_projection_from_config(projection: Dict):
    """
    Test parsing of projection information from config
    and check if it is written to the output file.
    """
    # Adding projection config to the example config
    config = yaml.safe_load(testconfig.VALID_EXAMPLE_CONFIG_YAML)
    config["inputs"]["danra_surface"]["projections"] = projection
    config["inputs"]["danra_height_levels"]["projections"] = projection
    config_yaml = yaml.dump(config)

    config = mdp.Config.from_yaml(config_yaml)

    ds = mdp.create_dataset(config=config)
    # Test CF-conform projection variable attributes
    # for var in set(ds.data_vars).intersection(VARIABLES_ON_PROJECTION):
    #     assert "grid_mapping" in ds[var].attrs
    #     assert ds[var].attrs["grid_mapping"] in projection.keys()
    # for var in set(ds.data_vars).difference(VARIABLES_ON_PROJECTION):
    #     assert "grid_mapping" not in ds[var].attrs

    for proj in projection.keys():
        assert proj in ds, "Projection variable not found in dataset"
        if "crs_wkt" in projection[proj]["attributes"]:
            assert pyproj.CRS.from_wkt(
                ds[proj].attrs["crs_wkt"]
            ) == pyproj.CRS.from_wkt(projection[proj]["attributes"]["crs_wkt"])
        else:
            ds[proj].attrs.pop("crs_wkt")
            assert pyproj.CRS.from_cf(ds[proj].attrs) == pyproj.CRS.from_cf(
                projection[proj]["attributes"]
            )


def test_danra_example():
    fp_config = Path(__file__).parent.parent / "example.danra.yaml"
    with tempfile.TemporaryDirectory(suffix=".zarr") as tmpdir:
        mdp.create_dataset_zarr(fp_config=fp_config, fp_zarr=tmpdir)
