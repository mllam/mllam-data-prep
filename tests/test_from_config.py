import shutil
import tempfile
from pathlib import Path

import isodate
import pytest
import yaml

import mllam_data_prep as mdp
import tests.data as testdata


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
        schema_version=testdata.SCHEMA_VERSION,
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
        schema_version=testdata.SCHEMA_VERSION,
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
        schema_version=testdata.SCHEMA_VERSION,
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


@pytest.mark.slow
def test_danra_example():
    fp_config = Path(__file__).parent.parent / "example.danra.yaml"
    with tempfile.TemporaryDirectory(suffix=".zarr") as tmpdir:
        mdp.create_dataset_zarr(fp_config=fp_config, fp_zarr=tmpdir)


@pytest.mark.parametrize("extra_content", [None, {"foobar": {"baz": 42}}])
def test_optional_extra_section(extra_content):
    """
    Test to ensure that the optional `extra` section of the config can contain
    arbitrary information and is not required for the config to be valid
    """
    tmpdir = tempfile.TemporaryDirectory()
    datasets = testdata.create_data_collection(
        data_kinds=["static"], fp_root=tmpdir.name
    )

    config_dict = dict(
        schema_version=testdata.SCHEMA_VERSION,
        dataset_version="v0.1.0",
        output=dict(
            variables=dict(
                static=["grid_index", "static_feature"],
            ),
        ),
        inputs=dict(
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

    if extra_content is not None:
        config_dict["extra"] = extra_content

    # write yaml config to file
    fn_config = "config.yaml"
    fp_config = Path(tmpdir.name) / fn_config
    with open(fp_config, "w") as f:
        yaml.dump(config_dict, f)

    mdp.create_dataset_zarr(fp_config=fp_config)


CONFIG_REVISION_EXAMPLES_PATH = Path(__file__).parent / "old_config_schema_examples"


def find_config_revision_examples():
    """
    Build a dictionary of examples for each revision of the config schema
    so that we can check that the examples are valid and up-to-date
    """
    examples = {}
    for fp in CONFIG_REVISION_EXAMPLES_PATH.rglob("*.yaml"):
        revision = fp.parent.name
        examples[revision] = fp

    return examples.values()


@pytest.mark.slow
@pytest.mark.parametrize("fp_example", find_config_revision_examples())
def test_config_revision_examples(fp_example):
    """
    Ensure that all the examples (which may be using different config schema
    versions)in the `config_examples` directory are valid
    """
    tmpdir = tempfile.TemporaryDirectory()

    # copy example to tempdir
    fp_config_copy = Path(tmpdir.name) / fp_example.name
    shutil.copy(fp_example, fp_config_copy)

    mdp.create_dataset_zarr(fp_config=fp_config_copy)


def test_sliced_dataset_can_instantiate_with_right_dimensions():
    """
    The sliced example has a 10x10 km slice, so there should be 4x4 = 16 points herekj.
    """
    fp = "tests/resources/sliced_example.danra.yaml"
    config = mdp.Config.from_yaml(open(fp))
    ds = mdp.create_dataset(config)
    # We pick a 10x10km slice of the data which should result in 16 grid points.
    assert ds.state.shape == (2, 49, 16)
