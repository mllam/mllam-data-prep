import tempfile
from pathlib import Path

import isodate
import pytest
import yaml

import mllam_data_prep.create_dataset as mdp
import tests.data as testdata
from mllam_data_prep.config import InvalidConfigException


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

    config = dict(
        schema_version="v0.1.0",
        dataset_version="v0.1.0",
        architecture=dict(
            sampling_dim="time",
            input_variables=dict(
                static=["grid_index", "static_feature"],
                state=["time", "grid_index", "state_feature"],
                forcing=["time", "grid_index", "forcing_feature"],
            ),
        ),
        inputs=dict(
            danra_surface=dict(
                name="danra_surface",
                path=datasets["surface_analysis"],
                dims=["analysis_time", "x", "y"],
                variables=testdata.DEFAULT_SURFACE_ANALYSIS_VARS,
                dim_mapping=dict(
                    time="analysis_time",
                    grid_index=dict(
                        method="flatten",
                        dims=["x", "y"],
                    ),
                    forcing_feature=dict(
                        method="stack_variables_by_var_name",
                        name_format="{var_name}",
                    ),
                ),
                target_architecture_variable="forcing",
            ),
            danra_static=dict(
                name="danra_static",
                path=datasets["static"],
                dims=["x", "y"],
                variables=testdata.DEFAULT_STATIC_VARS,
                dim_mapping=dict(
                    grid_index=dict(
                        method="flatten",
                        dims=["x", "y"],
                    ),
                    static_feature=dict(
                        method="stack_variables_by_var_name",
                        name_format="{var_name}",
                    ),
                ),
                target_architecture_variable="static",
            ),
        ),
    )

    # write yaml config to file
    fn_config = "config.yaml"
    fp_config = Path(tmpdir.name) / fn_config
    with open(fp_config, "w") as f:
        yaml.dump(config, f)

    mdp.main(fp_config=fp_config)


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
        schema_version="v0.1.0",
        dataset_version="v0.1.0",
        architecture=dict(
            sampling_dim="time",
            input_variables=dict(
                static=["grid_index", "feature"],
                state=["time", "grid_index", "feature"],
                forcing=["time", "grid_index", "feature"],
            ),
            input_range=dict(
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
                    time="analysis_time",
                    grid_index=dict(
                        method="flatten",
                        dims=["x", "y"],
                    ),
                    feature=dict(
                        method="stack_variables_by_var_name",
                        name_format="{var_name}",
                    ),
                ),
                target_architecture_variable="forcing",
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
        mdp.main(fp_config=fp_config)
    else:
        print(
            f"Expecting ValueError for source_data_contains_time_range={source_data_contains_time_range} and time_stepsize={time_stepsize}"
        )
        with pytest.raises(ValueError):
            mdp.main(fp_config=fp_config)


@pytest.mark.parametrize("use_common_feature_var_name", [True, False])
def test_feature_collision(use_common_feature_var_name):
    """
    Use to arch target_architecture_variable variables which have a different number of features and
    therefore need a unique feature dimension for each target_architecture_variable. This should raise
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
        schema_version="v0.1.0",
        dataset_version="v0.1.0",
        architecture=dict(
            sampling_dim="time",
            input_variables=dict(
                static=["grid_index", static_feature_var_name],
                state=["time", "grid_index", state_feature_var_name],
            ),
        ),
        inputs=dict(
            danra_surface=dict(
                name="danra_surface",
                path=datasets["surface_analysis"],
                dims=["analysis_time", "x", "y"],
                variables=testdata.DEFAULT_SURFACE_ANALYSIS_VARS,
                dim_mapping={
                    "time": "analysis_time",
                    "grid_index": dict(
                        method="flatten",
                        dims=["x", "y"],
                    ),
                    state_feature_var_name: dict(
                        method="stack_variables_by_var_name",
                        name_format="{var_name}",
                    ),
                },
                target_architecture_variable="state",
            ),
            danra_static=dict(
                name="danra_static",
                path=datasets["static"],
                dims=["x", "y"],
                variables=testdata.DEFAULT_STATIC_VARS,
                dim_mapping={
                    "grid_index": dict(
                        dims=["x", "y"],
                        method="flatten",
                    ),
                    static_feature_var_name: dict(
                        method="stack_variables_by_var_name",
                        stack_variables_by_var_name=True,
                        name_format="{var_name}",
                    ),
                },
                target_architecture_variable="static",
            ),
        ),
    )

    # write yaml config to file
    fn_config = "config.yaml"
    fp_config = Path(tmpdir.name) / fn_config
    with open(fp_config, "w") as f:
        yaml.dump(config, f)

    if use_common_feature_var_name:
        with pytest.raises(InvalidConfigException):
            mdp.main(fp_config=fp_config)
    else:
        mdp.main(fp_config=fp_config)


def test_danra_example():
    fp_config = Path(__file__).parent.parent / "example.danra.yaml"
    mdp.main(fp_config=fp_config)
