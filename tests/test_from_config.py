import tempfile
from pathlib import Path

import yaml

import mllam_data_prep.main as mdp
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
                    start="2000-01-01T00:00",
                    end="2001-01-01T00:00",
                    step="PT1H",
                )
            ),
        ),
        inputs=[
            dict(
                name="danra_surface",
                path=datasets["surface_analysis"],
                dims=["time", "x", "y"],
                variables=testdata.DEFAULT_SURFACE_ANALYSIS_VARS,
                dim_mapping=dict(
                    time="analysis_time",
                    grid_index=["x", "y"],
                    feature=dict(
                        stack_variables_by_var_name=True,
                        name="{var_name}",
                    ),
                ),
                target="forcing",
            ),
            dict(
                name="danra_static",
                path=datasets["static"],
                dims=["x", "y"],
                variables=testdata.DEFAULT_STATIC_VARS,
                dim_mapping=dict(
                    grid_index=["x", "y"],
                    feature=dict(
                        stack_variables_by_var_name=True,
                        name="{var_name}",
                    ),
                ),
                target="static",
            ),
        ],
    )

    # write yaml config to file
    fn_config = "config.yaml"
    fp_config = Path(tmpdir.name) / fn_config
    with open(fp_config, "w") as f:
        yaml.dump(config, f)

    mdp.main(fp_config=fp_config)
