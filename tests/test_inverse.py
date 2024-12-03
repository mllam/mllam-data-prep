from pathlib import Path

import xarray as xr

import mllam_data_prep as mdp


def test_danra_example_inverse():
    fp_config = Path(__file__).parent.parent / "example.danra.yaml"
    config: mdp.Config = mdp.Config.from_yaml_file(fp_config)

    ds_transformed = mdp.create_dataset(config=config)

    input_datasets_inverted = mdp.recreate_inputs(config=config, ds=ds_transformed)

    for input_name, input_config in config.inputs.items():
        ds_input = xr.open_dataset(input_config.path)
        ds_input_inverted = input_datasets_inverted[input_name]

        # find coordinate ranges in `ds_input_inverted` and subset `ds_input` to match
        for dim, coord in ds_input_inverted.coords.items():
            if dim in ds_input.coords:
                ds_input = ds_input.sel({dim: coord})

        # check that the variables in `ds_input_inverted` are present in `ds_input`
        for var in ds_input_inverted.data_vars:
            assert var in ds_input.data_vars
            # and check that the values are the same
            xr.testing.assert_equal(ds_input[var], ds_input_inverted[var])
