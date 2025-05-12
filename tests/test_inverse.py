from pathlib import Path

import xarray as xr

import mllam_data_prep as mdp


def _test():
    fp_config = Path(__file__).parent.parent / "example.danra.yaml"
    config: mdp.Config = mdp.Config.from_yaml_file(fp_config)

    ds_transformed = mdp.create_dataset(config=config)

    for recreation_config in [None, config]:
        input_datasets_inverted = mdp.recreate_inputs(
            config=recreation_config, ds=ds_transformed
        )

        for input_name, input_config in config.inputs.items():
            ds_input = xr.open_dataset(input_config.path)
            ds_input_inverted = input_datasets_inverted[input_name]

            # the config may have performed subsetting (i.e. ds.sel) so we will
            # find coordinate ranges in `ds_input_inverted` and subset
            # `ds_input` to match. This allows us to check that the values are the same
            # for each coordinate in `ds_input_inverted`, check if it is present in `ds_input`
            for dim in ds_input_inverted.dims.keys():
                coord_values = ds_input_inverted.coords[dim].values
                if dim in ds_input.coords:
                    ds_input = ds_input.sel({dim: coord_values})

            # check that the variables in `ds_input_inverted` are present in `ds_input`
            for var in ds_input_inverted.data_vars:
                assert var in ds_input.data_vars
                # and check that the values are the same
                da_orig = ds_input[var]
                da_inverted = ds_input_inverted[var].transpose(*da_orig.dims)
                xr.testing.assert_equal(da_orig.coords, da_inverted.coords)


def test_danra_example_inverse():
    import ipdb

    with ipdb.launch_ipdb_on_exception():
        _test()
