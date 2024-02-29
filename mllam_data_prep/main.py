from collections import defaultdict

import xarray as xr
import yaml

from .ops.loading import load_and_subset_dataset
from .ops.mapping import map_dims_and_variables


def _check_dataset_attributes(ds, expected_attributes, dataset_name):
    # check that the dataset has the expected attributes with the expected values
    missing_attributes = set(expected_attributes.keys()) - set(ds.attrs.keys())
    if len(missing_attributes) > 0:
        raise ValueError(
            f"Dataset {dataset_name} is missing the following attributes: {missing_attributes}"
        )

    # check for attributes having the wrong value
    incorrect_attributes = {
        k: v for k, v in expected_attributes.items() if ds.attrs[k] != v
    }
    if len(incorrect_attributes) > 0:
        s_list = "\n".join(
            [f"{k}: {v} != {ds.attrs[k]}" for k, v in incorrect_attributes.items()]
        )
        raise ValueError(
            f"Dataset {dataset_name} has the following incorrect attributes: {s_list}"
        )


def _merge_dataarrays_by_target(dataarrays_by_target):
    dataarrays = {}
    for target, das in dataarrays_by_target.items():
        concat_dim = None
        for da in das:
            d = da.attrs.get("variables_mapping_dim", None)
            if d is None:
                raise ValueError(
                    f"Dataarray for target {target} does not have the 'variables_mapping_dim' attribute"
                )
            if concat_dim is not None and d != concat_dim:
                raise ValueError(
                    f"Dataarrays for target {target} have different 'variables_mapping_dim' attributes: {d} != {concat_dim}"
                )
            concat_dim = d
        dataarrays[target] = xr.concat(das, dim=concat_dim)

    ds = xr.Dataset(dataarrays)
    return ds


def main(fp_config):
    with open(fp_config, "r") as fh:
        config = yaml.load(fh, Loader=yaml.FullLoader)

    architecture_config = config["architecture"]

    dataarrays_by_target = defaultdict(list)

    for dataset_config in config["inputs"]:
        path = dataset_config["path"]
        variables = dataset_config["variables"]
        dataset_name = dataset_config["name"]
        target_arch_var = dataset_config["target"]
        expected_input_attributes = dataset_config.get("attributes", {})

        arch_dims = architecture_config["input_variables"][target_arch_var]

        try:
            ds = load_and_subset_dataset(fp=path, variables=variables)
        except Exception as ex:
            raise Exception(f"Error loading dataset {dataset_name} from {path}") from ex
        _check_dataset_attributes(
            ds=ds,
            expected_attributes=expected_input_attributes,
            dataset_name=dataset_name,
        )

        dim_mapping = dataset_config["dim_mapping"]

        # check that there is an entry for each arch dimension
        # in the dim_mapping so that we know how to construct the
        # final dataset
        missing_dims = set(arch_dims) - set(dim_mapping.keys())
        if missing_dims:
            raise ValueError(
                f"Missing dimension mapping for {missing_dims}"
                f" for input dataset {dataset_name}, please provide"
                " a mapping for all architecture dimensions in"
                " using the 'dim_mapping' key in the input dataset"
            )

        da_target = map_dims_and_variables(ds=ds, dim_mapping=dim_mapping)
        dataarrays_by_target[target_arch_var].append(da_target)

    ds = _merge_dataarrays_by_target(dataarrays_by_target=dataarrays_by_target)
    print(ds)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to the config file")
    args = parser.parse_args()

    main(fp_config=args.config)
