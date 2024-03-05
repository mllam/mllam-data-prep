import shutil
from collections import defaultdict
from pathlib import Path

import xarray as xr
from loguru import logger

from .config import ConfigDict, InvalidConfigException
from .ops.loading import load_and_subset_dataset
from .ops.mapping import map_dims_and_variables
from .ops.selection import select_by_kwargs


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
    attrs_to_keep = ["source_dataset"]
    dataarrays = []
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

        for da in das:
            for attr in attrs_to_keep:
                # create a aux coord for each attribute we want to keep
                # (for example the name of the source dataset)
                # so that we have this in the resulting dataset
                da.coords[f"{concat_dim}_{attr}"] = xr.DataArray(
                    [da.attrs.pop(attr)] * int(da[concat_dim].count()),
                    dims=[concat_dim],
                )

        da_target = xr.concat(das, dim=concat_dim)
        da_target.name = target
        dataarrays.append(da_target)

    # by doing a merge with join="exact" we make sure that the dataarrays
    # are aligned along the same dimensions, and that the coordinates are
    # the same for all dataarrays. Otherwise xarray will fill in with NaNs
    # for any missing coordinate values
    try:
        ds = xr.merge(dataarrays, join="exact")
    except ValueError as ex:
        if ex.args[0].startswith("cannot align objects with join='exact'"):
            raise InvalidConfigException(
                f"Couldn't merge together the dataarrays for all targets ({', '.join(dataarrays_by_target.keys())})"
                f" This is likely because the dataarrays have different dimensions or coordinates."
                " Maybe you need to give the 'feature' dimension a unique name for each target variable?"
            ) from ex
        else:
            raise ex
    return ds


def main(fp_config):
    config = ConfigDict.load(fp_config=fp_config)

    architecture_config = config["architecture"]
    architecture_input_ranges = architecture_config.get("input_range", {})

    dataarrays_by_target = defaultdict(list)

    for dataset_name, input_config in config["inputs"].items():
        path = input_config["path"]
        variables = input_config["variables"]
        target_arch_var = input_config["target"]
        expected_input_attributes = input_config.get("attributes", {})
        expected_input_var_dims = input_config["dims"]

        arch_dims = architecture_config["input_variables"][target_arch_var]

        logger.info(f"Loading dataset {dataset_name} from {path}")
        try:
            ds = load_and_subset_dataset(fp=path, variables=variables)
        except Exception as ex:
            raise Exception(f"Error loading dataset {dataset_name} from {path}") from ex
        _check_dataset_attributes(
            ds=ds,
            expected_attributes=expected_input_attributes,
            dataset_name=dataset_name,
        )

        dim_mapping = input_config["dim_mapping"]

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

        logger.info(
            f"Mapping dimensions and variables for dataset {dataset_name} to {target_arch_var}"
        )
        da_target = map_dims_and_variables(
            ds=ds,
            dim_mapping=dim_mapping,
            expected_input_var_dims=expected_input_var_dims,
        )
        da_target.attrs["source_dataset"] = dataset_name

        if architecture_input_ranges is not None:
            da_target = select_by_kwargs(da_target, **architecture_input_ranges)

        dataarrays_by_target[target_arch_var].append(da_target)

    ds = _merge_dataarrays_by_target(dataarrays_by_target=dataarrays_by_target)
    # need to drop the encoding so that we can write to zarr with new chunksizes
    ds = ds.drop_encoding()

    # default to making a single chunk for each dimension if chunksize is not specified
    # in the config
    config_chunking = architecture_config.get("chunking", {})
    chunks = {d: config_chunking.get(d, int(ds[d].count())) for d in ds.dims}
    ds = ds.chunk(chunks)

    print(ds)

    fp_out = fp_config.parent / fp_config.name.replace(".yaml", ".zarr")
    if fp_out.exists():
        logger.info(f"Removing existing dataset at {fp_out}")
        shutil.rmtree(fp_out)
    ds.to_zarr(fp_out)
    logger.info(f"Wrote training-ready dataset to {fp_out}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to the config file", type=Path)
    args = parser.parse_args()

    main(fp_config=args.config)
