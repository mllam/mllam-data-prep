import datetime
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import xarray as xr
from loguru import logger
from numcodecs import Blosc

from . import __version__
from .config import Config, InvalidConfigException
from .ops.loading import load_and_subset_dataset
from .ops.mapping import map_dims_and_variables
from .ops.selection import select_by_kwargs
from .ops.statistics import calc_stats

# the `extra` field in the config that was added between v0.2.0 and v0.5.0 is
# optional, so we can support both v0.2.0 and v0.5.0
SUPPORTED_CONFIG_VERSIONS = ["v0.2.0", "v0.5.0"]


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
        logger.info(f"Merging dataarrays for target variable `{target}`")
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


def create_dataset(config: Config):
    """
    Create a dataset from the input datasets specified in the config file.

    Parameters
    ----------
    config : Config
        The configuration object defining the input datasets and how to map them to the output dataset.

    Returns
    -------
    xr.Dataset
        The dataset created from the input datasets with a variable for each output
        as defined in the config file.
    """
    if not config.schema_version in SUPPORTED_CONFIG_VERSIONS:
        raise ValueError(
            f"Unsupported schema version {config.schema_version}. Only schema versions "
            f" {', '.join(SUPPORTED_CONFIG_VERSIONS)} are supported by mllam-data-prep "
            f"v{__version__}."
        )
    if config.schema_version == "v0.2.0" and config.extra is not None:
        raise ValueError(
            "Config schema version v0.2.0 does not support the `extra` field. Please "
            "update the schema version used in your config to v0.5.0."
        )

    output_config = config.output
    output_coord_ranges = output_config.coord_ranges

    dataarrays_by_target = defaultdict(list)

    for dataset_name, input_config in config.inputs.items():
        path = input_config.path
        variables = input_config.variables
        target_output_var = input_config.target_output_variable
        expected_input_attributes = input_config.attributes or {}
        expected_input_var_dims = input_config.dims

        output_dims = output_config.variables[target_output_var]

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

        dim_mapping = input_config.dim_mapping

        # check that there is an entry for each arch dimension
        # in the dim_mapping so that we know how to construct the
        # final dataset
        missing_dims = set(output_dims) - set(dim_mapping.keys())
        if missing_dims:
            raise ValueError(
                f"Missing dimension mapping for {missing_dims}"
                f" for input dataset {dataset_name}, please provide"
                " a mapping for all output dimensions by"
                " using the 'dim_mapping' key in the input dataset"
            )

        logger.info(
            f"Mapping dimensions and variables for dataset {dataset_name} to {target_output_var}"
        )
        try:
            da_target = map_dims_and_variables(
                ds=ds,
                dim_mapping=dim_mapping,
                expected_input_var_dims=expected_input_var_dims,
            )
        except Exception as ex:
            raise Exception(
                f"There was an issue stacking dimensions and variables to"
                f" produce variable {target_output_var} from dataset {dataset_name}"
            ) from ex

        da_target.attrs["source_dataset"] = dataset_name

        # only need to do selection for the coordinates that the input dataset actually has
        if output_coord_ranges is not None:
            selection_kwargs = {}
            for dim in output_dims:
                if dim in output_coord_ranges:
                    selection_kwargs[dim] = output_coord_ranges[dim]
            da_target = select_by_kwargs(da_target, **selection_kwargs)

        dataarrays_by_target[target_output_var].append(da_target)

    ds = _merge_dataarrays_by_target(dataarrays_by_target=dataarrays_by_target)

    # need to drop the encoding so that we can write to zarr with new chunksizes
    ds = ds.drop_encoding()

    # default to making a single chunk for each dimension if chunksize is not specified
    # in the config
    chunking_config = config.output.chunking or {}
    logger.info(f"Chunking dataset with {chunking_config}")
    chunks = {d: chunking_config.get(d, int(ds[d].count())) for d in ds.dims}
    ds = ds.chunk(chunks)

    splitting = config.output.splitting

    if splitting is not None:
        splits = splitting.splits
        logger.info(
            f"Setting splitting information to define `{list(splits.keys())}` splits "
            f"along dimension `{splitting.dim}`"
        )

        for split_name, split_config in splits.items():
            if split_config.compute_statistics is not None:
                ds_split = ds.sel(
                    {splitting.dim: slice(split_config.start, split_config.end)}
                )
                logger.info(f"Computing statistics for split {split_name}")
                split_stats = calc_stats(
                    ds=ds_split,
                    statistics_config=split_config.compute_statistics,
                    splitting_dim=splitting.dim,
                )
                for op, op_dataarrays in split_stats.items():
                    for var_name, da in op_dataarrays.items():
                        ds[f"{var_name}__{split_name}__{op}"] = da

        # add a new variable which contains the start, stop for each split, the coords would then be the split names
        # and the data would be the start, stop values
        split_vals = np.array([[split.start, split.end] for split in splits.values()])
        da_splits = xr.DataArray(
            split_vals,
            dims=["split_name", "split_part"],
            coords={"split_name": list(splits.keys()), "split_part": ["start", "end"]},
        )
        ds["splits"] = da_splits

    ds.attrs = {}
    ds.attrs["schema_version"] = config.schema_version
    ds.attrs["dataset_version"] = config.dataset_version
    ds.attrs["created_on"] = datetime.datetime.now().replace(microsecond=0).isoformat()
    ds.attrs[
        "created_with"
    ] = "mllam-data-prep (https://github.com/mllam/mllam-data-prep)"
    ds.attrs["mdp_version"] = f"v{__version__}"

    return ds


def create_dataset_zarr(fp_config, fp_zarr: str = None):
    """
    Create a dataset from the input datasets specified in the config file and write it to a zarr file.
    The path to the zarr file is the same as the config file, but with the extension changed to '.zarr'.

    Parameters
    ----------
    fp_config : Path
        The path to the configuration file.
    fp_zarr : Path, optional
        The path to the zarr file to write the dataset to. If not provided, the zarr file will be written
        to the same directory as the config file with the extension changed to '.zarr'.
    """
    config = Config.from_yaml_file(file=fp_config)

    ds = create_dataset(config=config)

    logger.info("Writing dataset to zarr")
    if fp_zarr is None:
        fp_zarr = fp_config.parent / fp_config.name.replace(".yaml", ".zarr")
    else:
        fp_zarr = Path(fp_zarr)

    if fp_zarr.exists():
        logger.info(f"Removing existing dataset at {fp_zarr}")
        shutil.rmtree(fp_zarr)

    # use zstd compression since it has a good balance of speed and compression ratio
    # https://engineering.fb.com/2016/08/31/core-infra/smaller-and-faster-data-compression-with-zstandard/
    compressor = Blosc(cname="zstd", clevel=1, shuffle=Blosc.BITSHUFFLE)
    encoding = {v: {"compressor": compressor} for v in ds.data_vars}

    ds.to_zarr(fp_zarr, consolidated=True, mode="w", encoding=encoding)
    logger.info(f"Wrote training-ready dataset to {fp_zarr}")

    logger.info(ds)
