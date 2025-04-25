import datetime
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Optional, Union

import numpy as np
import xarray as xr
import yaml
import zarr
from loguru import logger
from packaging.version import Version

from mllam_data_prep.ops import selection

from . import __version__
from .config import (
    Config,
    InvalidConfigException,
    UnsupportedMllamDataPrepVersion,
    find_config_differences,
)
from .ops.chunking import chunk_dataset
from .ops.derive_variable import derive_variable
from .ops.loading import load_input_dataset
from .ops.mapping import map_dims_and_variables
from .ops.selection import select_by_kwargs
from .ops.statistics import calc_stats
from .ops.subsetting import extract_variable

if Version(zarr.__version__) >= Version("3"):
    from zarr.codecs import BloscCodec, BloscShuffle
else:
    from numcodecs import Blosc

# The config versions defined in SUPPORTED_CONFIG_VERSIONS are the ones currently supported.
# The `extra` field in the config that was added between v0.2.0 and v0.5.0 is optional, and
# the `derived_variables` field in the config added in v0.6.0 is also optional, so we can
# support v0.2.0, v0.5.0, and v0.6.0
SUPPORTED_CONFIG_VERSIONS = ["v0.2.0", "v0.5.0", "v0.6.0"]


def _check_dataset_attributes(ds, expected_attributes, dataset_name):
    # check that the dataset has the expected attributes with the expected values
    missing_attributes = set(expected_attributes.keys()) - set(ds.attrs.keys())
    if len(missing_attributes) > 0:
        raise ValueError(
            f"Dataset {dataset_name} is missing the following attributes: {missing_attributes}"
        )

    # check for attributes having the wrong value
    incorrect_attributes = {
        key: val for key, val in expected_attributes.items() if ds.attrs[key] != val
    }
    if len(incorrect_attributes) > 0:
        s_list = "\n".join(
            [
                f"{key}: {val} != {ds.attrs[key]}"
                for key, val in incorrect_attributes.items()
            ]
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
    if config.schema_version == "v0.2.0" and config.extra:
        raise ValueError(
            "Config schema version v0.2.0 does not support the `extra` field. Please "
            "update the schema version used in your config to v0.5.0."
        )

    output_config = config.output
    output_coord_ranges = output_config.coord_ranges
    chunking_config = config.output.chunking

    dataarrays_by_target = defaultdict(list)

    for dataset_name, input_config in config.inputs.items():
        path = input_config.path
        selected_variables = input_config.variables
        derived_variables = input_config.derived_variables
        target_output_var = input_config.target_output_variable
        expected_input_attributes = input_config.attributes
        expected_input_var_dims = input_config.dims

        output_dims = output_config.variables[target_output_var]

        logger.info(f"Loading dataset {dataset_name} from {path}")
        try:
            ds_input = load_input_dataset(fp=path)
        except Exception as ex:
            raise Exception(f"Error loading dataset {dataset_name} from {path}") from ex

        if input_config.coord_ranges is not None:
            ds_input = selection.select_by_kwargs(ds_input, **input_config.coord_ranges)

        # Initialize the output dataset
        ds = xr.Dataset()
        ds.attrs.update(ds_input.attrs)

        if selected_variables:
            logger.info(f"Extracting selected variables from dataset {dataset_name}")
            if isinstance(selected_variables, dict):
                for var_name, coords_to_sample in selected_variables.items():
                    ds[var_name] = extract_variable(
                        ds=ds_input,
                        var_name=var_name,
                        coords_to_sample=coords_to_sample,
                    )
            elif isinstance(selected_variables, list):
                for var_name in selected_variables:
                    ds[var_name] = extract_variable(ds=ds_input, var_name=var_name)
            else:
                raise ValueError(
                    "The `variables` argument should be a list or a dictionary"
                )

        if derived_variables:
            logger.info(f"Deriving variables from {dataset_name}")
            for var_name, derived_variable in derived_variables.items():
                ds[var_name] = derive_variable(
                    ds=ds_input,
                    derived_variable=derived_variable,
                    chunking=chunking_config,
                    target_dims=expected_input_var_dims,
                )

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
            output_coord_ranges = {
                k: w for k, w in output_coord_ranges.items() if k in output_dims
            }
            da_target = select_by_kwargs(da_target, **output_coord_ranges)

        dataarrays_by_target[target_output_var].append(da_target)

    ds = _merge_dataarrays_by_target(dataarrays_by_target=dataarrays_by_target)

    # need to drop the encoding so that we can write to zarr with new chunksizes
    ds = ds.drop_encoding()

    # default to making a single chunk for each dimension if chunksize is not specified
    # in the config
    logger.info(f"Chunking dataset with {chunking_config}")
    chunks = {dim: chunking_config.get(dim, int(ds[dim].count())) for dim in ds.dims}
    ds = chunk_dataset(ds, chunks)

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
    ds.attrs["creation_config"] = config.to_yaml()

    return ds


def create_dataset_zarr(
    fp_config: Path,
    fp_zarr: Optional[Union[str, Path]] = None,
    overwrite: str = "always",
):
    """
    Create a dataset from the input datasets specified in the config file and
    write it to a zarr dataset. The path to the zarr dataset is the same as the
    config file (unless `fp_zarr` is provided), but with the extension changed
    to '.zarr'.

    Parameters
    ----------
    fp_config : Path
        The path to the configuration file.
    fp_zarr : Path, optional
        The path to the zarr file to write the dataset to. If not provided, the zarr file will be written
        to the same directory as the config file with the extension changed to '.zarr'.
    overwrite : str, optional
        How to handle an existing dataset at the provided path. Options are:
        - "always": Always delete the existing dataset (default)
        - "never": Never delete the existing dataset
        - "on_config_change": Only delete the existing dataset if the configuration has changed
    """
    config = Config.from_yaml_file(file=fp_config)

    if fp_zarr is None:
        fp_zarr = fp_config.parent / fp_config.name.replace(".yaml", ".zarr")
    else:
        fp_zarr = Path(fp_zarr)

    if fp_zarr.exists():
        if overwrite == "never":
            ds_existing = xr.open_zarr(fp_zarr)
            try:
                config_differences = find_config_differences(
                    config=config, ds_existing=ds_existing
                )
            except UnsupportedMllamDataPrepVersion:
                config_differences = None

            ex_str = (
                f"There already exists a dataset at {fp_zarr}, and the overwrite option is set to 'never'. "
                "Either delete the existing dataset or set overwrite='always' to overwrite it. "
            )
            # try and parse the differences in the config in case the existing
            # dataset was created with a supported version
            if config_differences:
                ex_str += (
                    "The existing dataset was created with a different configuration than the current one. "
                    "Differences between existing and new configuration: \n"
                    f"{yaml.dump(config_differences, default_flow_style=False)}"
                )
            raise FileExistsError(ex_str)
        elif overwrite == "on_config_change":
            try:
                ds_existing = xr.open_zarr(fp_zarr)
                config_differences = find_config_differences(
                    config=config, ds_existing=ds_existing
                )
            except UnsupportedMllamDataPrepVersion as ex:
                raise FileExistsError(
                    f"There already exists a dataset at {fp_zarr}, however it was created with an older version of mllam-data-prep "
                    "and so doesn't contain a record of the configuration used to create it. Either delete the existing dataset or "
                    "set overwrite='always' to overwrite it."
                ) from ex

            if config_differences:
                logger.info(
                    "The existing dataset was created with a different configuration than the current one."
                )
                diff_yaml = yaml.dump(config_differences, default_flow_style=False)
                logger.info(
                    f"Differences between existing and new configuration:\n{diff_yaml}"
                )
                logger.info(f"Removing existing dataset at {fp_zarr}")
                shutil.rmtree(fp_zarr)
            else:
                logger.info(
                    f"Skipping creation of writing of dataset to {fp_zarr} as the configuration is unchanged"
                )
                return
        elif overwrite == "always":
            logger.info(f"Removing existing dataset at {fp_zarr}")
            shutil.rmtree(fp_zarr)
        else:
            raise NotImplementedError(
                f"Unsupported overwrite option {overwrite}. Options are 'always', 'never', or 'on_config_change'"
            )

    ds = create_dataset(config=config)

    logger.info("Writing dataset to zarr")

    # use zstd compression since it has a good balance of speed and compression ratio
    # https://engineering.fb.com/2016/08/31/core-infra/smaller-and-faster-data-compression-with-zstandard/
    if Version(zarr.__version__) >= Version("3"):
        compressor = BloscCodec(cname="zstd", clevel=3, shuffle=BloscShuffle.bitshuffle)
        encoding = {v: {"compressors": compressor} for v in ds.data_vars}
    else:
        compressor = Blosc(cname="zstd", clevel=1, shuffle=Blosc.BITSHUFFLE)
        encoding = {v: {"compressor": compressor} for v in ds.data_vars}

    # default mode to "w-" so that an error is raised if the dataset already exists
    ds.to_zarr(fp_zarr, consolidated=True, mode="w-", encoding=encoding)
    logger.info(f"Wrote training-ready dataset to {fp_zarr}")

    logger.info(ds)
