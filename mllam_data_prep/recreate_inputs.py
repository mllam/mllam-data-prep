import datetime
from typing import Optional

import cf_xarray as cfxr
import parse
import xarray as xr
from loguru import logger

from . import __version__
from .config import Config
from .create_dataset import SOURCE_DATASET_NAME_ATTR


def _split_coord_values_as_variables(
    da: xr.DataArray, name_format: str, target_dim: str
):
    """
    Split the coordinate values of a DataArray into separate variables based on a name format.

    Parameters
    ----------
    da : xr.DataArray
        The DataArray to split.
    name_format : str
        The format string used to parse the coordinate values.
    target_dim : str
        The name of the coordinate dimension to split.

    Returns
    -------
    dict[str, xr.DataArray]
        A dictionary of new DataArrays, where the keys are the variable names and the values are the DataArrays.
    """

    dataarrays = []
    coord_values = da[target_dim].values
    for coord_value in coord_values:

        da_feature = da.sel({target_dim: coord_value})
        name_parts = dict(parse.parse(name_format, coord_value).named)
        # the "var_name" part of the coordinate value is the name of the
        # variable that that the data came from
        var_name = name_parts.pop("var_name")
        # the rest are coordinate names and values
        coords = name_parts

        da_original = da_feature.copy().squeeze()
        da_original.name = var_name
        for k, v in coords.items():
            # TODO: in future we should enforce that the format strings contain
            # types so that we can parse the values to the correct type
            if "." in v:
                try:
                    v = float(v)
                except ValueError:
                    pass
            else:
                try:
                    v = int(v)
                except ValueError:
                    pass

            da_original[k] = v

        da_original = da_original.expand_dims(list(coords.keys()))

        var_units = da_feature[f"{target_dim}_units"].load().item()
        var_long_name = da_feature[f"{target_dim}_long_name"].load().item()

        da_original.attrs["units"] = var_units
        da_original.attrs["long_name"] = var_long_name

        # remove the coords (and aux coords) that represented the feature
        # coordinate, the units, long_name and source_dataset
        for d in [
            target_dim,
            f"{target_dim}_units",
            f"{target_dim}_long_name",
            f"{target_dim}_{SOURCE_DATASET_NAME_ATTR}",
        ]:
            da_original = da_original.drop_vars(d)

        dataarrays.append(da_original)

    ds = xr.merge(dataarrays, join="exact")

    return ds


def recreate_inputs(
    ds: xr.Dataset,
    config: Optional[Config] = None,
    only_selected_inputs: Optional[list[str]] = None,
) -> dict[str, xr.Dataset]:
    """
    Recreate the input datasets from a zarr file created by
    `create_dataset_zarr` by applying inverse operations to each step.

    Parameters
    ----------
    ds : xr.Dataset
        The mllam-data-prep dataset to recreate the input datasets from.
    config: Config, optional
        The configuration object defining the input datasets and how to map them to the output dataset.
        If not provided, the config will be read from the dataset attributes.
    only_selected_inputs : list[str], optional
        If provided, only the input datasets with these names will be recreated.
        If not provided, all input datasets will be recreated.

    Returns
    -------
    dict[str, xr.Dataset]
        A dictionary of input datasets, where the keys are the input dataset names
        and the values are the recreated input datasets.
    """
    input_datasets = {}
    if config is None:
        config = Config.from_yaml(ds.creation_config)

    if only_selected_inputs is None:
        only_selected_inputs = list(config.inputs.keys())

    for input_name in only_selected_inputs:
        input_config = config.inputs[input_name]
        dim_mapping = input_config.dim_mapping
        if input_config.target_output_variable not in ds:
            logger.warning(
                f"Target output variable {input_config.target_output_variable} "
                f"for input dataset {input_name} not found in dataset, skipping"
            )
            continue
        da_target = ds[input_config.target_output_variable]

        # 1. First, we need to split out the coordinate that was used to stack
        # multiple variables into. Find the dim mapping item that is the one
        # where variable names are stacked into a feature dimension
        feature_dim_name = None
        for output_dim, mapping_config in dim_mapping.items():
            if mapping_config.method == "stack_variables_by_var_name":
                feature_dim_name = output_dim
                name_format: str = str(mapping_config.name_format)
                break

        if feature_dim_name is None:
            raise ValueError(
                f"Could not find a feature dimension in the dim_mapping for input dataset {input_name}"
            )
        dim_mapping.pop(output_dim)
        ds_source = _split_coord_values_as_variables(
            da=da_target,
            name_format=name_format,
            target_dim=feature_dim_name,
        )

        # 2. And then we handle the other mapping of dimensions
        for output_dim, mapping_config in dim_mapping.items():
            method_name = mapping_config.method
            if method_name == "stack_variables_by_var_name":
                raise Exception(
                    "`stack_variables_by_var_name` should have been handled above"
                )
            elif method_name == "rename":
                # rename the dimension back again
                ds_source = ds_source.rename({output_dim: mapping_config.dim})
            elif method_name == "stack":
                # unstack the stacked dimension
                # To allow MultiIndex to zarr/netcdf
                # mllam_data_prep.create_dataset encodes these using
                # cf-compliant "gather compression" (see
                # https://cf-xarray.readthedocs.io/en/latest/coding.html).
                # To make sure decoding of these MultiIndex is possible we need
                # to ensure that the required stacked coordinates (defined
                # through the "compress" attribute) are included in the dataset
                compress_attr = ds_source[output_dim].attrs["compress"]
                required_coords = compress_attr.split(" ")
                for coord in required_coords:
                    if coord not in ds.coords:
                        raise ValueError(
                            f"Cannot unstack dimension {output_dim} as the required "
                            f"coordinate {coord} is not in the dataset"
                        )
                    ds_source[coord] = ds.coords[coord]
                ds_source = cfxr.decode_compress_to_multi_index(
                    ds_source, idxnames=output_dim
                ).unstack(output_dim)
            else:
                raise NotImplementedError(method_name)

        # 3. Finally, we remove any variables that were derived from the input
        # dataset
        if input_config.derived_variables is not None:
            derived_variables = input_config.derived_variables.keys()
            ds_source = ds_source.drop_vars(derived_variables)

        # 4. Remove chunking information so that we can save the dataset with a
        # new chunking
        for var in ds_source.data_vars:
            if "chunks" in ds_source[var].encoding:
                del ds_source[var].encoding["chunks"]

        input_datasets[input_name] = ds_source

    return input_datasets


def _parse_string_to_dict(input_string, value_type=int):
    """
    Parses a comma-separated key-value string into a dictionary.
    The format is 'key=value,key2=value2'. Empty values and multiple values for the same key are not allowed.

    Parameters
    ----------
    input_string : str
        The input string to parse. It should be in the format 'key=value,key2=value2'.
    value_type : type
        The type to which the values should be converted. Default is int.

    Returns
    -------
    dict
        A dictionary with keys and values parsed from the input string.

    Raises
    ------
        ValueError: If the input string is not in the correct format.
        TypeError: If the value cannot be converted to the specified type.
        KeyError: If a key appears more than once in the input string.
    """

    result = {}

    for item in input_string.split(","):
        key_value_pair = item.strip().split("=")
        if len(key_value_pair) != 2:
            raise ValueError(
                "Invalid format. Each key-value pair must be separated by '=' and the pair must be separated by ','."
            )

        key, value = key_value_pair
        if key in result:
            raise KeyError("Duplicate keys are not allowed.")

        result[key] = value_type(value)

    return result


@logger.catch(reraise=True)
def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        description="Recreate the input datasets from a zarr file created by create_dataset_zarr",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "zarr_dataset_path",
        type=str,
        help="The path to the zarr file to recreate the input datasets from",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="The path to the configuration file that was used to create the dataset. "
        "If not provided, the config will be read from the dataset attributes.",
    )
    parser.add_argument(
        "--output-path-format",
        default="{input_name}.zarr",
        type=str,
        help="The format string for the output path. The input name will be replaced with the input dataset name",
    )
    parser.add_argument(
        "--chunks",
        type=_parse_string_to_dict,
        default={},
        help="The chunks to use for the output datasets. The format is"
        "'key=value,key2=value2'. I.e. to use chunksize 1 along the time"
        "dimension and 100 along the x-dimension, use `--chunks time=1,x=100`",
    )
    parser.add_argument(
        "--only-selected-inputs",
        nargs="*",
        default=None,
        help="If provided, only the input datasets with these names will be recreated. "
        "If not provided, all input datasets will be recreated.",
    )

    args = parser.parse_args(argv)

    config = Config.from_yaml_file(args.config_path) if args.config_path else None

    ds = xr.open_zarr(args.zarr_dataset_path)
    input_datasets = recreate_inputs(
        ds=ds, config=config, only_selected_inputs=args.only_selected_inputs
    )
    if args.only_selected_inputs is not None:
        missing_inputs = set(args.only_selected_inputs) - set(input_datasets.keys())
        if missing_inputs:
            raise ValueError(
                f"The following input datasets were not found in the zarr file: {missing_inputs}. "
                f"The available input datasets are: {list(input_datasets.keys())})"
            )
        input_datasets = {
            k: v for k, v in input_datasets.items() if k in args.only_selected_inputs
        }

    for input_name, ds_input in input_datasets.items():
        ds_input.attrs = {}
        ds_input.attrs["recreated_from"] = args.zarr_dataset_path
        if config is not None:
            ds_input.attrs["recreation_config"] = config.to_yaml()
        ds_input.attrs["source_dataset_name"] = input_name
        ds_input.attrs["created_by"] = "mllam_data_prep.recreate_inputs"
        ds_input.attrs["created_on"] = datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat()
        ds_input.attrs["mdp-version"] = __version__
        output_path = args.output_path_format.format(input_name=input_name)
        logger.info(
            f"Saving input dataset {input_name} to {output_path} with chunks={args.chunks}"
        )
        ds_input.chunk(args.chunks).to_zarr(output_path, mode="w", consolidated=True)


if __name__ == "__main__":
    main()
