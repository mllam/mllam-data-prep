# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [unreleased](https://github.com/mllam/mllam-data-prep/compare/v0.1.0...HEAD)

- split dataset creation and storage to zarr into separate functions `mllam_data_prep.create_dataset(...)` and `mllam_data_prep.create_dataset_zarr(...)` respectively ![\#7](https://github.com/mllam/mllam-data-prep/pull/7)

- changes to spec from v0.1.0:
    - `sampling_dim` removed from `architectures` section of spec, this is not needed to create the training data
    - selection on variable coordinates values is now set with `inputs.{dataset_name}.variables.{variable_name}.values`
      rather than `inputs.{dataset_name}.variables.{variable_name}.sel`
    - when dimension-mapping method `stack_variables_by_var_name` is used the formatting string for the new variable
      is now called `name_format` rather than `name`
    - when dimension-mapping is done by simply renaming a dimension this configuration now needs to be set by providing
      the named method (`rename`) explicitly through the `method` key, i.e. rather than `{to_dim}: {from_dim}` it is now
      `{to_dim}: {method: rename, dim: {from_dim}}` to match the signature of the other dimension-mapping methods.
    - coordinate value ranges for the dimensions that the architecture expects as input has been renamed from
      `architecture.input_ranges` to `architecture.input_coord_ranges` to make the use more clear
    - attribute `inputs.{dataset_name}.name` attribute has been removed, with the key `dataset_name` this is
      superfluous

## [v0.1.0](https://github.com/mllam/mllam-data-prep/releases/tag/v0.1.0)

First tagged release of `mllam-data-prep` which includes functionality to
declaratively (in a yaml-config file) describe how the variables and
coordinates of a set of zarr-based source datasets are mapped to a new set of
variables with new coordinates to single a training dataset and write this
resulting single dataset to a new zarr dataset. This explicit mapping gives the
flexibility to target different different model architectures (which may
require different inputs with different shapes between architectures).
