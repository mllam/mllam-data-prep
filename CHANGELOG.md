# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [unreleased](https://github.com/mllam/mllam-data-prep/compare/v0.6.1...HEAD)

### Fixes
- fix bug where coordinate selection of an unshared dimension isn't applied to subsequent ouput variables when an output variable without this dimension is processed before the others [\#87](https://github.com/mllam/mllam-data-prep/pull/87) @zweihuehner

## [v0.6.1](https://github.com/mllam/mllam-data-prep/release/tag/v0.6.1)

[All changes](https://github.com/mllam/mllam-data-prep/compare/v0.6.1...v0.6.0)

This release contains bugfixes to update tests to use newer version of pre-commit, use correct python version, and remove uses of incompatible typing notation.

### Fixes
- use old union typing notation compatible with all required python versions [\#77](https://github.com/mllam/mllam-data-prep/pull/77) @SimonKamuk

### Maintenance
- update pre-commit action to v3.0.1 [\#77](https://github.com/mllam/mllam-data-prep/pull/77) @SimonKamuk
- fix tests to use expected python version from test matrix [\#77](https://github.com/mllam/mllam-data-prep/pull/77) @SimonKamuk

## [v0.6.0](https://github.com/mllam/mllam-data-prep/release/tag/v0.6.0)

[All changes](https://github.com/mllam/mllam-data-prep/compare/v0.6.0...v0.5.0)

This release adds the ability to slice input data by any coordinate, derive variables from input datasets, and store config in created datasets. It also adds support for zarr 3.0.0 and above, and a mypy typing action to pre-commit hooks. In addition a number of bugs were fixed related to adding unwanted dimensions to the dataset, chunk size estimates, and derived functions. The release also includes a number of maintenance updates including updating the DANRA test dataset to v0.2.0 (which smaller, leading to faster test execution) and updating the `dataclass-wizard` dependency to at least v0.29.2.

### Added

- add functionality to slice input data by any coordinate [\#55](https://github.com/mllam/mllam-data-prep/pull/55a)@matschreiner
- add ability to derive variables from input datasets [\#34](https://github.com/mllam/mllam-data-prep/pull/34), @ealerskans, @mafdmi
- add github PR template to guide development process on github [\#44](https://github.com/mllam/mllam-data-prep/pull/44), @leifdenby
- add support for zarr 3.0.0 and above [\#51](https://github.com/mllam/mllam-data-prep/pull/51), @kashif
- warn if the user tries to load a non-YAML file [\#50](https://github.com/mllam/mllam-data-prep/pull/50), @j6k4m8
- add mypy typing action to pre-commit hooks [\#67](https://github.com/mllam/mllam-data-prep/pull/67), @observingClouds
- add support for storing config in created datasets and option to only overwrite zarr dataset of config change [\#64](https://github.com/mllam/mllam-data-prep/pull/64), @leifdenby

### Fixes

- fix bug which adds unwanted dimensions to the dataset [\#60](https://github.com/mllam/mllam-data-prep/pull/60), @ealerskans, @observingClouds
- correct chunk size estimate [\#59](https://github.com/mllam/mllam-data-prep/pull/59), @ealerskans
- fix bug arising when variables provided to derived functions are renamed [\#56](https://github.com/mllam/mllam-data-prep/pull/56), @leifdenby
- ensure config fields defaulting to `None` are typed as `Optional` and fields defaulting to `{}` are given a default-factory so that serialization with default values works correctly [\#63](https://github.com/mllam/mllam-data-prep/pull/63), @leifdenby
- fix reading of exported config files [\#67](https://github.com/mllam/mllam-data-prep/pull/67), @observingClouds

### Maintenance

- update DANRA test dataset to v0.2.0 which uses a smaller cropped domain [\#62](https://github.com/mllam/mllam-data-prep/pull/62), @leifdenby
- update `dataclass-wizard` dependency to at least v0.29.2 allowing for use of `Union` types together with check for unmatched keys in config yaml [\#73](https://github.com/mllam/mllam-data-prep/pull/73), @leifdenby


## [v0.5.0](https://github.com/mllam/mllam-data-prep/releases/tag/v0.5.0)

[All changes](https://github.com/mllam/mllam-data-prep/compare/v0.4.0...v0.5.0)

This release adds support for an optional `extra` section in the config file (for user-defined extra information that is ignored by `mllam-data-prep`) and fixes a few minor issues. Note that to use `extra` section in the config file the schema version in the config file must be increased to `v0.5.0`.

### Added

- Add optional section called `extra` to config file to allow for user-defined extra information that is ignored by `mllam-data-prep` but can be used by downstream applications. ![\#18](https://github.com/mllam/mllam-data-prep/pull/18), @leifdenby

### Changed

- remove f-string from `name_format` in config examples [\#35](https://github.com/mllam/mllam-data-prep/pull/35)
- replace global config for `dataclass_wizard` on `mllam_data_prep.config.Config` with config specific to that dataclass (to avoid conflicts with other uses of `dataclass_wizard`) [\#36](https://github.com/mllam/mllam-data-prep/pull/36)
- Schema version bumped to `v0.5.0` to match release version that supports optional `extra` section in config [\#18](https://github.com/mllam/mllam-data-prep/pull/18)


## [v0.4.0](https://github.com/mllam/mllam-data-prep/releases/tag/v0.4.0)

[All changes](https://github.com/mllam/mllam-data-prep/compare/v0.3.0...v0.4.0)

This release adds support for defining the output path in the command line
interface and addresses bugs around optional dependencies for
`dask.distributed`.

### Added

- add access to CLI via `mllam_data_prep` and add tests for CLI with/without `dask.distributed` ![\25](https://github.com/mllam/mllam-data-prep/pull/25).
- add optional output path argument to parser. ![\#26](https://github.com/mllam/mllam-data-prep/pull/26)

### Changed

- fix bug by making dependency `distributed` optional ![\#27](https://github.com/mllam/mllam-data-prep/pull/27)
- change config example to call validation split `val` instead of `validation` [\#28](https://github.com/mllam/mllam-data-prep/pull/28)
- fix typo in install dependency `distributed` ![\#20](https://github.com/mllam/mllam-data-prep/pull/20)
- add missing `psutil` requirement. [\#21](https://github.com/mllam/mllam-data-prep/pull/21).


## [v0.3.0](https://github.com/mllam/mllam-data-prep/releases/tag/v0.3.0)

[All changes](https://github.com/mllam/mllam-data-prep/compare/v0.2.0...v0.3.0)

### Added

- add support for parallel processing using `dask.distributed` with command
  line flags `--dask-distributed-local-core-fraction` and
  `--dask-distributed-local-memory-fraction` to control the number of cores and
  memory to use on the local machine.
  ![\#16](https://github.com/mllam/mllam-data-prep/pull/16)


## [v0.2.0](https://github.com/mllam/mllam-data-prep/releases/tags/v0.2.0)

[All changes](https://github.com/mllam/mllam-data-prep/compare/v0.1.0...v0.2.0)

### Added

- add support for creating dataset splits (e.g. train, validation, test)
  through `output.splitting` section in the config file, and support for
  optionally compute statistics for a given split (with
  `output.splitting.splits.{split_name}.compute_statistics`).
  ![\#28](https://github.com/mllam/mllam-data-prep/pull/10).

- include `units` and `long_name` attributes for all stacked variables as
  `{output_variable}_units` and `{output_variable}_long_name`
  ![\#11](https://github.com/mllam/mllam-data-prep/pull/11).

- include version of `mllam-data-prep` in output
  ![\#12](https://github.com/mllam/mllam-data-prep/pull/12)

### Changed

- split dataset creation and storage to zarr into separate functions
  `mllam_data_prep.create_dataset(...)` and
  `mllam_data_prep.create_dataset_zarr(...)` respectively
  ![\#7](https://github.com/mllam/mllam-data-prep/pull/7)

- changes to spec from v0.1.0:
  - the `architecture` section has been renamed `output` to make it clearer
    that this section defines the properties of the output of `mllam-data-prep`
  - `sampling_dim` removed from `output` (previously `architecture`) section of
    spec, this is not needed to create the training data
  - the variables (and their dimensions) of the output definition has been
    renamed from `architecture.input_variables` to `output.variables`
  - coordinate value ranges for the dimensions of the output (i.e. what that
    the architecture expects as input) has been renamed from
    `architecture.input_ranges` to `output.coord_ranges` to make the use more
    clear
  - selection on variable coordinates values is now set with
    `inputs.{dataset_name}.variables.{variable_name}.values` rather than
    `inputs.{dataset_name}.variables.{variable_name}.sel`
  - when dimension-mapping method `stack_variables_by_var_name` is used the
    formatting string for the new variable is now called `name_format` rather
    than `name`
  - when dimension-mapping is done by simply renaming a dimension this
    configuration now needs to be set by providing the named method (`rename`)
    explicitly through the `method` key, i.e. rather than `{to_dim}:
    {from_dim}` it is now `{to_dim}: {method: rename, dim: {from_dim}}` to
    match the signature of the other dimension-mapping methods.
  - attribute `inputs.{dataset_name}.name` attribute has been removed, with the
    key `dataset_name` this is superfluous

- relax minimuim python version requirement to `>3.8` to simplify downstream
  usage ![\#13](https://github.com/mllam/mllam-data-prep/pull/13)

## [v0.1.0](https://github.com/mllam/mllam-data-prep/releases/tag/v0.1.0)

First tagged release of `mllam-data-prep` which includes functionality to
declaratively (in a yaml-config file) describe how the variables and
coordinates of a set of zarr-based source datasets are mapped to a new set of
variables with new coordinates to single a training dataset and write this
resulting single dataset to a new zarr dataset. This explicit mapping gives the
flexibility to target different different model architectures (which may
require different inputs with different shapes between architectures).
