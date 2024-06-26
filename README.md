# mllam-data-prep

This package aims to be a *declarative* way to prepare training-data for data-driven (i.e. machine learning) weather forecasting models.
A training dataset is constructed by declaring in a yaml configuration file (for example [example.danra.yaml](example.danra.yaml)) the data sources, the variables to extract, the transformations to apply to the data, and the target variable(s) of the model architecture to map the data to.

![](docs/processing_diagram.png)

The configuration is principally a means to represent how the dimensions of a given variable in a source dataset should be mapped to the dimensions and input variables of the model architecture to be trained.

The configuration is given in yaml-format and the file specification is defined using python3 [dataclasses](https://docs.python.org/3/library/dataclasses.html) (serialised to yaml using [dataclasses-wizard](https://dataclass-wizard.readthedocs.io/en/latest/)) and defined in [mllam_data_prep/config.py](mllam_data_prep/config.py).


## Installation

To simply use `mllam-data-prep` you can install the most recent tagged version from pypi with pip:

```bash
python -m pip install mllam-data-prep
```

## Developing `mllam-data-prep`

To work on developing `mllam-data-prep` it easiest to install and manage the dependencies with [pdm](https://pdm.fming.dev/). To get started clone your fork of [the main repo](https://github.com/mllam/mllam-data-prep) locally:

```bash
git clone https://github.com/<your-github-username>/mllam-data-prep
cd mllam-data-prep
```

Use pdm to create and use a virtualenv:

```bash
pdm venv create
pdm use --venv in-project
pdm install
```

All the linting is handelled by `pre-commit` which can be setup to automatically be run on each `git commit` by installing the git commit hook:

```bash
pdm run pre-commit install
```

The branch, commit, push and make a pull-request :)


## Usage

The package is designed to be used as a command-line tool. The main command is `mllam-data-prep` which takes a configuration file as input and outputs a training dataset in the form of a `.zarr` dataset named from the config file (e.g. `example.danra.yaml` produces `example.danra.zarr`).

```bash
python -m mllam_data_prep example.danra.yaml
```

Example output:

![](docs/example_output.png)

## Configuration file

A full example configuration file is given in [example.danra.yaml](example.danra.yaml), and reproduced here for completeness:

```yaml
schema_version: v0.2.0
dataset_version: v0.1.0

architecture:
  input_variables:
    static: [grid_index, static_feature]
    state: [time, grid_index, state_feature]
    forcing: [time, grid_index, forcing_feature]
  input_coord_ranges:
    time:
      start: 1990-09-03T00:00
      end: 1990-09-09T00:00
      step: PT3H
  chunking:
    time: 1

inputs:
  danra_height_levels:
    path: https://mllam-test-data.s3.eu-north-1.amazonaws.com/height_levels.zarr
    dims: [time, x, y, altitude]
    variables:
      u:
        altitude:
          values: [100,]
          units: m
      v:
        altitude:
          values: [100, ]
          units: m
    dim_mapping:
      time:
        method: rename
        dim: time
      state_feature:
        method: stack_variables_by_var_name
        dims: [altitude]
        name_format: f"{var_name}{altitude}m"
      grid_index:
        method: stack
        dims: [x, y]
    target_architecture_variable: state

  danra_surface:
    path: https://mllam-test-data.s3.eu-north-1.amazonaws.com/single_levels.zarr
    dims: [time, x, y]
    variables:
      # shouldn't really be using sea-surface pressure as "forcing", but don't
      # have radiation varibles in danra yet
      - pres_seasurface
    dim_mapping:
      time:
        method: rename
        dim: time
      grid_index:
        method: stack
        dims: [x, y]
      forcing_feature:
        method: stack_variables_by_var_name
        name_format: f"{var_name}"
    target_architecture_variable: forcing

  danra_lsm:
    path: https://mllam-test-data.s3.eu-north-1.amazonaws.com/lsm.zarr
    dims: [x, y]
    variables:
      - lsm
    dim_mapping:
      grid_index:
        method: stack
        dims: [x, y]
      static_feature:
        method: stack_variables_by_var_name
        name_format: f"{var_name}"
    target_architecture_variable: static
```

Apart from identifies to keep track of the configuration file format version and the datasets version, the configuration file is divided into two main sections:

- `architecture`: defines the input variables and dimensions of the model architecture to be trained. These are the variables and dimensions that the inputs datasets will be mapped to.
- `inputs`: a list of source datasets to extract data from. These are the datasets that will be mapped to the architecture defined in the `architecture` section.

### The `architecture` section

```yaml
architecture:
  input_variables:
    static: [grid_index, static_feature]
    state: [time, grid_index, state_feature]
    forcing: [time, grid_index, forcing_feature]
  input_coord_ranges:
    time:
      start: 1990-09-03T00:00
      end: 1990-09-09T00:00
      step: PT3H
  chunking:
    time: 1
```

The `architecture` section defines three things:

1. `input_variables`: what input variables the model architecture you are targeting expects, and what the dimensions are for each of these variables.
2. `input_coord_ranges`: the range of values for each of the dimensions that the model architecture expects as input. These are optional, but allows you to ensure that the training dataset is created with the correct range of values for each dimension.
3. `chunking`: the chunk sizes to use when writing the training dataset to zarr. This is optional, but can be used to optimise the performance of the zarr dataset. By default the chunk sizes are set to the size of the dimension, but this can be overridden by setting the chunk size in the configuration file. A common choice is to set the dimension along which you are batching to align with the of each training item (e.g. if you are training a model with time-step roll-out of 10 timesteps, you might choose a chunksize of 10 along the time dimension).

### The `inputs` section

```yaml
inputs:
  danra_height_levels:
    path: https://mllam-test-data.s3.eu-north-1.amazonaws.com/height_levels.zarr
    dims: [time, x, y, altitude]
    variables:
      u:
        altitude:
          values: [100,]
          units: m
      v:
        altitude:
          values: [100, ]
          units: m
    dim_mapping:
      time:
        method: rename
        dim: time
      state_feature:
        method: stack_variables_by_var_name
        dims: [altitude]
        name_format: f"{var_name}{altitude}m"
      grid_index:
        method: stack
        dims: [x, y]
    target_architecture_variable: state

  danra_surface:
    path: https://mllam-test-data.s3.eu-north-1.amazonaws.com/single_levels.zarr
    dims: [time, x, y]
    variables:
      # shouldn't really be using sea-surface pressure as "forcing", but don't
      # have radiation varibles in danra yet
      - pres_seasurface
    dim_mapping:
      time:
        method: rename
        dim: time
      grid_index:
        method: stack
        dims: [x, y]
      forcing_feature:
        method: stack_variables_by_var_name
        name_format: f"{var_name}"
    target_architecture_variable: forcing

  ...
```

The `inputs` section defines the source datasets to extract data from. Each source dataset is defined by a key (e.g. `danra_height_levels`) which names the source, and the attributes of the source dataset:

- `path`: the path to the source dataset. This can be a local path or a URL to e.g. a zarr dataset or netCDF file, anything that can be read by `xarray.open_dataset(...)`.
- `dims`: the dimensions that the source dataset is expected to have. This is used to check that the source dataset has the expected dimensions and also makes it clearer in the config file what the dimensions of the source dataset are.
- `variables`: selects which variables to extract from the source dataset. This may either be a list of variable names, or a dictionary where each key is the variable name and the value defines a dictionary of coordinates to do selection on. When doing selection you may also optionally define the units of the variable to check that the units of the variable match the units of the variable in the model architecture.
- `target_architecture_variable`: the variable in the model architecture that the source dataset should be mapped to.
- `dim_mapping`: defines how the dimensions of the source dataset should be mapped to the dimensions of the model architecture. This is done by defining a method to apply to each dimension. The methods are:
  - `rename`: simply rename the dimension to the new name
  - `stack`: stack the listed dimension to create the dimension in the output
  - `stack_variables_by_var_name`: stack the dimension into the new dimension, and also stack the variable name into the new variable name. This is useful when you have multiple variables with the same dimensions that you want to stack into a single variable.
