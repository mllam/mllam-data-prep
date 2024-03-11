# mllam-data-prep

This package aims to be a *declarative* way to prepare training-data for data-driven (i.e. machine learning) weather forecasting models.
A training dataset is constructed by declaring in a yaml configuration file (for example [example.danra.yaml](example.danra.yaml)) the data sources, the variables to extract, the transformations to apply to the data, and the target variable(s) of the model architecture to map the data to.

![](docs/processing_diagram.png)

The configuration is principally a means to represent how the dimensions of a given variable in a source dataset should be mapped to the dimensions and input variables of the model architecture to be trained.

The full configuration file specification is given in [mllam_data_prep/config/spec.py](mllam_data_prep/config/spec.py).


## Installation

The easiest way to install the package is to clone the repository and install it using pip:

```bash
git clone https://github.com/mllam/mllam-data-prep
cd mllam-data-prep
pip install .
```

## Usage

The package is designed to be used as a command-line tool. The main command is `mllam-data-prep` which takes a configuration file as input and outputs a training dataset in the form of a `.zarr` dataset named from the config file (e.g. `example.danra.yaml` produces `example.danra.zarr`).

```bash
python -m mllam_data_prep example.danra.yaml
```

Example output:

![](docs/example_output.png)
