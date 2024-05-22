# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [v0.1.0](https://github.com/mllam/mllam-data-prep/releases/tag/v0.1.0)

First tagged release of `mllam-data-prep` which includes functionality to
declaratively (in a yaml-config file) describe how the variables and
coordinates of a set of zarr-based source datasets are mapped to a new set of
variables with new coordinates to single a training dataset and write this
resulting single dataset to a new zarr dataset. This explicit mapping gives the
flexibility to target different different model architectures (which may
require different inputs with different shapes between architectures).
