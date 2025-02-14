# pragma: no cover

import importlib.metadata

try:  # pragma: no cover
    __version__ = importlib.metadata.version(__name__)
except importlib.metadata.PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"

# expose the public API
from .config import Config, InvalidConfigException  # noqa
from .create_dataset import create_dataset, create_dataset_zarr  # noqa
