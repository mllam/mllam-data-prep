from pathlib import Path

from dask.diagnostics import ProgressBar

from .create_dataset import create_dataset_zarr

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to the config file", type=Path)
    parser.add_argument(
        "--show-progress", help="Show progress bar", action="store_true"
    )
    args = parser.parse_args()

    if args.show_progress:
        ProgressBar().register()

    create_dataset_zarr(fp_config=args.config)
