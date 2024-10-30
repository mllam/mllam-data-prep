import os
from pathlib import Path

from loguru import logger

from .create_dataset import create_dataset_zarr

# Attempt to import psutil and dask.distributed modules
DASK_DISTRIBUTED_AVAILABLE = True
try:
    import psutil
    from dask.diagnostics import ProgressBar
    from dask.distributed import LocalCluster
except ImportError or ModuleNotFoundError:
    DASK_DISTRIBUTED_AVAILABLE = False

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("config", help="Path to the config file", type=Path)
    parser.add_argument(
        "-o", "--output", help="Path to the output zarr file", type=Path, default=None
    )
    parser.add_argument(
        "--show-progress", help="Show progress bar", action="store_true"
    )
    parser.add_argument(
        "--dask-distributed-local-core-fraction",
        help="Fraction of cores to use on the local machine to do multiprocessing with dask.distributed",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--dask-distributed-local-memory-fraction",
        help="Fraction of memory to use on the local machine (when doing multiprocessing with dask.distributed)",
        type=float,
        default=0.9,
    )
    args = parser.parse_args()

    if args.show_progress:
        ProgressBar().register()

    if args.dask_distributed_local_core_fraction > 0.0:
        # Only run this block if dask.distributed is available
        if not DASK_DISTRIBUTED_AVAILABLE:
            raise ModuleNotFoundError(
                "Currently dask.distributed isn't installed and therefore can't "
                "be used in mllam-data-prep. Please install the optional dependency "
                'with `python -m pip install "mllam-data-prep[dask-distributed]"`'
            )
        # get the number of system cores
        n_system_cores = os.cpu_count()
        # compute the number of cores to use
        n_local_cores = int(args.dask_distributed_local_core_fraction * n_system_cores)
        # get the total system memory
        total_memory = psutil.virtual_memory().total
        # compute the memory per worker
        memory_per_worker = (
            total_memory / n_local_cores * args.dask_distributed_local_memory_fraction
        )

        logger.info(
            f"Setting up dask.distributed.LocalCluster with {n_local_cores} cores and {memory_per_worker/1024/1024:0.0f} MB of memory per worker"
        )

        cluster = LocalCluster(
            n_workers=n_local_cores,
            threads_per_worker=1,
            memory_limit=memory_per_worker,
        )

        client = cluster.get_client()
        # print the dashboard link
        logger.info(f"Dashboard link: {cluster.dashboard_link}")

    create_dataset_zarr(fp_config=args.config, fp_zarr=args.output)
