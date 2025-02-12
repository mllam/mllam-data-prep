import numpy as np
from loguru import logger

# Max chunk size warning
CHUNK_MAX_SIZE_WARNING = 1 * 1024**3  # 1GB


def check_chunk_size(ds, chunks):
    """
    Check the chunk size and warn if it exceeds CHUNK_MAX_SIZE_WARNING.

    Parameters
    ----------
    ds: xr.Dataset
        Dataset to be chunked
    chunks: Dict[str, int]
        Dictionary with keys as dimensions to be chunked and
        chunk sizes as the values

    Returns
    -------
    ds: xr.Dataset
        Dataset with chunking applied
    """

    for var_name, var_data in ds.data_vars.items():
        total_chunk_size = 1

        # Loop over all dims in the dataset to be chunked
        for dim, chunk_size in chunks.items():
            chunk_dim_size = var_data.sizes.get(dim, None)
            if chunk_dim_size is None:
                continue  # Dimension 'dim' not found in the data-array
            total_chunk_size *= chunk_size

        dtype = var_data.dtype
        bytes_per_element = np.dtype(dtype).itemsize

        memory_usage = total_chunk_size * bytes_per_element

        if memory_usage > CHUNK_MAX_SIZE_WARNING:
            logger.warning(
                f"The chunk size for '{var_name}' exceeds '{CHUNK_MAX_SIZE_WARNING / 1024**3}' GB."
            )


def chunk_dataset(ds, chunks):
    """
    Check the chunk size and chunk the dataset.

    Parameters
    ----------
    ds: xr.Dataset
        Dataset to be chunked
    chunks: Dict[str, int]
        Dictionary with keys as dimensions to be chunked and
        chunk sizes as the values

    Returns
    -------
    ds: xr.Dataset
        Dataset with chunking applied
    """
    # Check the chunk size
    check_chunk_size(ds, chunks)

    # Try chunking
    try:
        ds = ds.chunk(chunks)
    except Exception as ex:
        raise Exception(f"Error chunking dataset: {ex}")

    return ds
