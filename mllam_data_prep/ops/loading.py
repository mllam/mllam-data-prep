import xarray as xr


def load_input_dataset(fp):
    """
    Load the dataset

    Parameters
    ----------
    fp : str
        Filepath to the source dataset, for example the path to a zarr dataset
        or a netCDF file (anything that is supported by `xarray.open_dataset` will work)

    Returns
    -------
    ds: xr.Dataset
        Source dataset
    """

    try:
        ds = xr.open_zarr(fp)
    except ValueError:
        ds = xr.open_dataset(fp)

    return ds
