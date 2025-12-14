import shutil
from copy import deepcopy
from unittest.mock import patch

import pytest
import xarray as xr

import mllam_data_prep.config as mdp_config
from mllam_data_prep.create_dataset import create_dataset_zarr


@pytest.fixture
def mock_config():
    return mdp_config.Config(
        schema_version="v0.6.0",
        dataset_version="v1.0.0",
        inputs={},
        output=mdp_config.Output(variables={}),
    )


@pytest.fixture
def mock_zarr_path(tmp_path):
    return tmp_path / "test.zarr"


def test_handle_existing_dataset_always_overwrite(mock_config, mock_zarr_path):
    config_path = mock_zarr_path.parent / "config.yaml"
    mock_config.to_yaml_file(config_path)
    ds = xr.Dataset(
        attrs={"mdp_version": "v0.6.0", "creation_config": mock_config.to_yaml()}
    )
    ds.to_zarr(str(mock_zarr_path))
    fn_rmtree = shutil.rmtree
    with patch("shutil.rmtree") as mock_rmtree:
        # ensure that the rmtree function is called, otherwise xarray won't be
        # able to write to the zarr path
        mock_rmtree.side_effect = fn_rmtree
        create_dataset_zarr(
            fp_config=config_path, fp_zarr=str(mock_zarr_path), overwrite="always"
        )
        mock_rmtree.assert_called_once_with(mock_zarr_path)


def test_handle_existing_dataset_never_overwrite(mock_config, mock_zarr_path):
    config_path = mock_zarr_path.parent / "config.yaml"
    mock_config.to_yaml_file(config_path)
    ds = xr.Dataset(
        attrs={"mdp_version": "v0.6.0", "creation_config": mock_config.to_yaml()}
    )
    ds.to_zarr(str(mock_zarr_path))
    with patch("shutil.rmtree") as mock_rmtree:
        with pytest.raises(FileExistsError, match="There already exists a dataset at"):
            create_dataset_zarr(
                fp_config=config_path, fp_zarr=str(mock_zarr_path), overwrite="never"
            )
        mock_rmtree.assert_not_called()


def test_handle_existing_dataset_on_config_change_same_config(
    mock_config, mock_zarr_path
):
    """
    Test that when the existing dataset has the same config as the current config, the zarr dataset is not deleted.
    """
    config_path = mock_zarr_path.parent / "config.yaml"
    mock_config.to_yaml_file(config_path)
    ds = xr.Dataset(
        attrs={"mdp_version": "v0.6.0", "creation_config": mock_config.to_yaml()}
    )
    ds.to_zarr(str(mock_zarr_path))
    with patch("shutil.rmtree") as mock_rmtree:
        create_dataset_zarr(
            fp_config=config_path,
            fp_zarr=str(mock_zarr_path),
            overwrite="on_config_change",
        )
        mock_rmtree.assert_not_called()


def test_handle_existing_dataset_on_config_change_different_config(
    mock_config, mock_zarr_path
):
    """
    Test that when the existing dataset has a different config than the current config, the zarr dataset is deleted.
    """
    different_config = deepcopy(mock_config)
    different_config.dataset_version = "2.0.0"
    config_path = mock_zarr_path.parent / "config.yaml"
    mock_config.to_yaml_file(config_path)
    ds = xr.Dataset(
        attrs={"mdp_version": "0.6.0", "creation_config": different_config.to_yaml()}
    )
    ds.to_zarr(str(mock_zarr_path))
    fn_rmtree = shutil.rmtree
    with patch("shutil.rmtree") as mock_rmtree:
        # ensure that the rmtree function is called, otherwise xarray won't be
        # able to write to the zarr path
        mock_rmtree.side_effect = fn_rmtree
        create_dataset_zarr(
            fp_config=config_path,
            fp_zarr=str(mock_zarr_path),
            overwrite="on_config_change",
        )
        mock_rmtree.assert_called_once_with(mock_zarr_path)


def test_handle_existing_dataset_older_version(mock_config, mock_zarr_path):
    """
    Test that when the existing dataset was created with an older version of mllam-data-prep, an exception is raised.
    Since for older versions we do not have the creation_config attribute, we cannot compare the configs.
    """
    config_path = mock_zarr_path.parent / "config.yaml"
    mock_config.to_yaml_file(config_path)
    ds = xr.Dataset(attrs={"mdp_version": "0.5.0"})
    ds.to_zarr(str(mock_zarr_path))
    with pytest.raises(FileExistsError, match="older version of mllam-data-prep"):
        create_dataset_zarr(
            fp_config=config_path,
            fp_zarr=str(mock_zarr_path),
            overwrite="on_config_change",
        )
