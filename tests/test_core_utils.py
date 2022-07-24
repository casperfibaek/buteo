""" Tests for core_utils.py """

# Standard library
import sys; sys.path.append("../")
import os

# External
import pytest

# Internal
from buteo.utils import core_utils


def test_folder_exists():
    """Test: Check if folder exists"""
    folder1 = "./geometry_and_rasters/"
    folder2 = "./geometry_and_rasters/test.tif"
    folder3 = "./frog_paintings/"
    assert core_utils.folder_exists(folder1)
    assert not core_utils.folder_exists(folder2)
    assert not core_utils.folder_exists(folder3)


def test_is_list_all_the_same():
    """Test: Is the folder all the same values? """
    arr1 = [1,1,1,1,1]
    arr2 = [1,1,1,2,1]
    arr3 = [1,1,1,1,"1"]
    arr4 = [1,1,1,1,[0]]

    assert core_utils.is_list_all_the_same(arr1)
    assert not core_utils.is_list_all_the_same(arr2)
    assert not core_utils.is_list_all_the_same(arr3)
    assert not core_utils.is_list_all_the_same(arr4)


def test_path_to_ext():
    """Test: Get the extension of a path"""
    path1 = "./tests/geometry_and_rasters/test.tif"
    path2 = "test.tif"
    path3 = "./bob/test"

    assert core_utils.path_to_ext(path1) == "tif"
    assert core_utils.path_to_ext(path2) == "tif"
    with pytest.raises(Exception):
        core_utils.path_to_ext(path3)


def test_path_to_folder():
    """Test: Get the folder of a path"""
    path1 = "./geometry_and_rasters/test.tif"
    path2 = "test.tif"

    assert core_utils.path_to_folder(path1) == os.path.abspath("./geometry_and_rasters/")
    assert core_utils.path_to_folder(path2) == os.path.abspath("./")


def test_get_augmented_path():
    """Test: Path augmentation """
    path1 = "./tests/geometry_and_rasters/img1.tif"
    path2 = "img2.tif"
    path3 = "./bob/img3.jp2"
    path4 = "./img4.tif"
    folder = "./geometry_and_rasters/"

    aug1 = core_utils.get_augmented_path(path1, prefix="1_", suffix="_1", add_uuid=False, folder=folder)
    aug2 = core_utils.get_augmented_path(path2, prefix="2_", suffix="_2", add_uuid=False, folder=None)
    aug3 = core_utils.get_augmented_path(path3, prefix="3_", suffix="_3", add_uuid=False, folder=folder)
    aug4 = core_utils.get_augmented_path(path4, prefix="4_", suffix="_4", add_uuid=False, folder=None)

    assert os.path.basename(aug1) == '1_img1_1.tif'
    assert os.path.basename(aug2) == '2_img2_2.tif'
    assert os.path.basename(aug3) == '3_img3_3.jp2'
    assert os.path.basename(aug4) == '4_img4_4.tif'
