""" Tests for io.py """

# Standard library
import sys; sys.path.append("../")
import os

# External
import numpy as np
import pytest
from osgeo import gdal

# Internal
from buteo.raster import core_raster
from buteo.utils import gdal_utils

# Setup tests
FOLDER = "geometry_and_rasters/"
s2_b04 = os.path.abspath(FOLDER + "s2_b04.jp2")
s2_b04_subset = os.path.abspath(FOLDER + "s2_b04_beirut_misaligned.tif")
s2_rgb = os.path.abspath(FOLDER + "s2_tci.jp2")

vector_file = os.path.abspath(FOLDER + "beirut_city_utm36.gpkg")


def test_image_paths():
    """Meta-test: Test if image paths are correct"""
    assert os.path.isfile(s2_b04)
    assert os.path.isfile(s2_rgb)


def test_read_image():
    """Test: Read images"""
    b04 = core_raster._open_raster(s2_b04)
    tci = core_raster._open_raster(s2_rgb)

    assert isinstance(b04, gdal.Dataset)
    assert isinstance(tci, gdal.Dataset)

    with pytest.raises(Exception):
        core_raster._open_raster("not_a_file")

    with pytest.raises(Exception):
        core_raster._open_raster(vector_file)

    assert len(gdal_utils.get_gdal_memory()) == 0
    gdal_utils.clear_gdal_memory()


def test_read_multiple():
    """Test: Read multiple images"""
    rasters = [s2_b04, s2_rgb]

    # Should not be able to open multiple files with the internal version.
    with pytest.raises(Exception):
        read = core_raster._open_raster(rasters)

    read = core_raster.open_raster(rasters)
    assert isinstance(read, list)
    assert len(read) == 2
    assert isinstance(read[0], gdal.Dataset)
    assert isinstance(read[1], gdal.Dataset)

    assert len(gdal_utils.get_gdal_memory()) == 0
    gdal_utils.clear_gdal_memory()


# Start tests
def test_raster_to_array():
    """Test: Convert raster to array"""
    b04_arr = core_raster.raster_to_array(s2_b04)
    tci_arr = core_raster.raster_to_array(s2_rgb)

    assert isinstance(b04_arr, np.ndarray)
    assert isinstance(tci_arr, np.ndarray)

    assert b04_arr.shape == (1830, 1830, 1)
    assert tci_arr.shape == (1830, 1830, 3)

    assert b04_arr.dtype == np.uint16
    assert tci_arr.dtype == np.uint8

    assert len(gdal_utils.get_gdal_memory()) == 0
    gdal_utils.clear_gdal_memory()


def test_raster_to_array_multiple():
    """Test: Open multiple rasters as array(s). """
    rasters = [s2_b04, s2_rgb]
    raster_misaligned = [s2_b04_subset]
    arr = core_raster.raster_to_array(rasters)

    assert isinstance(arr, np.ndarray)
    assert arr.shape == (1830, 1830, 4)
    assert arr.dtype == np.uint16

    arr_list = core_raster.raster_to_array(rasters, stack=False)

    assert isinstance(arr_list, list)
    assert len(arr_list) == 2
    assert isinstance(arr_list[0], np.ndarray)
    assert isinstance(arr_list[1], np.ndarray)

    assert arr_list[0].shape == (1830, 1830, 1)
    assert arr_list[1].shape == (1830, 1830, 3)

    arr_list = core_raster.raster_to_array(rasters + raster_misaligned, stack=False)
    assert isinstance(arr_list, list)
    assert len(arr_list) == 3
    assert isinstance(arr_list[0], np.ndarray)
    assert isinstance(arr_list[1], np.ndarray)
    assert isinstance(arr_list[2], np.ndarray)

    assert arr_list[0].shape == (1830, 1830, 1)
    assert arr_list[1].shape == (1830, 1830, 3)
    assert arr_list[2].shape == (423, 766, 1)

    with pytest.raises(Exception):
        arr_list = core_raster.raster_to_array(rasters + raster_misaligned, stack=True)

    assert len(gdal_utils.get_gdal_memory()) == 0
    gdal_utils.clear_gdal_memory()


def test_array_to_raster():
    """Test: Convert array to raster"""
    arr = core_raster.raster_to_array(s2_b04)
    ref = s2_rgb
    ref_opened = core_raster._open_raster(ref)
    bad_ref = "/vsimem/not_a_real_path.tif"
    ref_mis = s2_b04_subset

    assert isinstance(arr, np.ndarray)
    assert isinstance(ref, str)
    assert isinstance(ref_opened, gdal.Dataset)

    converted = core_raster.array_to_raster(arr, reference=ref)
    assert isinstance(converted, str)

    converted_arr = core_raster.raster_to_array(converted)
    assert isinstance(converted_arr, np.ndarray)
    assert converted_arr.shape == arr.shape
    assert converted_arr.dtype == arr.dtype
    assert np.array_equal(converted_arr, arr)

    with pytest.raises(Exception):
        core_raster.array_to_raster(arr, reference=bad_ref)

    with pytest.raises(Exception):
        core_raster.array_to_raster(arr, reference=ref_mis)

    assert len(gdal_utils.get_gdal_memory()) == 1
    gdal_utils.clear_gdal_memory()

# raster_set_datatype
# stack_rasters
# stack_rasters_vrt
# rasters_intersect
# rasters_intersection
# get_overlap_fraction
