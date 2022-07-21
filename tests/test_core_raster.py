# """ Tests for core_raster.py """


# # Standard library
# import sys; sys.path.append("../")
# import os

# # External
# import numpy as np
# import pytest
# from osgeo import gdal, ogr

# # Internal
# from buteo.raster import core_raster
# from buteo.utils import gdal_utils

# # Setup tests
# FOLDER = "geometry_and_rasters/"
# s2_b04 = os.path.abspath(FOLDER + "s2_b04.jp2")
# s2_b04_subset = os.path.abspath(FOLDER + "s2_b04_beirut_misaligned.tif")
# s2_b04_faraway = os.path.abspath(FOLDER + "s2_b04_baalbeck.tif")
# s2_rgb = os.path.abspath(FOLDER + "s2_tci.jp2")

# vector_file = os.path.abspath(FOLDER + "beirut_city_utm36.gpkg")


# def test_image_paths():
#     """Meta-test: Test if image paths are correct"""
#     assert os.path.isfile(s2_b04)
#     assert os.path.isfile(s2_rgb)


# def test_read_image():
#     """Test: Read images"""
#     mem_before = len(gdal_utils.get_gdal_memory())
#     b04 = core_raster._open_raster(s2_b04)
#     tci = core_raster._open_raster(s2_rgb)

#     assert isinstance(b04, gdal.Dataset)
#     assert isinstance(tci, gdal.Dataset)

#     with pytest.raises(Exception):
#         core_raster._open_raster("not_a_file")

#     with pytest.raises(Exception):
#         core_raster._open_raster(vector_file)

#     mem_after = len(gdal_utils.get_gdal_memory())
#     assert mem_before == mem_after


# def test_read_multiple():
#     """Test: Read multiple images"""
#     mem_before = len(gdal_utils.get_gdal_memory())

#     rasters = [s2_b04, s2_rgb]

#     # Should not be able to open multiple files with the internal version.
#     with pytest.raises(Exception):
#         read = core_raster._open_raster(rasters)

#     read = core_raster.open_raster(rasters)
#     assert isinstance(read, list)
#     assert len(read) == 2
#     assert isinstance(read[0], gdal.Dataset)
#     assert isinstance(read[1], gdal.Dataset)

#     mem_after = len(gdal_utils.get_gdal_memory())
#     assert mem_before == mem_after


# # Start tests
# def test_raster_to_array():
#     """Test: Convert raster to array"""
#     mem_before = len(gdal_utils.get_gdal_memory())
#     b04_arr = core_raster.raster_to_array(s2_b04)
#     tci_arr = core_raster.raster_to_array(s2_rgb)

#     assert isinstance(b04_arr, np.ndarray)
#     assert isinstance(tci_arr, np.ndarray)

#     assert b04_arr.shape == (1830, 1830, 1)
#     assert tci_arr.shape == (1830, 1830, 3)

#     assert b04_arr.dtype == np.uint16
#     assert tci_arr.dtype == np.uint8

#     mem_after = len(gdal_utils.get_gdal_memory())
#     assert mem_before == mem_after


# def test_raster_to_array_multiple():
#     """Test: Open multiple rasters as array(s). """
#     mem_before = len(gdal_utils.get_gdal_memory())
#     rasters = [s2_b04, s2_rgb]
#     raster_misaligned = [s2_b04_subset]
#     arr = core_raster.raster_to_array(rasters)

#     assert isinstance(arr, np.ndarray)
#     assert arr.shape == (1830, 1830, 4)
#     assert arr.dtype == np.uint16

#     arr_list = core_raster.raster_to_array(rasters, stack=False)

#     assert isinstance(arr_list, list)
#     assert len(arr_list) == 2
#     assert isinstance(arr_list[0], np.ndarray)
#     assert isinstance(arr_list[1], np.ndarray)

#     assert arr_list[0].shape == (1830, 1830, 1)
#     assert arr_list[1].shape == (1830, 1830, 3)

#     arr_list = core_raster.raster_to_array(rasters + raster_misaligned, stack=False)
#     assert isinstance(arr_list, list)
#     assert len(arr_list) == 3
#     assert isinstance(arr_list[0], np.ndarray)
#     assert isinstance(arr_list[1], np.ndarray)
#     assert isinstance(arr_list[2], np.ndarray)

#     assert arr_list[0].shape == (1830, 1830, 1)
#     assert arr_list[1].shape == (1830, 1830, 3)
#     assert arr_list[2].shape == (423, 766, 1)

#     with pytest.raises(Exception):
#         arr_list = core_raster.raster_to_array(rasters + raster_misaligned, stack=True)

#     mem_after = len(gdal_utils.get_gdal_memory())
#     assert mem_before == mem_after


# def test_array_to_raster():
#     """Test: Convert array to raster"""
#     mem_before = len(gdal_utils.get_gdal_memory())
#     arr = core_raster.raster_to_array(s2_b04)
#     ref = s2_rgb
#     ref_opened = core_raster._open_raster(ref)
#     bad_ref = "/vsimem/not_a_real_path.tif"
#     ref_mis = s2_b04_subset

#     assert isinstance(arr, np.ndarray)
#     assert isinstance(ref, str)
#     assert isinstance(ref_opened, gdal.Dataset)

#     converted = core_raster.array_to_raster(arr, reference=ref)
#     assert isinstance(converted, str)

#     converted_arr = core_raster.raster_to_array(converted)
#     assert isinstance(converted_arr, np.ndarray)
#     assert converted_arr.shape == arr.shape
#     assert converted_arr.dtype == arr.dtype
#     assert np.array_equal(converted_arr, arr)

#     with pytest.raises(Exception):
#         core_raster.array_to_raster(arr, reference=bad_ref)

#     with pytest.raises(Exception):
#         core_raster.array_to_raster(arr, reference=ref_mis)

#     gdal_utils.delete_if_in_memory(converted)

#     mem_after = len(gdal_utils.get_gdal_memory())
#     assert mem_before == mem_after


# def test_raster_set_datatype():
#     """Test: Set datatype of raster"""
#     mem_before = len(gdal_utils.get_gdal_memory())
#     arr = core_raster.raster_to_array(s2_b04)
#     to_float = core_raster.raster_set_datatype(s2_b04, "float32")
#     to_float_meta = core_raster.raster_to_metadata(to_float)

#     assert to_float_meta["datatype"] == "float32"
#     assert arr.shape == to_float_meta["shape"]

#     gdal_utils.delete_if_in_memory(to_float)

#     mem_after = len(gdal_utils.get_gdal_memory())
#     assert mem_before == mem_after


# def test_stack_rasters():
#     """Test: Stack rasters"""
#     mem_before = len(gdal_utils.get_gdal_memory())
#     rasters = [s2_b04, s2_rgb]
#     stacked = core_raster.stack_rasters(rasters)

#     assert isinstance(stacked, str)
#     assert stacked.endswith(".tif")

#     stacked_arr = core_raster.raster_to_array(stacked)
#     assert isinstance(stacked_arr, np.ndarray)
#     assert stacked_arr.shape == (1830, 1830, 4)
#     assert stacked_arr.dtype == np.uint16

#     gdal_utils.delete_if_in_memory(stacked)

#     mem_after = len(gdal_utils.get_gdal_memory())
#     assert mem_before == mem_after


# def test_stack_rasters_vrt():
#     """Test: Stack rasters using VRT"""
#     mem_before = len(gdal_utils.get_gdal_memory())
#     rasters = [s2_b04, s2_rgb]
#     stacked = core_raster.stack_rasters_vrt(rasters, "/vsimem/stacked.vrt", seperate=True)

#     assert isinstance(stacked, str)
#     assert stacked.endswith(".vrt")

#     stacked_arr = core_raster.raster_to_array(stacked)

#     assert isinstance(stacked_arr, np.ndarray)
#     assert stacked_arr.shape[:2] == (1830, 1830) # Only the first band is processed
#     assert stacked_arr.shape[2] == 2
#     assert stacked_arr.dtype == np.uint16

#     gdal_utils.delete_if_in_memory(stacked)

#     mem_after = len(gdal_utils.get_gdal_memory())
#     assert mem_before == mem_after


# def test_rasters_intersect():
#     """Test: Rasters intersect"""
#     mem_before = len(gdal_utils.get_gdal_memory())
#     raster1 = s2_b04
#     raster2 = s2_rgb
#     raster3 = s2_b04_subset
#     raster4 = s2_b04_faraway

#     assert core_raster.rasters_intersect(raster1, raster2)
#     assert core_raster.rasters_intersect(raster1, raster3)
#     assert core_raster.rasters_intersect(raster2, raster3)
#     assert core_raster.rasters_intersect(raster1, raster4)
#     assert core_raster.rasters_intersect(raster2, raster4)
#     assert not core_raster.rasters_intersect(raster3, raster4)

#     mem_after = len(gdal_utils.get_gdal_memory())
#     assert mem_before == mem_after


# def test_raster_intersection():
#     """Test: Get intersection of rasters """
#     mem_before = len(gdal_utils.get_gdal_memory())
#     raster1 = s2_b04
#     raster2 = s2_rgb
#     raster3 = s2_b04_subset
#     raster4 = s2_b04_faraway

#     geom1 = core_raster.rasters_intersection(raster1, raster2)
#     geom2 = core_raster.rasters_intersection(raster1, raster3)

#     assert isinstance(geom1, ogr.DataSource)
#     assert isinstance(geom2, ogr.DataSource)

#     with pytest.raises(Exception):
#         core_raster.rasters_intersection(raster3, raster4)

#     mem_after = len(gdal_utils.get_gdal_memory())
#     assert mem_before == mem_after


# def test_overlap_fraction():
#     """ Gets the fraction of overlap between two rasters """
#     mem_before = len(gdal_utils.get_gdal_memory())
#     raster1 = s2_b04
#     raster2 = s2_rgb
#     raster3 = s2_b04_subset
#     raster4 = s2_b04_faraway

#     overlap1 = core_raster.get_overlap_fraction(raster1, raster2)
#     overlap2 = core_raster.get_overlap_fraction(raster1, raster3)
#     overlap3 = core_raster.get_overlap_fraction(raster1, raster4)
#     overlap4 = core_raster.get_overlap_fraction(raster3, raster4)

#     assert overlap1 == 1.0
#     assert overlap2 < 0.1
#     assert overlap3 < 0.01
#     assert overlap4 == 0.0

#     mem_after = len(gdal_utils.get_gdal_memory())
#     assert mem_before == mem_after
