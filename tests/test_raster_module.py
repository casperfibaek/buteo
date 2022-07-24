""" Tests for core_raster.py """


# Standard library
import sys; sys.path.append("../")
import os

# External
import pytest

# Internal
from buteo.raster import core_raster
from buteo.raster.reproject import _reproject_raster, reproject_raster
from buteo.raster.resample import _resample_raster
from buteo.utils import gdal_utils


# Setup tests
FOLDER = "./geometry_and_rasters/"
s2_b04 = os.path.abspath(os.path.join(FOLDER, "s2_b04.jp2"))
s2_b04_subset = os.path.abspath(os.path.join(FOLDER, "s2_b04_beirut_misaligned.tif"))
s2_b04_faraway = os.path.abspath(os.path.join(FOLDER, "s2_b04_baalbeck.tif"))
s2_rgb = os.path.abspath(os.path.join(FOLDER, "s2_tci.jp2"))
s2_glob = os.path.abspath(os.path.join(FOLDER, "s2_b04*.tif:glob"))

vector_file = os.path.abspath(os.path.join(FOLDER, "beirut_city_utm36.gpkg"))
wgs84_file = os.path.abspath(os.path.join(FOLDER, "beirut_airport_wgs84.gpkg"))


def test_reproject_raster():
    """ Tests: Reprojections of rasters. """
    mem_before = len(gdal_utils.get_gdal_memory())

    assert not gdal_utils.projections_match(s2_b04, wgs84_file)

    reprojected = _reproject_raster(s2_b04, wgs84_file, suffix="_reprojected")

    assert gdal_utils.projections_match(reprojected, wgs84_file)

    reprojected = gdal_utils.delete_if_in_memory(reprojected)

    with pytest.raises(Exception):
        reprojected_multiple = _reproject_raster(s2_glob, wgs84_file, suffix="_mult", add_uuid=True)

    reprojected_multiple = reproject_raster(s2_glob, wgs84_file, suffix="_mult", add_uuid=True)

    for element in reprojected_multiple:
        assert gdal_utils.projections_match(element, wgs84_file)

    reprojected_multiple = gdal_utils.delete_if_in_memory_list(reprojected_multiple)

    mem_after = len(gdal_utils.get_gdal_memory())
    assert mem_before == mem_after


def test_resample_raster():
    """ Tests: Resampling of rasters. """
    mem_before = len(gdal_utils.get_gdal_memory())

    og_size = core_raster._raster_to_metadata(s2_b04)["size"]
    og_pixel_width = core_raster._raster_to_metadata(s2_b04)["pixel_width"]

    assert og_size == [1830, 1830]

    resampled_pixels = _resample_raster(s2_b04, og_size[0] // 2, target_in_pixels=True, suffix="_resampled_pixels")
    ta_size = core_raster._raster_to_metadata(resampled_pixels)["size"]
    resampled_pixels = gdal_utils.delete_if_in_memory(resampled_pixels)

    assert ta_size == [og_size[0] // 2, og_size[1] // 2]

    resampled_width = _resample_raster(s2_b04, og_pixel_width / 2, suffix="_resampled_width")
    ta_size_width = core_raster._raster_to_metadata(resampled_width)["pixel_width"]
    ta_size = core_raster._raster_to_metadata(resampled_width)["size"]

    assert og_pixel_width / 2 == ta_size_width
    assert ta_size == [og_size[0] * 2, og_size[1] * 2]

    resampled_width = gdal_utils.delete_if_in_memory(resampled_width)

    mem_after = len(gdal_utils.get_gdal_memory())
    assert mem_before == mem_after
