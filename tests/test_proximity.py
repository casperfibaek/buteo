""" Tests for core_raster.py """
# pylint: disable=missing-function-docstring, bare-except


# Standard library
import os
import sys; sys.path.append("../")

# External
from osgeo import gdal
import numpy as np

# Internal
from utils_tests import create_sample_raster
from buteo.raster.proximity import raster_get_proximity

tmpdir = "./tests/tmp/"

# TODO: Write more and better tests for proximity

def test_raster_get_proximity_basic():
    raster_path = create_sample_raster()
    target_value = 1
    max_dist = 500

    output_raster_path = os.path.join(tmpdir, "example_01.tif")

    raster_get_proximity(raster_path, target_value, max_dist=max_dist, out_path=output_raster_path)

    assert os.path.exists(output_raster_path), "The output raster should be created"
    os.remove(output_raster_path)


def test_raster_get_proximity_inverted():
    raster_path = create_sample_raster()
    target_value = 1
    max_dist = 500

    output_raster_path = os.path.join(tmpdir, "example_02.tif")

    raster_get_proximity(raster_path, target_value, max_dist=max_dist, inverted=True, out_path=output_raster_path)

    assert os.path.exists(output_raster_path), "The output raster should be created"
    os.remove(output_raster_path)


def test_raster_get_proximity_list():
    raster_path1 = create_sample_raster()
    raster_path2 = create_sample_raster()
    target_value = 1
    max_dist = 500

    output_raster_path1 = os.path.join(tmpdir, "example_03.tif")
    output_raster_path2 = os.path.join(tmpdir, "example_04.tif")

    raster_get_proximity([raster_path1, raster_path2], target_value, max_dist=max_dist, out_path=[output_raster_path1, output_raster_path2])

    assert os.path.exists(output_raster_path1), "The first output raster should be created"
    assert os.path.exists(output_raster_path2), "The second output raster should be created"

    os.remove(output_raster_path1)
    os.remove(output_raster_path2)


def test_raster_get_proximity_values_basic():
    raster_path = create_sample_raster()
    target_value = 1
    max_dist = 500

    output_raster_path = os.path.join(tmpdir, "example_05.tif")

    raster_get_proximity(raster_path, target_value, max_dist=max_dist, out_path=output_raster_path)

    output_raster = gdal.Open(output_raster_path)
    output_band = output_raster.GetRasterBand(1)
    output_array = output_band.ReadAsArray()

    assert np.any(output_array > 0), "There should be some non-zero values in the proximity raster"

    output_raster = None
    os.remove(output_raster_path)


# TODO: Is this test correct?
def test_raster_get_proximity_values_inverted():
    raster_path = create_sample_raster()
    target_value = 0
    max_dist = 500

    output_raster_path = os.path.join(tmpdir, "example_06.tif")

    raster_get_proximity(raster_path, target_value, max_dist=max_dist, inverted=True, out_path=output_raster_path)

    output_raster = gdal.Open(output_raster_path)
    output_band = output_raster.GetRasterBand(1)
    output_array = output_band.ReadAsArray()

    assert np.any(output_array == 0), "There should be some non-zero values in the inverted proximity raster"

    output_raster = None
    os.remove(output_raster_path)


def test_raster_get_proximity_values_with_border():
    raster_path = create_sample_raster()
    target_value = 10000
    max_dist = 500
    add_border = True
    border_value = 0

    output_raster_path = os.path.join(tmpdir, "example_07.tif")

    raster_get_proximity(raster_path, target_value, max_dist=max_dist, add_border=add_border, border_value=border_value, out_path=output_raster_path)

    output_raster = gdal.Open(output_raster_path)
    output_band = output_raster.GetRasterBand(1)
    output_array = output_band.ReadAsArray()

    assert np.all(output_array == max_dist)

    output_raster = None
    os.remove(output_raster_path)
