""" Tests for core_raster.py """
# pylint: disable=missing-function-docstring, bare-except


# Standard library
import os
import sys; sys.path.append("../")

# External
import numpy as np
from osgeo import gdal

# Internal
from utils_tests import create_sample_raster
from buteo.raster import core_raster_io
from buteo.raster.shift import raster_shift, raster_shift_pixel

tmpdir = "./tests/tmp/"

def test_raster_shift():
    raster_path = create_sample_raster()
    shift_list = [5, -5]

    output_raster_path = os.path.join(tmpdir, "shift_01.tif")

    raster_shift(raster_path, shift_list, out_path=output_raster_path)

    input_raster = gdal.Open(raster_path)
    output_raster = gdal.Open(output_raster_path)

    input_geotransform = input_raster.GetGeoTransform()
    output_geotransform = output_raster.GetGeoTransform()

    assert output_geotransform[0] == input_geotransform[0] + shift_list[0], "X origin of shifted raster should match input raster origin plus shift"
    assert output_geotransform[3] == input_geotransform[3] + shift_list[1], "Y origin of shifted raster should match input raster origin plus shift"

    input_raster = None
    output_raster = None

    try:
        os.remove(output_raster_path)
    except:
        pass

def test_raster_shift_multiple_rasters():
    raster_path_1 = create_sample_raster()
    raster_path_2 = create_sample_raster()
    shift_list = [5, -5]

    output_raster_path_1 = os.path.join(tmpdir, "shift_02.tif")
    output_raster_path_2 = os.path.join(tmpdir, "shift_03.tif")

    output_rasters = raster_shift([raster_path_1, raster_path_2], shift_list, out_path=[output_raster_path_1, output_raster_path_2])

    for idx, output_raster_path in enumerate(output_rasters):
        input_raster = gdal.Open([raster_path_1, raster_path_2][idx])
        output_raster = gdal.Open(output_raster_path)

        input_geotransform = input_raster.GetGeoTransform()
        output_geotransform = output_raster.GetGeoTransform()

        assert output_geotransform[0] == input_geotransform[0] + shift_list[0], "X origin of shifted raster should match input raster origin plus shift"
        assert output_geotransform[3] == input_geotransform[3] + shift_list[1], "Y origin of shifted raster should match input raster origin plus shift"

        input_raster = None
        output_raster = None

        try:
            os.remove(output_raster_path)
        except:
            pass


def test_raster_shift_pixel():
    raster_path = create_sample_raster()
    shift_list = [5, -5]

    output_raster_path = os.path.join(tmpdir, "shift_01_pixel.tif")

    raster_shift_pixel(raster_path, shift_list, out_path=output_raster_path)

    input_raster = gdal.Open(raster_path)
    output_raster = gdal.Open(output_raster_path)

    input_geotransform = input_raster.GetGeoTransform()
    output_geotransform = output_raster.GetGeoTransform()

    assert np.allclose(input_geotransform, output_geotransform), "GeoTransform of shifted raster should match input raster"

    input_array = core_raster_io.raster_to_array(raster_path)
    output_array = core_raster_io.raster_to_array(output_raster_path)

    assert input_array.shape == output_array.shape, "Shape of shifted raster should match input raster shape"

    input_raster = None
    output_raster = None

    try:
        os.remove(output_raster_path)
    except:
        pass

def test_raster_shift_pixel_no_shift():
    raster_path = create_sample_raster()
    shift_list = [0, 0]

    output_raster_path = os.path.join(tmpdir, "shift_02_pixel.tif")

    raster_shift_pixel(raster_path, shift_list, out_path=output_raster_path)

    input_raster = gdal.Open(raster_path)
    output_raster = gdal.Open(output_raster_path)

    input_geotransform = input_raster.GetGeoTransform()
    output_geotransform = output_raster.GetGeoTransform()

    assert np.allclose(input_geotransform, output_geotransform), "GeoTransform of shifted raster should match input raster"

    input_array = core_raster_io.raster_to_array(raster_path)
    output_array = core_raster_io.raster_to_array(output_raster_path)

    assert input_array.shape == output_array.shape, "Shape of shifted raster should match input raster shape"
    assert np.allclose(input_array, output_array), "Arrays should be equal when no shift is applied"

    input_raster = None
    output_raster = None

    try:
        os.remove(output_raster_path)
    except:
        pass
