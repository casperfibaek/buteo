""" Tests for core_raster.py """
# pylint: disable=missing-function-docstring

# Standard library
import os
import sys; sys.path.append("../")

# External
from osgeo import gdal, ogr
import numpy as np

# Internal
from utils_tests import create_sample_raster
from buteo.raster.clip import raster_clip

tmpdir = "./tests/tmp/"

def test_raster_clip_default():
    raster_path = create_sample_raster()
    clip_raster_path = create_sample_raster(width=5, height=5)

    result = raster_clip(raster=raster_path, clip_geom=clip_raster_path)
    result_ds = gdal.Open(result)

    assert result_ds.RasterXSize == 5
    assert result_ds.RasterYSize == 5


def test_raster_clip_output_path():
    raster_path = create_sample_raster()
    clip_raster_path = create_sample_raster(width=5, height=5)
    output_path = os.path.join(tmpdir, 'test_raster_clip_output_path.tif')

    try:
        result = raster_clip(raster=raster_path, clip_geom=clip_raster_path, out_path=output_path)
        result_ds = gdal.Open(result)

        assert result_ds.RasterXSize == 5
        assert result_ds.RasterYSize == 5
        assert os.path.abspath(result) == os.path.abspath(output_path)
    finally:
        result_ds = None
        if os.path.exists(output_path):
            os.remove(output_path)


def test_raster_clip_resample_alg():
    raster_path = create_sample_raster()
    clip_raster_path = create_sample_raster(width=5, height=5)
    resample_alg = "bilinear"

    result = raster_clip(raster=raster_path, clip_geom=clip_raster_path, resample_alg=resample_alg)
    result_ds = gdal.Open(result)

    assert result_ds.RasterXSize == 5
    assert result_ds.RasterYSize == 5


def test_raster_clip_multiple_rasters():
    raster_path_1 = create_sample_raster()
    raster_path_2 = create_sample_raster()
    clip_raster_path = create_sample_raster(width=5, height=5)

    result_paths = raster_clip(raster=[raster_path_1, raster_path_2], clip_geom=clip_raster_path)
    assert len(result_paths) == 2

    for result_path in result_paths:
        result_ds = gdal.Open(result_path)
        assert result_ds.RasterXSize == 5
        assert result_ds.RasterYSize == 5


def test_raster_clip_dst_nodata():
    raster_path = create_sample_raster()
    clip_raster_path = create_sample_raster(width=5, height=5)
    dst_nodata = 127

    result = raster_clip(raster=raster_path, clip_geom=clip_raster_path, dst_nodata=dst_nodata)
    result_ds = gdal.Open(result)

    nodata_value = result_ds.GetRasterBand(1).GetNoDataValue()

    assert nodata_value == dst_nodata


def test_raster_clip_no_intersection():
    raster_path = create_sample_raster()
    polygon_wkt = 'POLYGON ((-10 -10, -10 -5, -5 -5, -5 -10, -10 -10))'
    clip_geom = ogr.CreateGeometryFromWkt(polygon_wkt)

    output_path = os.path.join(tmpdir, 'clipped.tif')
    clipped_raster = raster_clip(raster_path, clip_geom, out_path=output_path)

    clipped_ds = gdal.Open(clipped_raster)
    assert clipped_ds is not None, "Clipped raster should be created even if there's no intersection"


def test_raster_clip_partial_intersection():
    raster_path = create_sample_raster()
    polygon_wkt = 'POLYGON ((3 3, 3 8, 8 8, 8 3, 3 3))'
    clip_geom = ogr.CreateGeometryFromWkt(polygon_wkt)

    output_path = os.path.join(tmpdir, 'clipped.tif')
    clipped_raster = raster_clip(raster_path, clip_geom, out_path=output_path)

    clipped_ds = gdal.Open(clipped_raster)
    assert clipped_ds is not None, "Clipped raster should be created for partial intersection"

    clipped_array = clipped_ds.GetRasterBand(1).ReadAsArray()
    assert np.count_nonzero(clipped_array) > 0, "Clipped raster should contain non-zero values for partial intersection"


def test_raster_clip_complete_intersection():
    raster_path = create_sample_raster()
    polygon_wkt = 'POLYGON ((0 0, 0 10, 10 10, 10 0, 0 0))'
    clip_geom = ogr.CreateGeometryFromWkt(polygon_wkt)

    output_path = os.path.join(tmpdir, 'clipped.tif')
    clipped_raster = raster_clip(raster_path, clip_geom, out_path=output_path)

    clipped_ds = gdal.Open(clipped_raster)
    assert clipped_ds is not None, "Clipped raster should be created for complete intersection"

    input_ds = gdal.Open(raster_path)
    input_array = input_ds.GetRasterBand(1).ReadAsArray()
    clipped_array = clipped_ds.GetRasterBand(1).ReadAsArray()

    assert np.array_equal(input_array, clipped_array), "Input and clipped rasters should have the same pixel values for complete intersection"
