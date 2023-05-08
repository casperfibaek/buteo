""" Tests for core_raster.py """
# pylint: disable=missing-function-docstring, bare-except


# Standard library
import os
import sys; sys.path.append("../")

# External
from osgeo import gdal

# Internal
from utils_tests import create_sample_raster
from buteo.raster.resample import raster_resample

tmpdir = "./tests/tmp/"


def test_raster_resample_basic():
    raster_path = create_sample_raster(width=10, height=10, pixel_height=1, pixel_width=1)
    target_size = (2, 2)
    output_raster_path = os.path.join(tmpdir, "resample_01.tif")

    raster_resample(raster_path, target_size, out_path=output_raster_path, target_in_pixels=False)

    output_raster = gdal.Open(output_raster_path)
    assert output_raster.RasterXSize == 5, "XSize of resampled raster should half original size"
    assert output_raster.RasterYSize == 5, "YSize of resampled raster should half original size"
    output_raster = None
    try:
        os.remove(output_raster_path)
    except:
        pass

def test_raster_resample_target_in_pixels():
    raster_path = create_sample_raster(width=5, height=5)
    target_size = (2, 2)
    output_raster_path = os.path.join(tmpdir, "resample_02.tif")

    raster_resample(raster_path, target_size, target_in_pixels=True, out_path=output_raster_path)

    output_raster = gdal.Open(output_raster_path)
    assert output_raster.RasterXSize == target_size[0], "XSize of resampled raster should match target size"
    assert output_raster.RasterYSize == target_size[1], "YSize of resampled raster should match target size"
    output_raster = None
    try:
        os.remove(output_raster_path)
    except:
        pass

def test_raster_resample_dtype():
    raster_path = create_sample_raster()
    target_size = (2, 2)
    dtype = "UInt16"
    output_raster_path = os.path.join(tmpdir, "resample_03.tif")

    raster_resample(raster_path, target_size, dtype=dtype, out_path=output_raster_path)

    output_raster = gdal.Open(output_raster_path)
    output_band = output_raster.GetRasterBand(1)
    assert gdal.GetDataTypeName(output_band.DataType) == dtype, "Output data type should match the specified data type"
    output_raster = None
    try:
        os.remove(output_raster_path)
    except:
        pass

def test_raster_resample_resample_alg():
    raster_path = create_sample_raster()
    target_size = (2, 2)
    resample_alg = "bilinear"
    output_raster_path = os.path.join(tmpdir, "resample_04.tif")

    raster_resample(raster_path, target_size, resample_alg=resample_alg, out_path=output_raster_path)

    # No direct assertion for resampling algorithm, but this test checks if the function runs without errors when using 'bilinear'

    try:
        os.remove(output_raster_path)
    except:
        pass

def test_raster_resample_dst_nodata():
    raster_path = create_sample_raster()
    target_size = (2, 2)
    dst_nodata = 127
    output_raster_path = os.path.join(tmpdir, "resample_05.tif")

    raster_resample(raster_path, target_size, dst_nodata=dst_nodata, out_path=output_raster_path)

    output_raster = gdal.Open(output_raster_path)
    output_band = output_raster.GetRasterBand(1)
    assert output_band.GetNoDataValue() == dst_nodata, "Output raster should have the specified NoData value"
    output_raster = None
    try:
        os.remove(output_raster_path)
    except:
        pass

def test_raster_resample_multiple_rasters():
    raster_path_1 = create_sample_raster(height=10, width=10)
    raster_path_2 = create_sample_raster(height=10, width=10)
    target_size = (2, 2)

    output_raster_path_1 = os.path.join(tmpdir, "resample_06.tif")
    output_raster_path_2 = os.path.join(tmpdir, "resample_07.tif")
    output_rasters = raster_resample([raster_path_1, raster_path_2], target_size, out_path=[output_raster_path_1, output_raster_path_2])

    for output_raster_path in output_rasters:
        output_raster = gdal.Open(output_raster_path)
        assert output_raster.RasterXSize == 5, "XSize of resampled raster should match target size"
        assert output_raster.RasterYSize == 5, "YSize of resampled raster should match target size"
        output_raster = None
        try:
            os.remove(output_raster_path)
        except:
            pass
