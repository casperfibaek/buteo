""" Tests for core_raster.py """
# pylint: disable=missing-function-docstring

# Standard library
import sys; sys.path.append("../")

# External
from osgeo import gdal

# Internal
from utils_tests import create_sample_raster
from buteo.raster.borders import raster_add_border


def test_raster_add_border_default_values():
    sample_raster = create_sample_raster()
    result_raster = raster_add_border(sample_raster)

    # Open the original raster and the result raster
    original = gdal.Open(sample_raster)
    result = gdal.Open(result_raster)

    # Check if the border is added correctly
    original_width = original.RasterXSize
    original_height = original.RasterYSize
    result_width = result.RasterXSize
    result_height = result.RasterYSize

    assert result_width == original_width + 2 * 100
    assert result_height == original_height + 2 * 100


def test_raster_add_border_custom_size():
    sample_raster = create_sample_raster()
    border_size = 50
    result_raster = raster_add_border(sample_raster, border_size=border_size)

    # Open the original raster and the result raster
    original = gdal.Open(sample_raster)
    result = gdal.Open(result_raster)

    # Check if the border is added correctly
    original_width = original.RasterXSize
    original_height = original.RasterYSize
    result_width = result.RasterXSize
    result_height = result.RasterYSize

    assert result_width == original_width + 2 * border_size
    assert result_height == original_height + 2 * border_size


def test_raster_add_border_custom_value():
    sample_raster = create_sample_raster()
    border_value = 127
    result_raster = raster_add_border(sample_raster, border_value=border_value)

    # Open the result raster
    result = gdal.Open(result_raster)
    result_data = result.GetRasterBand(1).ReadAsArray()

    # Check if the border value is set correctly
    border_mask = result_data == border_value

    assert border_mask[:, :100].all()
    assert border_mask[:, -100:].all()
    assert border_mask[:100, :].all()
    assert border_mask[-100:, :].all()


def test_raster_add_border_nodata_value():
    nodata_value = -9999
    sample_raster = create_sample_raster(nodata=nodata_value, datatype=gdal.GDT_Float32)
    result_raster = raster_add_border(sample_raster, border_value=nodata_value)

    # Open the original raster and the result raster
    original = gdal.Open(sample_raster)
    result = gdal.Open(result_raster)

    # Check if the NoData value is preserved
    original_nodata = original.GetRasterBand(1).GetNoDataValue()
    result_nodata = result.GetRasterBand(1).GetNoDataValue()

    assert original_nodata == result_nodata
