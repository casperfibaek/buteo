""" Tests for raster/convolution.py """
# pylint: disable=missing-function-docstring

# Standard library
import sys; sys.path.append("../")
import os

# External
from osgeo import gdal
import numpy as np
import pytest

# Internal
from buteo.raster import core_stack, core_io
from utils_tests import create_sample_raster



def test_raster_stack_list_single_band():
    # Create sample rasters
    raster1 = create_sample_raster(width=10, height=10, bands=1)
    raster2 = create_sample_raster(width=10, height=10, bands=1)
    raster3 = create_sample_raster(width=10, height=10, bands=1)

    # Stack rasters
    stacked_raster_path = core_stack.raster_stack_list([raster1, raster2, raster3])

    # Check output raster properties
    stacked_raster = gdal.Open(stacked_raster_path)
    assert stacked_raster is not None
    assert stacked_raster.RasterCount == 3
    assert stacked_raster.RasterXSize == 10
    assert stacked_raster.RasterYSize == 10

    # Clean up
    os.remove(stacked_raster_path)

# def test_raster_stack_list_multi_band():
#     # Create sample rasters
#     raster1 = create_sample_raster(width=10, height=10, bands=2)
#     raster2 = create_sample_raster(width=10, height=10, bands=2)

#     # Stack rasters
#     stacked_raster_path = core_stack.raster_stack_list([raster1, raster2])

#     # Check output raster properties
#     stacked_raster = gdal.Open(stacked_raster_path)
#     assert stacked_raster is not None
#     assert stacked_raster.RasterCount == 4
#     assert stacked_raster.RasterXSize == 10
#     assert stacked_raster.RasterYSize == 10

#     # Clean up
#     os.remove(stacked_raster_path)

# def test_raster_stack_list_multi_band_uneven():
#     # Create sample rasters
#     raster1 = create_sample_raster(width=10, height=10, bands=1)
#     raster2 = create_sample_raster(width=10, height=10, bands=3)

#     # Stack rasters
#     stacked_raster_path = core_stack.raster_stack_list([raster1, raster2])

#     # Check output raster properties
#     stacked_raster = gdal.Open(stacked_raster_path)
#     assert stacked_raster is not None
#     assert stacked_raster.RasterCount == 4
#     assert stacked_raster.RasterXSize == 10
#     assert stacked_raster.RasterYSize == 10

#     # Clean up
#     os.remove(stacked_raster_path)

# def test_raster_stack_list_dtype():
#     # Create sample rasters
#     raster1 = create_sample_raster(width=10, height=10, bands=1, datatype=gdal.GDT_Int16)
#     raster2 = create_sample_raster(width=10, height=10, bands=1, datatype=gdal.GDT_Int16)

#     # Stack rasters
#     stacked_raster_path = core_stack.raster_stack_list([raster1, raster2], dtype="Float32")

#     # Check output raster properties
#     stacked_raster = gdal.Open(stacked_raster_path)
#     assert stacked_raster is not None
#     assert stacked_raster.GetRasterBand(1).DataType == gdal.GDT_Float32

#     # Clean up
#     os.remove(stacked_raster_path)

# def test_raster_stack_list_raises_error_on_unaligned_rasters():
#     # Create unaligned sample rasters
#     raster1 = create_sample_raster(width=10, height=10, bands=1)
#     raster2 = create_sample_raster(width=20, height=10, bands=1)

#     with pytest.raises(AssertionError):
#         core_stack.raster_stack_list([raster1, raster2])
