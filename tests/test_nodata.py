""" Tests for core_raster.py """
# pylint: disable=missing-function-docstring


# Standard library
import os
import sys; sys.path.append("../")

# External
import pytest
from osgeo import gdal

# Internal
from utils_tests import create_sample_raster
from buteo.raster.nodata import raster_has_nodata, raster_get_nodata, raster_set_nodata
from buteo.raster.metadata import raster_to_metadata

tmpdir = "./tests/tmp/"

def test_raster_has_nodata_value():
    raster_path = create_sample_raster()
    has_nodata = raster_has_nodata(raster_path)

    assert isinstance(has_nodata, bool), "The returned value should be a boolean"

def test_raster_get_nodata_value():
    raster_path = create_sample_raster()
    nodata_value = raster_get_nodata(raster_path)

    assert isinstance(nodata_value, (float, int, type(None))), "The nodata value should be of type float, int or None"

def test_raster_set_invalid_nodata():
    raster_path = create_sample_raster()
    nodata_value = -9999
    with pytest.raises(ValueError):
        raster_set_nodata(raster_path, nodata_value)

def test_raster_set_nodata():
    raster_path = create_sample_raster()
    nodata_value = 255
    raster_path = raster_set_nodata(raster_path, nodata_value, in_place=False)

    metadata = raster_to_metadata(raster_path)
    assert metadata['dtype_name'] in ['uint8', 'int16', 'uint16', 'int32', 'uint32', 'float32', 'float64'], "The dtype_name should be one of the valid data types"
    new_nodata_value = raster_get_nodata(raster_path)
    assert new_nodata_value == nodata_value, f"The nodata value should be {nodata_value}"

def test_raster_set_nodata_in_place():
    raster_path = create_sample_raster()
    nodata_value = 255
    raster_path = raster_set_nodata(raster_path, nodata_value, in_place=True)

    metadata = raster_to_metadata(raster_path)
    assert metadata['dtype_name'] in ['uint8', 'int16', 'uint16', 'int32', 'uint32', 'float32', 'float64'], "The dtype_name should be one of the valid data types"
    new_nodata_value = raster_get_nodata(raster_path)
    assert new_nodata_value == nodata_value, f"The nodata value should be {nodata_value}"

def test_raster_has_nodata_value_list():
    raster_path1 = create_sample_raster()
    raster_path2 = create_sample_raster()
    has_nodata_list = raster_has_nodata([raster_path1, raster_path2])

    assert isinstance(has_nodata_list, list), "The returned value should be a list"
    assert all(isinstance(value, bool) for value in has_nodata_list), "All values in the list should be boolean"

def test_raster_get_nodata_value_list():
    raster_path1 = create_sample_raster(datatype=gdal.GDT_Int32)
    raster_path2 = create_sample_raster(datatype=gdal.GDT_Int32)
    nodata_values_list = raster_get_nodata([raster_path1, raster_path2])

    assert isinstance(nodata_values_list, list), "The returned value should be a list"
    assert all(isinstance(value, (float, int, type(None))) for value in nodata_values_list), "All values in the list should be of type float, int or None"

def test_raster_set_nodata_list():
    raster_path1 = create_sample_raster(datatype=gdal.GDT_Float32)
    raster_path2 = create_sample_raster(datatype=gdal.GDT_Float32)
    nodata_value = -9999
    out_raster_paths = [os.path.join(tmpdir, 'test_nodata1.tif'), os.path.join(tmpdir, 'test_nodata2.tif')]

    out_raster_paths = raster_set_nodata([raster_path1, raster_path2], nodata_value, out_path=out_raster_paths, in_place=False)

    for i, out_raster_path in enumerate(out_raster_paths):
        metadata = raster_to_metadata(out_raster_path)
        assert metadata['dtype_name'] in ['uint8', 'int16', 'uint16', 'int32', 'uint32', 'float32', 'float64'], f"The dtype_name for raster {i + 1} should be one of the valid data types"
        new_nodata_value = raster_get_nodata(out_raster_path)
        assert new_nodata_value == nodata_value, f"The nodata value for raster {i + 1} should be {nodata_value}"

    for f in out_raster_paths:
        os.remove(f)

def test_raster_set_nodata_list_in_place():
    raster_path1 = create_sample_raster(datatype=gdal.GDT_Float32)
    raster_path2 = create_sample_raster(datatype=gdal.GDT_Float32)
    nodata_value = -9999

    out_raster_paths = raster_set_nodata([raster_path1, raster_path2], nodata_value, out_path=None, in_place=True)

    for i, out_raster_path in enumerate(out_raster_paths):
        metadata = raster_to_metadata(out_raster_path)
        assert metadata['dtype_name'] in ['uint8', 'int16', 'uint16', 'int32', 'uint32', 'float32', 'float64'], f"The dtype_name for raster {i + 1} should be one of the valid data types"
        new_nodata_value = raster_get_nodata(out_raster_path)
        assert new_nodata_value == nodata_value, f"The nodata value for raster {i + 1} should be {nodata_value}"
