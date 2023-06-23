""" Tests for core_raster.py """
# pylint: disable=missing-function-docstring, bare-except


# Standard library
import os
import sys; sys.path.append("../")

# External
from osgeo import gdal

# Internal
from utils_tests import create_sample_raster
from buteo.raster.datatype import raster_set_datatype, raster_get_datatype


tmpdir = "./tests/tmp/"

def test_raster_set_datatype_int16():
    raster_path = create_sample_raster()
    dtype = 'int16'

    output_path = os.path.join(tmpdir, 'converted_01.tif')
    converted_raster = raster_set_datatype(raster_path, dtype, out_path=output_path)

    converted_ds = gdal.Open(converted_raster)
    assert converted_ds is not None, "Converted raster should be created"

    converted_dtype = gdal.GetDataTypeName(converted_ds.GetRasterBand(1).DataType)

    assert converted_dtype.lower() == dtype.lower(), f"Converted raster should have {dtype} datatype"

    converted_ds = None
    try:
        os.remove(output_path)
    except:
        pass

def test_raster_set_datatype_float32():
    raster_path = create_sample_raster()
    dtype = 'float32'

    output_path = os.path.join(tmpdir, 'converted_02.tif')
    converted_raster = raster_set_datatype(raster_path, dtype, out_path=output_path)

    converted_ds = gdal.Open(converted_raster)
    assert converted_ds is not None, "Converted raster should be created"

    converted_dtype = gdal.GetDataTypeName(converted_ds.GetRasterBand(1).DataType)
    assert converted_dtype.lower() == dtype.lower(), f"Converted raster should have {dtype} datatype"

    converted_ds = None
    try:
        os.remove(output_path)
    except:
        pass

# def test_raster_set_datatype_uint8():
#     raster_path = create_sample_raster(datatype=gdal.GDT_Float32)
#     dtype = 'byte'

#     output_path = os.path.join(tmpdir, 'converted_03.tif')
#     converted_raster = raster_set_datatype(raster_path, dtype, out_path=output_path)

#     converted_ds = gdal.Open(converted_raster)
#     assert converted_ds is not None, "Converted raster should be created"

#     converted_dtype = gdal.GetDataTypeName(converted_ds.GetRasterBand(1).DataType)
#     assert converted_dtype.lower() == dtype.lower(), f"Converted raster should have {dtype} datatype"

#     converted_ds = None
#     try:
#         os.remove(output_path)
#     except:
#         pass

def test_raster_get_datatype_single():
    raster_path = create_sample_raster(datatype=gdal.GDT_Float32)
    datatype = raster_get_datatype(raster_path)
    assert datatype.lower() == 'float32', "The returned datatype should be 'float32'"

def test_raster_get_datatype_multiple():
    raster_path1 = create_sample_raster(datatype=gdal.GDT_Float64)
    raster_path2 = create_sample_raster(datatype=gdal.GDT_Int16)

    datatypes = raster_get_datatype([raster_path1, raster_path2])
    assert datatypes == ['float64', 'int16'], "The returned datatypes should be ['float64', 'int16']"

def test_raster_get_datatype_after_conversion():
    raster_path = create_sample_raster(datatype=gdal.GDT_Float32)
    converted_raster_path = raster_set_datatype(raster_path, 'int16')

    datatype = raster_get_datatype(converted_raster_path)
    assert datatype.lower() == 'int16', "The returned datatype should be 'int16'"
