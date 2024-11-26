# pylint: skip-file
# type: ignore

# Standard library
import sys; sys.path.append("../../")

import pytest
import numpy as np
from osgeo import gdal
from buteo.utils.utils_translate import (
    _get_available_drivers,
    _get_valid_raster_driver_extensions,
    _get_valid_vector_driver_extensions,
    _check_is_valid_driver_extension,
    _translate_resample_method,
    _translate_dtype_gdal_to_numpy,
    _translate_dtype_numpy_to_gdal,
    _get_default_nodata_value,
    _get_range_for_numpy_datatype,
    _check_is_value_within_dtype_range,
    _check_is_gdal_dtype_float,
    _check_is_gdal_dtype_int,
    _parse_dtype,
    _check_is_int_numpy_dtype,
    _check_is_float_numpy_dtype,
    _safe_numpy_casting,
)

# Fixtures
@pytest.fixture
def sample_drivers():
    return _get_available_drivers()

@pytest.fixture
def sample_array():
    return np.array([1, 2, 3, 4, 5], dtype=np.int32)

# Driver Tests
def test_get_available_drivers():
    raster_drivers, vector_drivers = _get_available_drivers()
    assert isinstance(raster_drivers, list)
    assert isinstance(vector_drivers, list)
    assert len(raster_drivers) > 0
    assert len(vector_drivers) > 0

def test_valid_raster_extensions():
    extensions = _get_valid_raster_driver_extensions()
    assert isinstance(extensions, list)
    assert "tif" in extensions

def test_valid_vector_extensions():
    extensions = _get_valid_vector_driver_extensions()
    assert isinstance(extensions, list)
    assert "shp" in extensions

def test_check_valid_driver_extension():
    assert _check_is_valid_driver_extension("tif") == True
    assert _check_is_valid_driver_extension("invalid") == False
    assert _check_is_valid_driver_extension("") == False
    assert _check_is_valid_driver_extension(None) == False

# Resample Method Tests
def test_translate_resample_method():
    assert _translate_resample_method("nearest") == gdal.GRA_NearestNeighbour
    assert _translate_resample_method("bilinear") == gdal.GRA_Bilinear
    with pytest.raises(ValueError):
        _translate_resample_method("invalid")

# Datatype Translation Tests
def test_translate_dtype_gdal_to_numpy():
    assert _translate_dtype_gdal_to_numpy(gdal.GDT_Byte) == np.dtype("uint8")
    assert _translate_dtype_gdal_to_numpy(gdal.GDT_Float32) == np.dtype("float32")
    with pytest.raises(TypeError):
        _translate_dtype_gdal_to_numpy("invalid")

def test_translate_dtype_numpy_to_gdal():
    assert _translate_dtype_numpy_to_gdal(np.dtype("uint8")) == gdal.GDT_Byte
    assert _translate_dtype_numpy_to_gdal("float32") == gdal.GDT_Float32
    with pytest.raises(ValueError):
        _translate_dtype_numpy_to_gdal("invalid")

# NoData Tests
def test_get_default_nodata_value():
    assert _get_default_nodata_value("int16") == -32767
    assert _get_default_nodata_value("float32") == -9999.0
    with pytest.raises(ValueError):
        _get_default_nodata_value("invalid")

# Range Tests
def test_get_range_for_numpy_datatype():
    assert _get_range_for_numpy_datatype("uint8") == (0, 255)
    assert _get_range_for_numpy_datatype("int16") == (-32768, 32767)
    with pytest.raises(ValueError):
        _get_range_for_numpy_datatype("invalid")

def test_check_is_value_within_dtype_range():
    assert _check_is_value_within_dtype_range(127, "int8") == True
    assert _check_is_value_within_dtype_range(256, "uint8") == False
    assert _check_is_value_within_dtype_range(np.nan, "float32") == True

# GDAL Dtype Tests
def test_check_is_gdal_dtype_float():
    assert _check_is_gdal_dtype_float(gdal.GDT_Float32) == True
    assert _check_is_gdal_dtype_float(gdal.GDT_Byte) == False

def test_check_is_gdal_dtype_int():
    assert _check_is_gdal_dtype_int(gdal.GDT_Byte) == True
    assert _check_is_gdal_dtype_int(gdal.GDT_Float32) == False

# Dtype Parsing Tests
def test_parse_dtype():
    assert _parse_dtype("int32") == np.dtype("int32")
    assert _parse_dtype(np.int32) == np.dtype("int32")
    with pytest.raises(ValueError):
        _parse_dtype("invalid")

def test_check_is_int_numpy_dtype():
    assert _check_is_int_numpy_dtype("int32") == True
    assert _check_is_int_numpy_dtype("float32") == False

def test_check_is_float_numpy_dtype():
    assert _check_is_float_numpy_dtype("float32") == True
    assert _check_is_float_numpy_dtype("int32") == False

# Array Casting Tests
def test_safe_numpy_casting(sample_array):
    result = _safe_numpy_casting(sample_array, "float32")
    assert result.dtype == np.dtype("float32")
    
    result = _safe_numpy_casting(sample_array, "uint8")
    assert result.dtype == np.dtype("uint8")
    assert np.all(result <= 255)

    with pytest.raises(TypeError):
        _safe_numpy_casting([1, 2, 3], "int32")