""" Tests for raster/convolution.py """
# pylint: disable=missing-function-docstring

# Standard library
import sys; sys.path.append("../")

# External
import numpy as np
import pytest

# Internal
from buteo.raster import core_io
from utils_tests import create_sample_raster


def test_raster_to_array_shape_dtype():
    raster_path = create_sample_raster(width=10, height=10, bands=1)
    array = core_io.raster_to_array(raster_path)

    assert array.shape == (10, 10, 1)
    assert array.dtype == np.uint8


def test_raster_to_array_multiple_bands():
    raster_path = create_sample_raster(width=10, height=10, bands=3)
    array = core_io.raster_to_array(raster_path, bands='all')

    assert array.shape == (10, 10, 3)


def test_raster_to_array_invalid_pixel_offsets():
    raster_path = create_sample_raster(width=10, height=10, bands=1)
    with pytest.raises(ValueError):
        core_io.raster_to_array(raster_path, pixel_offsets=(-1, 0, 10, 10))


def test_raster_to_array_invalid_extent():
    raster_path = create_sample_raster(width=10, height=10, bands=1)
    with pytest.raises(ValueError):
        core_io.raster_to_array(raster_path, bbox=[-1, 0, 10, 11])

def test_raster_to_array_filled():
    raster_path = create_sample_raster(width=10, height=10, bands=1, nodata=255)
    array_filled = core_io.raster_to_array(raster_path, filled=True)
    array_not_filled = core_io.raster_to_array(raster_path, filled=False)

    assert np.ma.isMaskedArray(array_not_filled)
    assert not np.ma.isMaskedArray(array_filled)
    assert array_filled.dtype == np.uint8
    assert array_not_filled.dtype == np.uint8

def test_raster_to_array_custom_fill_value():
    raster_path = create_sample_raster(width=10, height=10, bands=1, nodata=255)
    array = core_io.raster_to_array(raster_path, filled=True, fill_value=128)

    assert not np.ma.isMaskedArray(array)
    assert array.dtype == np.uint8
    assert np.count_nonzero(array == 128) >= 1

def test_raster_to_array_cast_dtype():
    raster_path = create_sample_raster(width=10, height=10, bands=1)
    array = core_io.raster_to_array(raster_path, cast=np.int16)

    assert array.shape == (10, 10, 1)
    assert array.dtype == np.int16

def test_raster_to_array_cast_dtype():
    raster_path = create_sample_raster(width=10, height=10, bands=1)
    array = core_io.raster_to_array(raster_path, cast=np.int16)

    assert array.shape == (10, 10, 1)
    assert array.dtype == np.int16

def test_raster_to_array_channel_last():
    raster_path = create_sample_raster(width=10, height=10, bands=3)
    array_channel_last = core_io.raster_to_array(raster_path, bands='all', channel_last=True)
    array_channel_first = core_io.raster_to_array(raster_path, bands='all', channel_last=False)

    assert array_channel_last.shape == (10, 10, 3)
    assert array_channel_first.shape == (3, 10, 10)

def test_raster_to_array_multiple_rasters():
    raster_path1 = create_sample_raster(width=10, height=10, bands=2)
    raster_path2 = create_sample_raster(width=10, height=10, bands=3)
    array = core_io.raster_to_array([raster_path1, raster_path2], bands='all')

    assert array.shape == (10, 10, 5)
