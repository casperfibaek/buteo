""" Tests for core_raster.py """
# pylint: disable=missing-function-docstring


# Standard library
import os
import sys; sys.path.append("../")

# External
from osgeo import gdal

# Internal
from utils_tests import create_sample_raster
from buteo.raster.metadata import raster_to_metadata


def test_raster_to_metadata_single():
    raster_path = create_sample_raster()
    metadata = raster_to_metadata(raster_path)

    assert isinstance(metadata, dict), "The returned metadata should be a dictionary"
    assert metadata['path'] == raster_path, f"The path should be {raster_path}"
    assert metadata['basename'] == os.path.basename(raster_path), "The basename should match the raster's basename"
    assert metadata['width'] == metadata['shape'][1], "The width should match the raster's shape"

def test_raster_to_metadata_multiple():
    raster_path1 = create_sample_raster()
    raster_path2 = create_sample_raster()

    metadata_list = raster_to_metadata([raster_path1, raster_path2])
    assert isinstance(metadata_list, list), "The returned metadata should be a list"
    assert len(metadata_list) == 2, "The list should have two metadata dictionaries"

    for metadata in metadata_list:
        assert isinstance(metadata, dict), "The metadata in the list should be a dictionary"

def test_raster_to_metadata_projection():
    raster_path = create_sample_raster()
    metadata = raster_to_metadata(raster_path)

    projection_wkt = metadata['projection_wkt']
    assert isinstance(projection_wkt, str), "The projection_wkt should be a string"
    assert len(projection_wkt) > 0, "The projection_wkt should not be empty"

def test_raster_to_metadata_gdal_dataset():
    raster_path = create_sample_raster()
    dataset = gdal.Open(raster_path)
    metadata = raster_to_metadata(dataset)

    assert isinstance(metadata, dict), "The returned metadata should be a dictionary"
    assert metadata['path'] == raster_path, f"The path should be {raster_path}"
    assert metadata['width'] == dataset.RasterXSize, "The width should match the raster dataset's width"

def test_raster_to_metadata_dtype_name():
    raster_path = create_sample_raster()
    metadata = raster_to_metadata(raster_path)
    dtype_name = metadata['dtype_name']

    assert isinstance(dtype_name, str), "The dtype_name should be a string"
    assert dtype_name.lower() in ['uint8', 'int16', 'uint16', 'int32', 'uint32', 'float32', 'float64'], "The dtype_name should be one of the valid data types"

def test_raster_to_metadata_area():
    raster_path = create_sample_raster()
    metadata = raster_to_metadata(raster_path)
    area = metadata['area']

    assert isinstance(area, float), "The area should be a float"
    assert area > 0, "The area should be positive"

def test_raster_to_metadata_shape():
    raster_path = create_sample_raster()
    metadata = raster_to_metadata(raster_path)
    shape = metadata['shape']

    assert isinstance(shape, list), "The shape should be a list"
    assert len(shape) == 3, "The shape should have three elements (height, width, bands)"
    assert shape[0] == metadata['height'], "The height in the shape should match the metadata's height"
    assert shape[1] == metadata['width'], "The width in the shape should match the metadata's width"
    assert shape[2] == metadata['bands'], "The bands in the shape should match the metadata's bands"
