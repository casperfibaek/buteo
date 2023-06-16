""" Tests for core_raster.py """
# pylint: disable=missing-function-docstring, bare-except


# Standard library
import os
import sys; sys.path.append("../")

# External
from osgeo import ogr
import pytest

# Internal
from utils_tests import create_sample_vector
from buteo.vector.buffer import vector_buffer
from buteo.vector import core_vector

tmpdir = "./tests/tmp/"


def test_vector_buffer_with_distance():
    sample_vector = create_sample_vector(geom_type="point", num_features=10)
    sample_area = core_vector._get_basic_metadata_vector(sample_vector)["area"]
    buffer_distance = 1.0
    out_path = os.path.join(tmpdir, "buffered_vector_01.gpkg")

    buffered = vector_buffer(sample_vector, buffer_distance, out_path=out_path, overwrite=True)

    # Check if output vector is created
    out_vector = ogr.Open(buffered)
    assert out_vector is not None

    # Check if the buffer operation is performed correctly
    # Here we assume that the buffer operation increases the feature count
    area_after = core_vector._get_basic_metadata_vector(out_vector)["area"]
    assert area_after > sample_area

    out_vector = None

    try:
        os.remove(out_path)
    except:
        pass

def test_vector_buffer_invalid_distance():
    sample_vector = create_sample_vector(geom_type="point", num_features=10)
    with pytest.raises(AttributeError):
        vector_buffer(sample_vector, "invalid_attribute")

def test_vector_buffer_output_path():
    sample_vector = create_sample_vector(geom_type="point", num_features=10)
    out_path = os.path.join(tmpdir, "buffered_vector.gpkg")
    vector_buffer(sample_vector, 1.0, out_path=out_path, overwrite=True)

    # Check if the output file is created
    opened = ogr.Open(out_path)
    assert opened is not None

    opened = None

    try:
        os.remove(out_path)
    except:
        pass


def test_vector_buffer_zero_distance():
    sample_vector = create_sample_vector(geom_type="point", num_features=10)
    sample_area = core_vector._get_basic_metadata_vector(sample_vector)["area"]
    out_path = os.path.join(tmpdir, "buffered_vector_zero.gpkg")

    # Buffer with zero distance, output should be the same as input
    buffered = vector_buffer(sample_vector, 0.0, out_path=out_path, overwrite=True)
    buffered_area = core_vector._get_basic_metadata_vector(buffered)["area"]

    # Check if the area remains the same
    assert buffered_area == sample_area

    try:
        os.remove(out_path)
    except:
        pass


def test_vector_buffer_with_polygon():
    # Create a sample polygon vector
    sample_polygon_vector = create_sample_vector(geom_type="polygon", num_features=10)
    sample_polygon_area = core_vector._get_basic_metadata_vector(sample_polygon_vector)["area"]
    out_path = os.path.join(tmpdir, "buffered_vector_polygon.gpkg")

    buffered = vector_buffer(sample_polygon_vector, 1.0, out_path=out_path, overwrite=True)

    # Check if output vector is created
    out_vector = ogr.Open(buffered)
    assert out_vector is not None

    # Check if the buffer operation is performed correctly
    # Here we assume that the buffer operation increases the feature count
    area_after = core_vector._get_basic_metadata_vector(out_vector)["area"]
    assert area_after > sample_polygon_area

    out_vector = None

    try:
        os.remove(out_path)
    except:
        pass


def test_vector_buffer_negative_distance():
    sample_vector = create_sample_vector(geom_type="point", num_features=10)

    with pytest.raises(AssertionError):
        vector_buffer(sample_vector, -1.0, out_path=os.path.join(tmpdir, "buffered_vector_negative.gpkg"), overwrite=True)


def test_vector_buffer_multiple_layers():
    # Create a sample vector with multiple layers
    sample_vector_multiple = create_sample_vector(geom_type="point", num_features=10, n_layers=3)
    out_path = os.path.join(tmpdir, "buffered_vector_multiple.gpkg")

    buffered = vector_buffer(sample_vector_multiple, 1.0, out_path=out_path, overwrite=True)

    # Check if output vector is created
    out_vector = ogr.Open(buffered)
    assert out_vector is not None

    # Check if all layers are present in the output
    out_metadata = core_vector._get_basic_metadata_vector(out_vector)
    assert len(out_metadata["layers"]) == 3

    out_vector = None

    try:
        os.remove(out_path)
    except:
        pass


def test_vector_buffer_overwrite():
    sample_vector = create_sample_vector(geom_type="point", num_features=10)
    sample_area = core_vector._get_basic_metadata_vector(sample_vector)["area"]

    out_path = os.path.join(tmpdir, "buffered_vector_overwrite.gpkg")

    # First buffer operation
    vector_buffer(sample_vector, 1.0, out_path=out_path, overwrite=True)

    # Second buffer operation with different buffer distance
    vector_buffer(sample_vector, 2.0, out_path=out_path, overwrite=True)

    # Check if the output file is created and the buffer distance is applied correctly
    out_vector = ogr.Open(out_path)
    assert out_vector is not None

    area_after = core_vector._get_basic_metadata_vector(out_vector)["area"]
    assert area_after > sample_area  # The area should be larger than the initial one

    out_vector = None

    try:
        os.remove(out_path)
    except:
        pass


def test_vector_buffer_without_out_path():
    sample_vector = create_sample_vector(geom_type="point", num_features=10)
    # Buffer operation without specifying an output path
    buffered = vector_buffer(sample_vector, 1.0, out_path=None, overwrite=True)

    # Check if output vector is created
    out_vector = ogr.Open(buffered)
    assert out_vector is not None
    out_vector = None
