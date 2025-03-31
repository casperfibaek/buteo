"""### Tests for vector buffer functions. ###"""

# Standard library
import os
from typing import Union

# External
import pytest
from osgeo import ogr

# Internal
from buteo.vector import vector_buffer
from buteo.core_vector.core_vector_info import get_metadata_vector


def test_vector_buffer_fixed_distance(test_polygon, temp_dir):
    """Test buffering a polygon with a fixed distance."""
    buffer_dist = 0.01  # ~1km at equator
    
    # Test with output path
    out_path = os.path.join(temp_dir, "buffer_output.gpkg")
    result = vector_buffer(
        test_polygon,
        buffer_dist,
        out_path=out_path,
    )
    
    # Check result is the expected path - normalize both paths for comparison
    result_path = result if isinstance(result, str) else result[0]
    assert os.path.normpath(result_path) == os.path.normpath(out_path)
    assert os.path.exists(out_path)
    
    # Verify the buffer operation
    metadata = get_metadata_vector(out_path)
    assert metadata is not None
    assert len(metadata["layers"]) == 1
    
    # The buffered polygon should be larger than the original
    orig_metadata = get_metadata_vector(test_polygon)
    assert metadata["layers"][0]["area_bbox"] > orig_metadata["layers"][0]["area_bbox"]


def test_vector_buffer_attribute(test_polygon, temp_dir):
    """Test buffering a polygon using an attribute field."""
    out_path = os.path.join(temp_dir, "buffer_attribute.gpkg")
    
    # Use the "buffer_dist" attribute field
    result = vector_buffer(
        test_polygon,
        "buffer_dist",  # field name
        out_path=out_path,
    )
    
    assert os.path.exists(out_path)
    
    # Verify the buffer operation
    metadata = get_metadata_vector(out_path)
    assert metadata is not None
    assert len(metadata["layers"]) == 1
    
    # The buffered polygon should be larger than the original
    orig_metadata = get_metadata_vector(test_polygon)
    assert metadata["layers"][0]["area_bbox"] > orig_metadata["layers"][0]["area_bbox"]


def test_vector_buffer_multiple_features(test_points, temp_dir):
    """Test buffering multiple point features with a fixed distance."""
    buffer_dist = 0.02  # ~2km at equator
    
    out_path = os.path.join(temp_dir, "buffer_points.gpkg")
    result = vector_buffer(
        test_points,
        buffer_dist,
        out_path=out_path,
    )
    
    assert os.path.exists(out_path)
    
    # Verify the output contains buffered points (now polygons)
    metadata = get_metadata_vector(out_path)
    assert metadata is not None
    assert metadata["layers"][0]["geom_type_name"] in ["polygon", "multipolygon"]
    assert metadata["layers"][0]["feature_count"] == 5  # Same as input


def test_vector_buffer_in_place(test_polygon, temp_dir):
    """Test in-place buffering of a polygon."""
    # Make a copy since in-place modifications will change the original
    copy_path = os.path.join(temp_dir, "polygon_copy.gpkg")
    
    # Create a copy manually
    driver = ogr.GetDriverByName("GPKG")
    driver.CopyDataSource(ogr.Open(test_polygon), copy_path)
    
    # Buffer in-place
    buffer_dist = 0.01
    result = vector_buffer(
        copy_path,
        buffer_dist,
        in_place=True,
    )
    
    # Normalize paths for comparison
    result_path = result if isinstance(result, str) else result[0]
    assert os.path.normpath(result_path) == os.path.normpath(copy_path)
    
    # Verify the buffer operation was performed in-place
    metadata = get_metadata_vector(copy_path)
    assert metadata is not None
    
    # Compare with original to ensure it was modified
    orig_metadata = get_metadata_vector(test_polygon)
    assert metadata["layers"][0]["area_bbox"] > orig_metadata["layers"][0]["area_bbox"]


def test_vector_buffer_linestring(test_linestrings, temp_dir):
    """Test buffering a linestring."""
    buffer_dist = 0.01  # ~1km at equator
    
    out_path = os.path.join(temp_dir, "buffer_line.gpkg")
    result = vector_buffer(
        test_linestrings,
        buffer_dist,
        out_path=out_path,
    )
    
    assert os.path.exists(out_path)
    
    # Verify the output is a polygon from buffered linestring
    metadata = get_metadata_vector(out_path)
    assert metadata is not None
    assert metadata["layers"][0]["geom_type_name"] in ["polygon", "multipolygon"]
    
    # Buffered linestring should have area
    assert metadata["layers"][0]["area_bbox"] > 0


def test_vector_buffer_attribute_list(test_points, temp_dir):
    """Test buffering multiple features using attribute values."""
    out_path = os.path.join(temp_dir, "buffer_points_attr.gpkg")
    
    # Use the "buffer_dist" attribute field
    result = vector_buffer(
        test_points,
        "buffer_dist",  # each point has a different buffer distance
        out_path=out_path,
    )
    
    assert os.path.exists(out_path)
    
    # Verify we have 5 buffered points
    metadata = get_metadata_vector(out_path)
    assert metadata is not None
    assert metadata["layers"][0]["feature_count"] == 5
