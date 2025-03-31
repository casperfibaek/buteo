"""### Tests for vector extract by location functions. ###"""

# Standard library
import os
from typing import Union

# External
import pytest
from osgeo import ogr

# Internal
from buteo.vector import vector_extract_by_location
from buteo.core_vector.core_vector_info import get_metadata_vector


def test_extract_by_location_intersects(test_polygon, test_points, temp_dir):
    """Test extracting points that intersect a polygon."""
    out_path = os.path.join(temp_dir, "extract_intersects.gpkg")
    
    # Extract points that intersect the polygon
    result = vector_extract_by_location(
        test_points,
        reference=test_polygon,
        relationship="intersects",
        out_path=out_path,
    )
    
    # Check result is the expected path
    assert result == out_path
    assert os.path.exists(out_path)
    
    # The extracted vector should have features
    metadata = get_metadata_vector(out_path)
    assert metadata is not None
    assert len(metadata["layers"]) == 1
    
    # Should have at least one feature (assuming test data has points that intersect the polygon)
    assert metadata["layers"][0]["feature_count"] > 0


def test_extract_by_location_within(test_polygon, test_points, temp_dir):
    """Test extracting points that are within a polygon."""
    out_path = os.path.join(temp_dir, "extract_within.gpkg")
    
    # Extract points that are within the polygon
    result = vector_extract_by_location(
        test_points,
        reference=test_polygon,
        relationship="within",
        out_path=out_path,
    )
    
    assert os.path.exists(out_path)
    
    # Verify the extraction operation
    metadata = get_metadata_vector(out_path)
    assert metadata is not None
    
    # Verify each point is actually within the polygon
    if metadata["layers"][0]["feature_count"] > 0:
        # Open both datasets
        points_ds = ogr.Open(out_path)
        polygon_ds = ogr.Open(test_polygon)
        
        points_layer = points_ds.GetLayer(0)
        polygon_layer = polygon_ds.GetLayer(0)
        
        # Get the polygon geometry (assuming only one polygon for simplicity)
        polygon_feat = polygon_layer.GetNextFeature()
        polygon_geom = polygon_feat.GetGeometryRef()
        
        # Check each point is within the polygon
        points_layer.ResetReading()
        for point_feat in points_layer:
            point_geom = point_feat.GetGeometryRef()
            assert point_geom.Within(polygon_geom)


def test_extract_by_location_invert(test_polygon, test_points, temp_dir):
    """Test inverting the extraction - get points NOT intersecting the polygon."""
    out_path = os.path.join(temp_dir, "extract_invert.gpkg")
    
    # Get all points that do NOT intersect with the polygon
    result = vector_extract_by_location(
        test_points,
        reference=test_polygon,
        relationship="intersects",
        invert=True,
        out_path=out_path,
    )
    
    assert os.path.exists(out_path)
    
    # Verify the inverse extraction
    metadata = get_metadata_vector(out_path)
    assert metadata is not None
    
    # Each point should NOT intersect with the polygon
    if metadata["layers"][0]["feature_count"] > 0:
        points_ds = ogr.Open(out_path)
        polygon_ds = ogr.Open(test_polygon)
        
        points_layer = points_ds.GetLayer(0)
        polygon_layer = polygon_ds.GetLayer(0)
        
        polygon_feat = polygon_layer.GetNextFeature()
        polygon_geom = polygon_feat.GetGeometryRef()
        
        points_layer.ResetReading()
        for point_feat in points_layer:
            point_geom = point_feat.GetGeometryRef()
            assert not point_geom.Intersects(polygon_geom)


def test_extract_by_location_contains(test_multipolygon, test_polygon, temp_dir):
    """Test extracting polygons that contain another polygon."""
    out_path = os.path.join(temp_dir, "extract_contains.gpkg")
    
    # Extract polygons that contain the reference polygon
    result = vector_extract_by_location(
        test_multipolygon,
        reference=test_polygon,
        relationship="contains",
        out_path=out_path,
    )
    
    assert os.path.exists(out_path)
    metadata = get_metadata_vector(out_path)
    
    # If we have results, verify the 'contains' relationship
    if metadata["layers"][0]["feature_count"] > 0:
        # Check that all extracted features actually contain the test_polygon
        source_ds = ogr.Open(out_path)
        ref_ds = ogr.Open(test_polygon)
        
        source_layer = source_ds.GetLayer(0)
        ref_layer = ref_ds.GetLayer(0)
        
        # Get reference geometry
        ref_feat = ref_layer.GetNextFeature()
        ref_geom = ref_feat.GetGeometryRef()
        
        # Check each source feature contains the reference
        source_layer.ResetReading()
        for source_feat in source_layer:
            source_geom = source_feat.GetGeometryRef()
            assert source_geom.Contains(ref_geom)


def test_extract_by_location_specific_layer(test_multilayer, test_polygon, temp_dir):
    """Test extracting from a specific layer in a multi-layer vector."""
    out_path = os.path.join(temp_dir, "extract_layer.gpkg")
    
    # Get the number of layers and check if it has multiple layers
    metadata = get_metadata_vector(test_multilayer)
    if len(metadata["layers"]) > 1:
        # Extract from only the first layer
        result = vector_extract_by_location(
            test_multilayer,
            reference=test_polygon,
            relationship="intersects",
            out_path=out_path,
            process_layer=0,  # Process only the first layer
        )
        
        assert os.path.exists(out_path)
        
        # Output should have only one layer
        out_metadata = get_metadata_vector(out_path)
        assert len(out_metadata["layers"]) == 1
    else:
        pytest.skip("Test requires multi-layer vector input")
