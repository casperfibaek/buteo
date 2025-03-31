"""### Tests for vector extract by attribute functions. ###"""

# Standard library
import os
from typing import Union

# External
import pytest
from osgeo import ogr

# Internal
from buteo.vector import vector_extract_by_attribute
from buteo.core_vector.core_vector_info import get_metadata_vector


def test_extract_by_attribute_single_value(test_multipolygon, temp_dir):
    """Test extracting polygons with a single attribute value."""
    out_path = os.path.join(temp_dir, "extract_single.gpkg")
    
    # Extract features where class = 'forest'
    result = vector_extract_by_attribute(
        test_multipolygon,
        attribute="class",
        values="forest",
        out_path=out_path,
    )
    
    # Check result is the expected path
    assert result == out_path
    assert os.path.exists(out_path)
    
    # The extracted vector should have features
    metadata = get_metadata_vector(out_path)
    assert metadata is not None
    assert len(metadata["layers"]) == 1
    
    # All features should have class = 'forest'
    ds = ogr.Open(out_path)
    layer = ds.GetLayer(0)
    feature_count = layer.GetFeatureCount()
    
    assert feature_count > 0
    
    for i in range(feature_count):
        feature = layer.GetNextFeature()
        assert feature.GetField("class") == "forest"


def test_extract_by_attribute_multiple_values(test_points, temp_dir):
    """Test extracting features with multiple attribute values."""
    out_path = os.path.join(temp_dir, "extract_multiple.gpkg")
    
    # Extract features where type in ('city', 'town')
    result = vector_extract_by_attribute(
        test_points,
        attribute="type",
        values=["city", "town"],
        out_path=out_path,
    )
    
    assert os.path.exists(out_path)
    
    # Verify the extraction operation
    ds = ogr.Open(out_path)
    layer = ds.GetLayer(0)
    feature_count = layer.GetFeatureCount()
    
    assert feature_count > 0
    
    for i in range(feature_count):
        feature = layer.GetNextFeature()
        assert feature.GetField("type") in ["city", "town"]


def test_extract_by_attribute_no_match(test_multipolygon, temp_dir):
    """Test extracting with a value that doesn't match any features."""
    out_path = os.path.join(temp_dir, "extract_no_match.gpkg")
    
    # Extract features with a non-existent value
    result = vector_extract_by_attribute(
        test_multipolygon,
        attribute="class",
        values="nonexistent_value",
        out_path=out_path,
    )
    
    assert os.path.exists(out_path)
    
    # Should create a file with empty layer(s)
    metadata = get_metadata_vector(out_path)
    assert metadata is not None
    
    # Layer should exist but have 0 features
    ds = ogr.Open(out_path)
    layer = ds.GetLayer(0)
    assert layer.GetFeatureCount() == 0


def test_extract_by_attribute_numeric(test_points, temp_dir):
    """Test extracting features with numeric attribute values."""
    out_path = os.path.join(temp_dir, "extract_numeric.gpkg")
    
    # Extract features where population > 10000
    result = vector_extract_by_attribute(
        test_points,
        attribute="population",
        values=[50000, 100000, 500000],  # Match specific population values
        out_path=out_path,
    )
    
    assert os.path.exists(out_path)
    
    # Check that all extracted features have the specified population values
    ds = ogr.Open(out_path)
    layer = ds.GetLayer(0)
    feature_count = layer.GetFeatureCount()
    
    for i in range(feature_count):
        feature = layer.GetNextFeature()
        assert feature.GetField("population") in [50000, 100000, 500000]


def test_extract_by_attribute_specific_layer(test_multilayer, temp_dir):
    """Test extracting from a specific layer in a multi-layer vector."""
    out_path = os.path.join(temp_dir, "extract_layer.gpkg")
    
    # Get the number of layers and check if it has multiple layers
    metadata = get_metadata_vector(test_multilayer)
    if len(metadata["layers"]) > 1:
        # Extract from only the first layer
        result = vector_extract_by_attribute(
            test_multilayer,
            attribute="name",  # Assumes first layer has a 'name' attribute
            values=["test1", "test2"],
            out_path=out_path,
            process_layer=0,  # Process only the first layer
        )
        
        assert os.path.exists(out_path)
        
        # Output should have only one layer
        out_metadata = get_metadata_vector(out_path)
        assert len(out_metadata["layers"]) == 1
    else:
        pytest.skip("Test requires multi-layer vector input")
