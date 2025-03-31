"""### Tests for vector dissolve functions. ###"""

# Standard library
import os
from typing import Union

# External
import pytest
from osgeo import ogr

# Internal
from buteo.vector import vector_dissolve
from buteo.core_vector.core_vector_info import get_metadata_vector


def test_vector_dissolve_all(test_polygon, temp_dir):
    """Test dissolving all geometries in a polygon."""
    out_path = os.path.join(temp_dir, "dissolve_all.gpkg")
    
    # Test dissolving the entire vector (no attribute)
    result = vector_dissolve(
        test_polygon,
        out_path=out_path,
    )
    
    # Check result is the expected path
    assert result == out_path
    assert os.path.exists(out_path)
    
    # The dissolved vector should have one feature
    metadata = get_metadata_vector(out_path)
    assert metadata is not None
    assert len(metadata["layers"]) == 1
    
    # Should have exactly one feature (all geometries dissolved)
    assert metadata["layers"][0]["feature_count"] == 1


def test_vector_dissolve_by_attribute(test_multipolygon, temp_dir):
    """Test dissolving polygons by attribute."""
    out_path = os.path.join(temp_dir, "dissolve_attr.gpkg")
    
    # Dissolve by the "class" attribute field
    result = vector_dissolve(
        test_multipolygon,
        attribute="class",
        out_path=out_path,
    )
    
    assert os.path.exists(out_path)
    
    # Verify the dissolve operation
    metadata = get_metadata_vector(out_path)
    assert metadata is not None
    
    # The number of features should match the number of unique class values
    orig_metadata = get_metadata_vector(test_multipolygon)
    
    # The output should have fewer features than the original
    assert metadata["layers"][0]["feature_count"] <= orig_metadata["layers"][0]["feature_count"]
    
    # Check that class field exists in the output
    assert "class" in metadata["layers"][0]["field_names"]


def test_vector_dissolve_specific_layer(test_multilayer, temp_dir):
    """Test dissolving specific layer in a multi-layer vector."""
    out_path = os.path.join(temp_dir, "dissolve_layer.gpkg")
    
    # Get the number of layers in the input
    metadata = get_metadata_vector(test_multilayer)
    if len(metadata["layers"]) > 1:
        # Test dissolving only the first layer
        result = vector_dissolve(
            test_multilayer,
            out_path=out_path,
            process_layer=0,  # Process only the first layer
        )
        
        assert os.path.exists(out_path)
        
        # Output should still have the same number of layers
        out_metadata = get_metadata_vector(out_path)
        assert len(out_metadata["layers"]) == 1
    else:
        pytest.skip("Test requires multi-layer vector input")


def test_vector_dissolve_multiple_vectors(test_polygon, test_multipolygon, temp_dir):
    """Test dissolving multiple vectors."""
    # List of vectors to dissolve
    vectors = [test_polygon, test_multipolygon]
    
    # Dissolve each vector
    results = vector_dissolve(
        vectors,
        prefix="dissolved_",
    )
    
    # Should get a list of outputs
    assert isinstance(results, list)
    assert len(results) == len(vectors)
    
    # Each output should exist
    for path in results:
        assert os.path.exists(path)
        
        # Each output should have one feature per layer (full dissolve)
        metadata = get_metadata_vector(path)
        for layer in metadata["layers"]:
            assert layer["feature_count"] == 1
