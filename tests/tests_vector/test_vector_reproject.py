"""### Tests for vector reproject functions. ###"""

# Standard library
import os

# External
import pytest
from osgeo import ogr, osr

# Internal
from buteo.vector import vector_reproject
from buteo.core_vector.core_vector_info import get_metadata_vector


def test_vector_reproject(test_polygon, temp_dir):
    """Test reprojecting a vector from WGS84 to Web Mercator."""
    # Target projection: Web Mercator (EPSG:3857)
    target_epsg = 3857
    
    # Output path
    out_path = os.path.join(temp_dir, "reprojected.gpkg")
    
    # Reproject
    result = vector_reproject(
        test_polygon,
        target_epsg,
        out_path=out_path,
    )
    
    # Check result - normalize paths for comparison
    result_path = result if isinstance(result, str) else result[0]
    assert os.path.normpath(result_path) == os.path.normpath(out_path)
    assert os.path.exists(out_path)
    
    # Verify projection
    metadata = get_metadata_vector(out_path)
    
    # Get the EPSG code from the projection
    projection = metadata["layers"][0]["projection_osr"]
    assert projection is not None
    
    # Check if the output matches Web Mercator
    web_mercator = osr.SpatialReference()
    web_mercator.ImportFromEPSG(3857)
    
    assert projection.IsSame(web_mercator)


def test_vector_reproject_from_vector(test_polygon, test_linestrings, temp_dir):
    """Test reprojecting a vector using another vector as reference."""
    # First reproject the linestrings to UTM Zone 32N (EPSG:32632)
    utm_zone = 32632
    
    utm_lines_path = os.path.join(temp_dir, "utm_lines.gpkg")
    vector_reproject(
        test_linestrings,
        utm_zone,
        out_path=utm_lines_path,
    )
    
    # Now use the UTM lines as reference to reproject the polygon
    out_path = os.path.join(temp_dir, "reprojected_from_vector.gpkg")
    result = vector_reproject(
        test_polygon,
        utm_lines_path,  # Reference vector
        out_path=out_path,
    )
    
    # Check result - normalize paths for comparison
    result_path = result if isinstance(result, str) else result[0]
    assert os.path.normpath(result_path) == os.path.normpath(out_path)
    assert os.path.exists(out_path)
    
    # Verify both have the same projection
    polygon_meta = get_metadata_vector(out_path)
    lines_meta = get_metadata_vector(utm_lines_path)
    
    polygon_proj = polygon_meta["layers"][0]["projection_osr"]
    lines_proj = lines_meta["layers"][0]["projection_osr"]
    
    assert polygon_proj.IsSame(lines_proj)


def test_vector_reproject_in_memory(test_polygon):
    """Test reprojecting a vector to memory."""
    # Target projection: Web Mercator (EPSG:3857)
    target_epsg = 3857
    
    # Reproject to memory (no out_path)
    result = vector_reproject(
        test_polygon,
        target_epsg,
    )
    
    # Result should be a valid path to an in-memory dataset
    assert isinstance(result, str)
    assert result.startswith("/vsimem/") or result.endswith(".mem")
    
    # Verify projection
    metadata = get_metadata_vector(result)
    projection = metadata["layers"][0]["projection_osr"]
    
    # Check if the output matches Web Mercator
    web_mercator = osr.SpatialReference()
    web_mercator.ImportFromEPSG(3857)
    
    assert projection.IsSame(web_mercator)


def test_vector_reproject_list(test_polygon, test_points, temp_dir):
    """Test reprojecting multiple vectors."""
    # Target projection: Web Mercator (EPSG:3857)
    target_epsg = 3857
    
    # Create output directory
    out_dir = os.path.join(temp_dir, "reprojected")
    os.makedirs(out_dir, exist_ok=True)
    
    # Reproject list of vectors
    results = vector_reproject(
        [test_polygon, test_points],
        target_epsg,
        out_path=out_dir,
    )
    
    # Check results
    assert isinstance(results, list)
    assert len(results) == 2
    
    for path in results:
        assert os.path.exists(path)
        
        # Verify projection
        metadata = get_metadata_vector(path)
        projection = metadata["layers"][0]["projection_osr"]
        
        # Check if the output matches Web Mercator
        web_mercator = osr.SpatialReference()
        web_mercator.ImportFromEPSG(3857)
        
        assert projection.IsSame(web_mercator)
