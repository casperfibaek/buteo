# pylint: skip-file
# type: ignore

import os
import pytest
from osgeo import ogr, osr
from buteo.core_vector.core_vector_extent import vector_to_extent

# Standard library
import sys; sys.path.append("../../")

@pytest.fixture
def simple_vector(tmp_path):
    """Create a simple vector with a single polygon."""
    vector_path = tmp_path / "simple.gpkg"
    driver = ogr.GetDriverByName('GPKG')
    ds = driver.CreateDataSource(str(vector_path))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    layer = ds.CreateLayer('test', srs, ogr.wkbPolygon)

    # Create a simple polygon
    poly_wkt = "POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))"
    poly = ogr.CreateGeometryFromWkt(poly_wkt)
    
    feature = ogr.Feature(layer.GetLayerDefn())
    feature.SetGeometry(poly)
    layer.CreateFeature(feature)
    
    ds = None
    return str(vector_path)

@pytest.fixture
def complex_vector(tmp_path):
    """Create a complex vector with multiple features."""
    vector_path = tmp_path / "complex.gpkg"
    driver = ogr.GetDriverByName('GPKG')
    ds = driver.CreateDataSource(str(vector_path))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    layer = ds.CreateLayer('test', srs, ogr.wkbPolygon)

    # Create two polygons with different extents
    poly1_wkt = "POLYGON ((0 0, 0 2, 2 2, 2 0, 0 0))"
    poly1 = ogr.CreateGeometryFromWkt(poly1_wkt)
    
    poly2_wkt = "POLYGON ((-1 -1, -1 1, 1 1, 1 -1, -1 -1))"
    poly2 = ogr.CreateGeometryFromWkt(poly2_wkt)
    
    for poly in [poly1, poly2]:
        feature = ogr.Feature(layer.GetLayerDefn())
        feature.SetGeometry(poly)
        layer.CreateFeature(feature)
    
    ds = None
    return str(vector_path)

class TestVectorToExtent:
    def test_simple_extent(self, simple_vector, tmp_path):
        """Test creating extent from simple vector."""
        out_path = str(tmp_path / "simple_extent.gpkg")
        result = vector_to_extent(simple_vector, out_path)
        
        # Check if output exists
        assert os.path.exists(result)
        
        # Check if extent is correct
        ds = ogr.Open(result)
        layer = ds.GetLayer()
        feature = layer.GetNextFeature()
        extent = feature.GetGeometryRef()
        
        # Check bounds
        env = extent.GetEnvelope()
        assert env[0] == 0  # minX
        assert env[1] == 1  # maxX
        assert env[2] == 0  # minY
        assert env[3] == 1  # maxY
        ds = None

    def test_complex_extent(self, complex_vector, tmp_path):
        """Test creating extent from complex vector."""
        out_path = str(tmp_path / "complex_extent.gpkg")
        result = vector_to_extent(complex_vector, out_path)
        
        # Check if output exists
        assert os.path.exists(result)
        
        # Check if extent is correct
        ds = ogr.Open(result)
        layer = ds.GetLayer()
        feature = layer.GetNextFeature()
        extent = feature.GetGeometryRef()
        
        # Check bounds (should encompass both polygons)
        env = extent.GetEnvelope()
        assert env[0] == -1  # minX
        assert env[1] == 2   # maxX
        assert env[2] == -1  # minY
        assert env[3] == 2   # maxY
        ds = None

    def test_latlng_extent(self, simple_vector, tmp_path):
        """Test creating extent with latlng option."""
        out_path = str(tmp_path / "latlng_extent.gpkg")
        result = vector_to_extent(simple_vector, out_path, latlng=True)
        
        # Check if output exists
        assert os.path.exists(result)
        
        # Check if projection is correct (should be EPSG:4326)
        ds = ogr.Open(result)
        layer = ds.GetLayer()
        srs = layer.GetSpatialRef()
        assert srs.GetAuthorityCode(None) == '4326'
        ds = None

    def test_memory_output(self, simple_vector):
        """Test creating extent in memory."""
        result = vector_to_extent(simple_vector, out_path=None)
        
        # Check if path is memory path
        assert "/vsimem/" in result
        
        # Check if dataset exists and can be opened
        ds = ogr.Open(result)
        assert ds is not None
        ds = None

    def test_invalid_output_path(self, simple_vector):
        """Test error on invalid output path."""
        with pytest.raises(ValueError):
            vector_to_extent(simple_vector, out_path="invalid/path/test.gpkg")

    def test_invalid_input(self, tmp_path):
        """Test error on invalid input."""
        out_path = str(tmp_path / "invalid_extent.gpkg")
        with pytest.raises((ValueError, RuntimeError)):
            vector_to_extent("nonexistent.gpkg", out_path)
