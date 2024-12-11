# pylint: skip-file
# type: ignore

# Standard library
import sys; sys.path.append("../../")
import os

import pytest
from osgeo import ogr

from buteo.core_vector.core_vector_index import (
    vector_create_index,
    vector_delete_index,
    check_vector_has_index,
)



@pytest.fixture
def sample_vectors(tmp_path):
    """Create test vector files in different formats"""
    drivers = {
        'shp': 'ESRI Shapefile',
        'gpkg': 'GPKG',
        'fgb': 'FlatGeobuf', 
        'geojson': 'GeoJSON'
    }
    
    files = {}
    
    for ext, driver_name in drivers.items():
        filename = str(tmp_path / f"test.{ext}")
        driver = ogr.GetDriverByName(driver_name)
        if driver is None:
            continue
        ds = driver.CreateDataSource(filename)
        layer_name = os.path.splitext(os.path.basename(filename))[0]
        layer = ds.CreateLayer(layer_name, geom_type=ogr.wkbPolygon)
        
        # Add a field
        layer.CreateField(ogr.FieldDefn('id', ogr.OFTInteger))
        
        # Create a square polygon using WKT
        wkt = "POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))"
        poly = ogr.CreateGeometryFromWkt(wkt)
        
        # Add features
        for i in range(5):
            feature = ogr.Feature(layer.GetLayerDefn())
            feature.SetField('id', i)
            feature.SetGeometry(poly)
            layer.CreateFeature(feature)
        
        layer.SyncToDisk()
        layer = None
        files[ext] = filename
        ds = None
        
    return files

class TestVectorIndex:
    
    def test_create_index_shapefile(self, sample_vectors):
        """Test creating index on shapefile"""
        if 'shp' not in sample_vectors:
            pytest.skip("Shapefile driver not available")

        result = vector_create_index(sample_vectors['shp'])
        assert result is False
        assert check_vector_has_index(sample_vectors['shp']) == result

    def test_create_index_geopackage(self, sample_vectors):
        """Test creating index on geopackage"""
        if 'gpkg' not in sample_vectors:
            pytest.skip("GPKG driver not available")
            
        result = vector_create_index(sample_vectors['gpkg'])
        assert result is True
        assert check_vector_has_index(sample_vectors['gpkg']) == result

    def test_create_index_flatgeobuf(self, sample_vectors):
        """Test creating index on flatgeobuf"""
        if 'fgb' not in sample_vectors:
            pytest.skip("FlatGeobuf driver not available")
            
        result = vector_create_index(sample_vectors['fgb'])
        assert result is True
        assert check_vector_has_index(sample_vectors['fgb']) == result

    def test_create_index_geojson(self, sample_vectors):
        """Test creating index on geojson"""
        if 'geojson' not in sample_vectors:
            pytest.skip("GeoJSON driver not available")
            
        result = vector_create_index(sample_vectors['geojson'])
        assert result is False
        assert check_vector_has_index(sample_vectors['geojson']) == result

    def test_delete_index(self, sample_vectors):
        """Test deleting index"""
        if 'gpkg' not in sample_vectors:
            pytest.skip("Shapefile driver not available")
            
        # First create index
        vector_create_index(sample_vectors['gpkg'])
        
        # Then delete it
        result = vector_delete_index(sample_vectors['gpkg'])
        assert result is True
        assert check_vector_has_index(sample_vectors['gpkg']) is False

    def test_overwrite_index(self, sample_vectors):
        """Test overwriting existing index"""
        if 'gpkg' not in sample_vectors:
            pytest.skip("GPKG driver not available")
            
        # Create initial index
        vector_create_index(sample_vectors['gpkg'])
        
        # Overwrite it
        result = vector_create_index(sample_vectors['gpkg'], overwrite=True)
        assert isinstance(result, bool)
        assert result is True

    def test_invalid_input(self):
        """Test error handling for invalid input"""
        with pytest.raises(ValueError):
            check_vector_has_index("nonexistent_file.shp")
            
        with pytest.raises(ValueError):
            vector_create_index("nonexistent_file.shp")
            
        with pytest.raises(ValueError):
            vector_delete_index("nonexistent_file.shp")
