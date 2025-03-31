"""### Test fixtures for vector tests. ###"""

# Standard library
import os
import shutil
import tempfile
from pathlib import Path

# External
import pytest
from osgeo import ogr, osr

@pytest.fixture(scope="function")
def temp_dir():
    """Create a temporary directory for tests and clean it up afterward."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

@pytest.fixture(scope="function")
def test_polygon(temp_dir):
    """Create a simple polygon shapefile for testing."""
    out_path = os.path.join(temp_dir, "test_polygon.gpkg")
    
    # Create datasource
    driver = ogr.GetDriverByName("GPKG")
    datasource = driver.CreateDataSource(out_path)
    
    # Create spatial reference
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)  # WGS84
    
    # Create layer
    layer = datasource.CreateLayer("test_layer", srs, ogr.wkbPolygon)
    
    # Add attribute field
    field_defn = ogr.FieldDefn("buffer_dist", ogr.OFTReal)
    layer.CreateField(field_defn)
    
    # Create feature
    feature_defn = layer.GetLayerDefn()
    feature = ogr.Feature(feature_defn)
    feature.SetField("buffer_dist", 0.01)  # ~1km at equator
    
    # Create geometry (rectangle)
    ring = ogr.Geometry(ogr.wkbLinearRing)
    ring.AddPoint(0.0, 0.0)
    ring.AddPoint(0.0, 1.0)
    ring.AddPoint(1.0, 1.0)
    ring.AddPoint(1.0, 0.0)
    ring.AddPoint(0.0, 0.0)
    
    polygon = ogr.Geometry(ogr.wkbPolygon)
    polygon.AddGeometry(ring)
    
    feature.SetGeometry(polygon)
    layer.CreateFeature(feature)
    
    # Clean up
    feature = None
    datasource = None
    
    return out_path

@pytest.fixture(scope="function")
def test_points(temp_dir):
    """Create a points shapefile for testing."""
    out_path = os.path.join(temp_dir, "test_points.gpkg")
    
    # Create datasource
    driver = ogr.GetDriverByName("GPKG")
    datasource = driver.CreateDataSource(out_path)
    
    # Create spatial reference
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)  # WGS84
    
    # Create layer
    layer = datasource.CreateLayer("test_layer", srs, ogr.wkbPoint)
    
    # Add attribute fields
    buffer_field = ogr.FieldDefn("buffer_dist", ogr.OFTReal)
    layer.CreateField(buffer_field)
    
    type_field = ogr.FieldDefn("type", ogr.OFTString)
    layer.CreateField(type_field)
    
    pop_field = ogr.FieldDefn("population", ogr.OFTInteger)
    layer.CreateField(pop_field)
    
    # Create 5 points with different attributes
    feature_defn = layer.GetLayerDefn()
    
    # Types for the points: city, town, village, etc.
    point_types = ["city", "town", "village", "hamlet", "settlement"]
    # Population values
    populations = [100000, 50000, 10000, 5000, 1000]
    
    for i in range(5):
        feature = ogr.Feature(feature_defn)
        feature.SetField("buffer_dist", 0.01 * (i + 1))
        feature.SetField("type", point_types[i])
        feature.SetField("population", populations[i])
        
        point = ogr.Geometry(ogr.wkbPoint)
        point.AddPoint(i * 0.1, i * 0.1)
        
        feature.SetGeometry(point)
        layer.CreateFeature(feature)
    
    # Clean up
    datasource = None
    
    return out_path

@pytest.fixture(scope="function")
def test_multipolygon(temp_dir):
    """Create a multipolygon shapefile for testing."""
    out_path = os.path.join(temp_dir, "test_multipolygon.gpkg")
    
    # Create datasource
    driver = ogr.GetDriverByName("GPKG")
    datasource = driver.CreateDataSource(out_path)
    
    # Create spatial reference
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)  # WGS84
    
    # Create layer
    layer = datasource.CreateLayer("test_layer", srs, ogr.wkbMultiPolygon)
    
    # Add attribute fields
    id_field = ogr.FieldDefn("id", ogr.OFTInteger)
    layer.CreateField(id_field)
    
    # Use "class" instead of "category" to match the tests
    class_field = ogr.FieldDefn("class", ogr.OFTString)
    layer.CreateField(class_field)
    
    # Create features (2 polygons in a multipolygon)
    feature_defn = layer.GetLayerDefn()
    
    # Categories for classification
    categories = ["forest", "residential", "commercial", "industrial"]
    
    for i in range(3):
        feature = ogr.Feature(feature_defn)
        feature.SetField("id", i + 1)
        feature.SetField("class", categories[i % len(categories)])
        
        # Create a multipolygon with two polygons
        multipolygon = ogr.Geometry(ogr.wkbMultiPolygon)
        
        # First polygon
        ring1 = ogr.Geometry(ogr.wkbLinearRing)
        ring1.AddPoint(i * 1.0, i * 1.0)
        ring1.AddPoint(i * 1.0, i * 1.0 + 0.5)
        ring1.AddPoint(i * 1.0 + 0.5, i * 1.0 + 0.5)
        ring1.AddPoint(i * 1.0 + 0.5, i * 1.0)
        ring1.AddPoint(i * 1.0, i * 1.0)
        
        polygon1 = ogr.Geometry(ogr.wkbPolygon)
        polygon1.AddGeometry(ring1)
        multipolygon.AddGeometry(polygon1)
        
        # Second polygon
        ring2 = ogr.Geometry(ogr.wkbLinearRing)
        ring2.AddPoint(i * 1.0 + 0.75, i * 1.0)
        ring2.AddPoint(i * 1.0 + 0.75, i * 1.0 + 0.25)
        ring2.AddPoint(i * 1.0 + 1.0, i * 1.0 + 0.25)
        ring2.AddPoint(i * 1.0 + 1.0, i * 1.0)
        ring2.AddPoint(i * 1.0 + 0.75, i * 1.0)
        
        polygon2 = ogr.Geometry(ogr.wkbPolygon)
        polygon2.AddGeometry(ring2)
        multipolygon.AddGeometry(polygon2)
        
        feature.SetGeometry(multipolygon)
        layer.CreateFeature(feature)
    
    # Clean up
    datasource = None
    
    return out_path

@pytest.fixture(scope="function")
def test_multilayer(temp_dir):
    """Create a vector with multiple layers for testing."""
    out_path = os.path.join(temp_dir, "test_multilayer.gpkg")
    
    # Create datasource
    driver = ogr.GetDriverByName("GPKG")
    datasource = driver.CreateDataSource(out_path)
    
    # Create spatial reference
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)  # WGS84
    
    # Create first layer (points)
    layer1 = datasource.CreateLayer("points_layer", srs, ogr.wkbPoint)
    
    # Add attribute fields to first layer
    name_field = ogr.FieldDefn("name", ogr.OFTString)
    layer1.CreateField(name_field)
    
    # Create points
    feature_defn1 = layer1.GetLayerDefn()
    for i in range(3):
        feature = ogr.Feature(feature_defn1)
        feature.SetField("name", f"Point {i+1}")
        
        point = ogr.Geometry(ogr.wkbPoint)
        point.AddPoint(i * 0.2, i * 0.2)
        
        feature.SetGeometry(point)
        layer1.CreateFeature(feature)
    
    # Create second layer (polygons)
    layer2 = datasource.CreateLayer("polygons_layer", srs, ogr.wkbPolygon)
    
    # Add attribute fields to second layer
    type_field = ogr.FieldDefn("type", ogr.OFTString)
    layer2.CreateField(type_field)
    
    area_field = ogr.FieldDefn("area", ogr.OFTReal)
    layer2.CreateField(area_field)
    
    # Create polygons
    feature_defn2 = layer2.GetLayerDefn()
    for i in range(2):
        feature = ogr.Feature(feature_defn2)
        feature.SetField("type", f"Zone {i+1}")
        feature.SetField("area", 0.25 * (i+1))
        
        ring = ogr.Geometry(ogr.wkbLinearRing)
        ring.AddPoint(i * 0.5, i * 0.5)
        ring.AddPoint(i * 0.5, i * 0.5 + 0.5)
        ring.AddPoint(i * 0.5 + 0.5, i * 0.5 + 0.5)
        ring.AddPoint(i * 0.5 + 0.5, i * 0.5)
        ring.AddPoint(i * 0.5, i * 0.5)
        
        polygon = ogr.Geometry(ogr.wkbPolygon)
        polygon.AddGeometry(ring)
        
        feature.SetGeometry(polygon)
        layer2.CreateFeature(feature)
    
    # Clean up
    datasource = None
    
    return out_path

@pytest.fixture(scope="function")
def test_linestrings(temp_dir):
    """Create a linestring shapefile for testing."""
    out_path = os.path.join(temp_dir, "test_linestrings.gpkg")
    
    # Create datasource
    driver = ogr.GetDriverByName("GPKG")
    datasource = driver.CreateDataSource(out_path)
    
    # Create spatial reference
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)  # WGS84
    
    # Create layer
    layer = datasource.CreateLayer("test_layer", srs, ogr.wkbLineString)
    
    # Add attribute field
    field_defn = ogr.FieldDefn("buffer_dist", ogr.OFTReal)
    layer.CreateField(field_defn)
    
    # Create feature
    feature_defn = layer.GetLayerDefn()
    
    # Create a simple line
    feature = ogr.Feature(feature_defn)
    feature.SetField("buffer_dist", 0.02)
    
    line = ogr.Geometry(ogr.wkbLineString)
    line.AddPoint(0.0, 0.0)
    line.AddPoint(1.0, 1.0)
    
    feature.SetGeometry(line)
    layer.CreateFeature(feature)
    
    # Clean up
    datasource = None
    
    return out_path
