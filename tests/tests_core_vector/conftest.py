"""Fixtures for vector tests."""

import pytest
import os
from osgeo import ogr, osr

@pytest.fixture
def valid_vector(tmp_path):
    """Create a valid vector with point geometry and attributes.
    
    Returns:
        str: Path to a valid GeoPackage vector file with point geometry
    """
    vector_path = tmp_path / "valid.gpkg"
    driver = ogr.GetDriverByName('GPKG')
    ds = driver.CreateDataSource(str(vector_path))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    layer = ds.CreateLayer('test', srs, ogr.wkbPoint)

    # Add attributes
    field_defn = ogr.FieldDefn('name', ogr.OFTString)
    layer.CreateField(field_defn)

    # Add feature
    feature = ogr.Feature(layer.GetLayerDefn())
    point = ogr.Geometry(ogr.wkbPoint)
    point.AddPoint(0, 0)
    feature.SetGeometry(point)
    feature.SetField('name', 'test')
    layer.CreateFeature(feature)

    ds.FlushCache()
    ds = None

    return str(vector_path)

@pytest.fixture
def vector_no_geometry(tmp_path):
    """Create a vector without geometry.
    
    Returns:
        str: Path to a GeoPackage vector file with no geometry
    """
    vector_path = tmp_path / "no_geometry.gpkg"
    driver = ogr.GetDriverByName('GPKG')
    ds = driver.CreateDataSource(str(vector_path))
    layer = ds.CreateLayer('test', None, ogr.wkbNone)

    field_defn = ogr.FieldDefn('name', ogr.OFTString)
    layer.CreateField(field_defn)

    feature = ogr.Feature(layer.GetLayerDefn())
    feature.SetField('name', 'test')
    layer.CreateFeature(feature)

    ds.FlushCache()
    ds = None

    return str(vector_path)

@pytest.fixture
def complex_vector(tmp_path):
    """Create a vector with multiple geometry types and attributes.
    
    Returns:
        str: Path to a GeoPackage vector file with multiple layers of different types
    """
    vector_path = tmp_path / "complex.gpkg"
    driver = ogr.GetDriverByName('GPKG')
    ds = driver.CreateDataSource(str(vector_path))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)

    # Point layer
    point_layer = ds.CreateLayer('points', srs, ogr.wkbPoint)
    point_layer.CreateField(ogr.FieldDefn('name', ogr.OFTString))
    point_layer.CreateField(ogr.FieldDefn('value', ogr.OFTInteger))

    point_feature = ogr.Feature(point_layer.GetLayerDefn())
    point = ogr.Geometry(ogr.wkbPoint)
    point.AddPoint(0, 0)
    point_feature.SetGeometry(point)
    point_feature.SetField('name', 'test_point')
    point_feature.SetField('value', 1)
    point_layer.CreateFeature(point_feature)

    # Polygon layer
    poly_layer = ds.CreateLayer('polygons', srs, ogr.wkbPolygon)
    poly_layer.CreateField(ogr.FieldDefn('area', ogr.OFTReal))

    poly_feature = ogr.Feature(poly_layer.GetLayerDefn())
    ring = ogr.Geometry(ogr.wkbLinearRing)
    ring.AddPoint(0, 0)
    ring.AddPoint(0, 1)
    ring.AddPoint(1, 1)
    ring.AddPoint(1, 0)
    ring.AddPoint(0, 0)

    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(ring)
    poly_feature.SetGeometry(poly)
    poly_feature.SetField('area', 1.0)
    poly_layer.CreateFeature(poly_feature)

    ds.FlushCache()
    ds = None

    return str(vector_path)

@pytest.fixture
def multitype_vector(tmp_path):
    """Create a vector with multiple layers of different types.
    
    Returns:
        str: Path to a GeoPackage with point, line, polygon, and table layers
    """
    vector_path = tmp_path / "multitype.gpkg"
    driver = ogr.GetDriverByName('GPKG')
    ds = driver.CreateDataSource(str(vector_path))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)

    # Point layer
    _ = ds.CreateLayer('points', srs, ogr.wkbPoint)
    # Line layer
    _ = ds.CreateLayer('lines', srs, ogr.wkbLineString)
    # Polygon layer
    _ = ds.CreateLayer('polygons', srs, ogr.wkbPolygon)
    # Table layer (no geometry)
    _ = ds.CreateLayer('table', None, ogr.wkbNone)

    ds.FlushCache()
    ds = None
    return str(vector_path)

@pytest.fixture
def multilayer_vector(tmp_path):
    """Create a vector with multiple layers sharing same geometry type.
    
    Returns:
        str: Path to a GeoPackage with multiple point layers
    """
    vector_path = tmp_path / "multilayer.gpkg"
    driver = ogr.GetDriverByName('GPKG')
    ds = driver.CreateDataSource(str(vector_path))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)

    for i in range(3):
        layer = ds.CreateLayer(f'layer_{i}', srs, ogr.wkbPoint)
        layer.CreateField(ogr.FieldDefn(f'field_{i}', ogr.OFTString))
        feature = ogr.Feature(layer.GetLayerDefn())
        point = ogr.Geometry(ogr.wkbPoint)
        point.AddPoint(i, i)
        feature.SetGeometry(point)
        feature.SetField(f'field_{i}', f'value_{i}')
        layer.CreateFeature(feature)

    ds.FlushCache()
    ds = None
    return str(vector_path)

@pytest.fixture
def point_vector(tmp_path):
    """Create a simple point vector.
    
    Returns:
        str: Path to a GeoPackage with a single point
    """
    vector_path = tmp_path / "point.gpkg"
    driver = ogr.GetDriverByName('GPKG')
    ds = driver.CreateDataSource(str(vector_path))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    layer = ds.CreateLayer('points', srs, ogr.wkbPoint)
    
    # Add a point feature
    feature = ogr.Feature(layer.GetLayerDefn())
    point = ogr.Geometry(ogr.wkbPoint)
    point.AddPoint(10, 20)
    feature.SetGeometry(point)
    layer.CreateFeature(feature)
    
    ds.FlushCache()
    ds = None
    return str(vector_path)

@pytest.fixture
def line_vector(tmp_path):
    """Create a simple line vector.
    
    Returns:
        str: Path to a GeoPackage with a single line
    """
    vector_path = tmp_path / "line.gpkg"
    driver = ogr.GetDriverByName('GPKG')
    ds = driver.CreateDataSource(str(vector_path))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    layer = ds.CreateLayer('lines', srs, ogr.wkbLineString)
    
    # Add a line feature
    feature = ogr.Feature(layer.GetLayerDefn())
    line = ogr.Geometry(ogr.wkbLineString)
    line.AddPoint(0, 0)
    line.AddPoint(1, 1)
    line.AddPoint(2, 0)
    feature.SetGeometry(line)
    layer.CreateFeature(feature)
    
    ds.FlushCache()
    ds = None
    return str(vector_path)

@pytest.fixture
def polygon_vector(tmp_path):
    """Create a simple polygon vector.
    
    Returns:
        str: Path to a GeoPackage with a single polygon
    """
    vector_path = tmp_path / "polygon.gpkg"
    driver = ogr.GetDriverByName('GPKG')
    ds = driver.CreateDataSource(str(vector_path))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    layer = ds.CreateLayer('polygons', srs, ogr.wkbPolygon)
    
    # Add a polygon feature
    feature = ogr.Feature(layer.GetLayerDefn())
    ring = ogr.Geometry(ogr.wkbLinearRing)
    ring.AddPoint(0, 0)
    ring.AddPoint(0, 1)
    ring.AddPoint(1, 1)
    ring.AddPoint(1, 0)
    ring.AddPoint(0, 0)
    
    polygon = ogr.Geometry(ogr.wkbPolygon)
    polygon.AddGeometry(ring)
    feature.SetGeometry(polygon)
    layer.CreateFeature(feature)
    
    ds.FlushCache()
    ds = None
    return str(vector_path)
