# pylint: skip-file
# type: ignore

# Standard library
import sys; sys.path.append("../../")

import pytest
from osgeo import ogr, osr, gdal

from buteo.core_vector.core_vector_read import (
    _open_vector,
    check_vector_has_geometry,
    check_vector_has_attributes,
    check_vector_has_crs,
    check_vector_has_invalid_geometry,
    vector_fix_geometry
)

@pytest.fixture
def valid_vector(tmp_path):
    """Create a valid vector with point geometry and attributes."""
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
    
    ds = None
    return str(vector_path)

@pytest.fixture
def vector_no_geometry(tmp_path):
    """Create a vector without geometry."""
    vector_path = tmp_path / "no_geometry.gpkg"
    driver = ogr.GetDriverByName('GPKG')
    ds = driver.CreateDataSource(str(vector_path))
    layer = ds.CreateLayer('test', None, ogr.wkbNone)
    
    field_defn = ogr.FieldDefn('name', ogr.OFTString)
    layer.CreateField(field_defn)
    
    feature = ogr.Feature(layer.GetLayerDefn())
    feature.SetField('name', 'test')
    layer.CreateFeature(feature)
    
    ds = None
    return str(vector_path)

@pytest.fixture
def vector_invalid_geometry(tmp_path):
    """Create a vector with invalid geometry (self-intersecting polygon)."""
    vector_path = tmp_path / "invalid_geom.gpkg"
    driver = ogr.GetDriverByName('GPKG')
    ds = driver.CreateDataSource(str(vector_path))
    layer = ds.CreateLayer('test', None, ogr.wkbPolygon)
    
    # Create self-intersecting polygon
    ring = ogr.Geometry(ogr.wkbLinearRing)
    ring.AddPoint(0, 0)
    ring.AddPoint(2, 2)
    ring.AddPoint(0, 2)
    ring.AddPoint(2, 0)
    ring.AddPoint(0, 0)
    
    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(ring)
    
    feature = ogr.Feature(layer.GetLayerDefn())
    feature.SetGeometry(poly)
    layer.CreateFeature(feature)
    
    ds = None
    return str(vector_path)

@pytest.fixture
def complex_vector(tmp_path):
    """Create a vector with multiple geometry types and attributes."""
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
    
    ds = None
    return str(vector_path)

@pytest.fixture
def multitype_vector(tmp_path):
    """Create a vector with multiple layers of different types."""
    vector_path = tmp_path / "multitype.gpkg"
    driver = ogr.GetDriverByName('GPKG')
    ds = driver.CreateDataSource(str(vector_path))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    
    # Point layer
    point_layer = ds.CreateLayer('points', srs, ogr.wkbPoint)
    # Line layer
    line_layer = ds.CreateLayer('lines', srs, ogr.wkbLineString)
    # Polygon layer
    poly_layer = ds.CreateLayer('polygons', srs, ogr.wkbPolygon)
    # Table layer (no geometry)
    table_layer = ds.CreateLayer('table', None, ogr.wkbNone)
    
    ds = None
    return str(vector_path)

@pytest.fixture
def multilayer_vector(tmp_path):
    """Create a vector with multiple layers sharing same geometry type."""
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
    
    ds = None
    return str(vector_path)


class TestOpenVector:
    def test_open_valid_vector(self, valid_vector):
        """Test opening a valid vector file."""
        ds = _open_vector(valid_vector)
        assert isinstance(ds, ogr.DataSource)
        assert ds.GetLayerCount() == 1

    def test_open_nonexistent_vector(self):
        """Test opening a nonexistent vector file."""
        with pytest.raises(ValueError):
            _open_vector("nonexistent.gpkg")

    def test_open_invalid_input(self):
        """Test opening with invalid input type."""
        with pytest.raises(TypeError):
            _open_vector(123)

    def test_open_write_mode(self, valid_vector):
        """Test opening in write mode."""
        ds = _open_vector(valid_vector, writeable=True)
        assert isinstance(ds, ogr.DataSource)
        layer = ds.GetLayer()
        assert layer.TestCapability(ogr.OLCRandomWrite)

    def test_open_complex_vector(self, complex_vector):
        """Test opening a complex vector with multiple layers."""
        ds = _open_vector(complex_vector)
        assert ds.GetLayerCount() == 2
        assert ds.GetLayer('points') is not None
        assert ds.GetLayer('polygons') is not None
        
    def test_open_write_mode_multilayer(self, multilayer_vector):
        """Test opening multilayer vector in write mode."""
        ds = _open_vector(multilayer_vector, writeable=True)
        for i in range(3):
            layer = ds.GetLayer(i)
            assert layer.TestCapability(ogr.OLCRandomWrite)
            
    def test_open_vsimem(self):
        """Test opening a vector in virtual memory."""
        vsimem_path = '/vsimem/test.gpkg'
        driver = ogr.GetDriverByName('GPKG')
        ds = driver.CreateDataSource(vsimem_path)
        layer = ds.CreateLayer('test')
        ds = None
        
        opened_ds = _open_vector(vsimem_path)
        assert opened_ds is not None
        assert opened_ds.GetLayer(0) is not None
        gdal.Unlink(vsimem_path)

class TestCheckVectorHasGeometry:
    def test_vector_with_geometry(self, valid_vector):
        """Test checking vector with geometry."""
        assert check_vector_has_geometry(valid_vector) is True

    def test_vector_without_geometry(self, vector_no_geometry):
        """Test checking vector without geometry."""
        assert check_vector_has_geometry(vector_no_geometry) is False

    def test_specific_layer(self, valid_vector):
        """Test checking specific layer."""
        assert check_vector_has_geometry(valid_vector, layer_name_or_id="test") is True
        assert check_vector_has_geometry(valid_vector, layer_name_or_id="nonexistent") is False
        
    def test_layer_without_geometry(self, multitype_vector):
        """Test checking layer without geometry."""
        assert not check_vector_has_geometry(multitype_vector, layer_name_or_id="table")

class TestCheckVectorHasAttributes:
    def test_vector_with_attributes(self, valid_vector):
        """Test checking vector with attributes."""
        assert check_vector_has_attributes(valid_vector) is True
        assert check_vector_has_attributes(valid_vector, attributes="name") is True
        assert check_vector_has_attributes(valid_vector, attributes=["name"]) is True

    def test_vector_specific_attributes(self, valid_vector):
        """Test checking for specific attributes."""
        assert check_vector_has_attributes(valid_vector, attributes="nonexistent") is False
        assert check_vector_has_attributes(valid_vector, attributes=["name", "nonexistent"]) is False

    def test_specific_layer(self, valid_vector):
        """Test checking attributes in specific layer."""
        assert check_vector_has_attributes(valid_vector, layer_name_or_id="test") is True
        assert check_vector_has_attributes(valid_vector, layer_name_or_id="nonexistent") is False

    def test_multiple_attributes(self, complex_vector):
        """Test checking multiple attributes."""
        assert check_vector_has_attributes(complex_vector, attributes=["name", "value"])
        assert check_vector_has_attributes(complex_vector, attributes=["area"])
        
    def test_attributes_by_layer(self, complex_vector):
        """Test checking attributes in specific layers."""
        assert check_vector_has_attributes(
            complex_vector,
            layer_name_or_id="points",
            attributes=["name", "value"]
        )
        assert not check_vector_has_attributes(
            complex_vector,
            layer_name_or_id="points",
            attributes=["area"]
        )
        
    def test_nonexistent_attributes(self, complex_vector):
        """Test checking for nonexistent attributes."""
        assert not check_vector_has_attributes(complex_vector, attributes=["nonexistent"])

class TestCheckVectorHasCRS:
    def test_vector_with_crs(self, valid_vector):
        """Test checking vector with CRS."""
        assert check_vector_has_crs(valid_vector) is True

    def test_vector_without_crs(self, vector_no_geometry):
        """Test checking vector without CRS."""
        assert check_vector_has_crs(vector_no_geometry) is False

    def test_specific_layer(self, valid_vector):
        """Test checking CRS in specific layer."""
        assert check_vector_has_crs(valid_vector, layer_name_or_id="test") is True
        assert check_vector_has_crs(valid_vector, layer_name_or_id="nonexistent") is False

    def test_multiple_layer_crs(self, multitype_vector):
        """Test CRS check across multiple layers."""
        assert check_vector_has_crs(multitype_vector)
        
    def test_mixed_crs_layers(self, tmp_path):
        """Test with layers having different CRS settings."""
        vector_path = tmp_path / "mixed_crs.gpkg"
        driver = ogr.GetDriverByName('GPKG')
        ds = driver.CreateDataSource(str(vector_path))
        
        # Layer with CRS
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        layer1 = ds.CreateLayer('with_crs', srs, ogr.wkbPoint)
        
        # Layer without CRS
        layer2 = ds.CreateLayer('without_crs', None, ogr.wkbPoint)
        
        ds = None
        
        assert check_vector_has_crs(str(vector_path), layer_name_or_id="with_crs")
        assert not check_vector_has_crs(str(vector_path), layer_name_or_id="without_crs")

class TestCheckVectorHasInvalidGeometry:
    def test_valid_geometry(self, valid_vector):
        """Test checking valid geometry."""
        assert check_vector_has_invalid_geometry(valid_vector) is False

    def test_invalid_geometry(self, vector_invalid_geometry):
        """Test checking invalid geometry."""
        assert check_vector_has_invalid_geometry(vector_invalid_geometry) is True

    def test_empty_geometry_handling(self, valid_vector):
        """Test handling of empty geometries."""
        assert check_vector_has_invalid_geometry(valid_vector, allow_empty=True) is False

    def test_complex_invalid_geometries(self, tmp_path):
        """Test detecting various types of invalid geometries."""
        vector_path = tmp_path / "invalid_geoms.gpkg"
        driver = ogr.GetDriverByName('GPKG')
        ds = driver.CreateDataSource(str(vector_path))
        layer = ds.CreateLayer('test', None, ogr.wkbPolygon)
        
        # Self-intersecting polygon
        ring = ogr.Geometry(ogr.wkbLinearRing)
        ring.AddPoint(0, 0)
        ring.AddPoint(2, 2)
        ring.AddPoint(0, 2)
        ring.AddPoint(2, 0)
        ring.AddPoint(0, 0)
        
        poly = ogr.Geometry(ogr.wkbPolygon)
        poly.AddGeometry(ring)
        
        feature = ogr.Feature(layer.GetLayerDefn())
        feature.SetGeometry(poly)
        layer.CreateFeature(feature)
        
        ds = None
        
        assert check_vector_has_invalid_geometry(str(vector_path))
        
    def test_empty_geometries(self, tmp_path):
        """Test handling of empty geometries."""
        vector_path = tmp_path / "empty_geoms.gpkg"
        driver = ogr.GetDriverByName('GPKG')
        ds = driver.CreateDataSource(str(vector_path))
        layer = ds.CreateLayer('test', None, ogr.wkbPoint)
        
        feature = ogr.Feature(layer.GetLayerDefn())
        empty_point = ogr.Geometry(ogr.wkbPoint)
        feature.SetGeometry(empty_point)
        layer.CreateFeature(feature)
        
        ds = None

        assert check_vector_has_invalid_geometry(str(vector_path))
        assert not check_vector_has_invalid_geometry(str(vector_path), allow_empty=True)

class TestVectorFixGeometry:
    def test_fix_invalid_geometry(self, vector_invalid_geometry):
        """Test fixing invalid geometry."""
        assert vector_fix_geometry(vector_invalid_geometry) is True
        assert check_vector_has_invalid_geometry(vector_invalid_geometry) is False

    def test_fix_valid_geometry(self, valid_vector):
        """Test fixing already valid geometry."""
        assert vector_fix_geometry(valid_vector) is True

    def test_specific_layer(self, vector_invalid_geometry):
        """Test fixing geometry in specific layer."""
        assert vector_fix_geometry(vector_invalid_geometry, layer_name_or_id="test") is True
        assert check_vector_has_invalid_geometry(vector_invalid_geometry, layer_name_or_id="test") is False

    def test_fix_multiple_invalid_geometries(self, tmp_path):
        """Test fixing multiple invalid geometries."""
        vector_path = tmp_path / "multiple_invalid.gpkg"
        driver = ogr.GetDriverByName('GPKG')
        ds = driver.CreateDataSource(str(vector_path))
        layer = ds.CreateLayer('test', None, ogr.wkbPolygon)
        
        # Create multiple invalid polygons
        for i in range(3):
            ring = ogr.Geometry(ogr.wkbLinearRing)
            ring.AddPoint(i, i)
            ring.AddPoint(i+2, i+2)
            ring.AddPoint(i, i+2)
            ring.AddPoint(i+2, i)
            ring.AddPoint(i, i)
            
            poly = ogr.Geometry(ogr.wkbPolygon)
            poly.AddGeometry(ring)
            
            feature = ogr.Feature(layer.GetLayerDefn())
            feature.SetGeometry(poly)
            layer.CreateFeature(feature)
        
        ds = None
        
        assert vector_fix_geometry(str(vector_path))
        assert not check_vector_has_invalid_geometry(str(vector_path))