# pylint: skip-file
# type: ignore

# Standard library
import sys; sys.path.append("../../")

import pytest
from osgeo import ogr, osr, gdal

from buteo.core_vector.core_vector_read import (
    open_vector,
    _vector_get_layer,
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

    ds.FlushCache()
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

    ds.FlushCache()
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

    ds.FlushCache()
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

    ds.FlushCache()
    ds = None
    return str(vector_path)


class TestOpenVector:
    def test_open_valid_vector(self, valid_vector):
        """Test opening a valid vector file."""
        ds = open_vector(valid_vector)
        assert isinstance(ds, ogr.DataSource)
        assert ds.GetLayerCount() == 1

    def test_open_nonexistent_vector(self):
        """Test opening a nonexistent vector file."""
        with pytest.raises(ValueError):
            open_vector("nonexistent.gpkg")

    def test_open_invalid_input(self):
        """Test opening with invalid input type."""
        with pytest.raises(TypeError):
            open_vector(123)

    def test_open_write_mode(self, valid_vector):
        """Test opening in write mode."""
        ds = open_vector(valid_vector, writeable=True)
        assert isinstance(ds, ogr.DataSource)
        layer = ds.GetLayer()
        assert layer.TestCapability(ogr.OLCRandomWrite)

    def test_open_complex_vector(self, complex_vector):
        """Test opening a complex vector with multiple layers."""
        ds = open_vector(complex_vector)
        assert ds.GetLayerCount() == 2
        assert ds.GetLayer('points') is not None
        assert ds.GetLayer('polygons') is not None

    def test_open_write_mode_multilayer(self, multilayer_vector):
        """Test opening multilayer vector in write mode."""
        ds = open_vector(multilayer_vector, writeable=True)
        for i in range(3):
            layer = ds.GetLayer(i)
            assert layer.TestCapability(ogr.OLCRandomWrite)

    def test_open_vsimem(self):
        """Test opening a vector in virtual memory."""
        vsimem_path = '/vsimem/test.gpkg'
        driver = ogr.GetDriverByName('GPKG')
        ds = driver.CreateDataSource(vsimem_path)
        _ = ds.CreateLayer('test')
        ds = None

        opened_ds = open_vector(vsimem_path)
        assert opened_ds is not None
        assert opened_ds.GetLayer(0) is not None
        gdal.Unlink(vsimem_path)

    def test_open_vector_no_geometry(self, vector_no_geometry):
        """Test opening a vector without geometry."""
        ds = open_vector(vector_no_geometry)
        assert isinstance(ds, ogr.DataSource)
        assert ds.GetLayerCount() == 1
        layer = ds.GetLayer()
        assert layer.GetGeomType() == ogr.wkbNone

    def test_open_multitype_vector(self, multitype_vector):
        """Test opening a vector with multiple layers of different types."""
        ds = open_vector(multitype_vector)
        assert isinstance(ds, ogr.DataSource)
        assert ds.GetLayerCount() == 4
        assert ds.GetLayer('points') is not None
        assert ds.GetLayer('lines') is not None
        assert ds.GetLayer('polygons') is not None
        assert ds.GetLayer('table') is not None

class TestVectorGetLayer:
    def test_get_layer_by_index(self, valid_vector):
        """Test getting a layer by index."""
        ds = open_vector(valid_vector)
        layers = _vector_get_layer(ds, 0)
        assert len(layers) == 1
        assert layers[0].GetName() == 'test'

    def test_get_layer_by_name(self, valid_vector):
        """Test getting a layer by name."""
        ds = open_vector(valid_vector)
        layers = _vector_get_layer(ds, 'test')
        assert len(layers) == 1
        assert layers[0].GetName() == 'test'

    def test_get_all_layers(self, multilayer_vector):
        """Test getting all layers from a vector."""
        ds = open_vector(multilayer_vector)
        layers = _vector_get_layer(ds)
        assert len(layers) == 3
        for i, layer in enumerate(layers):
            assert layer.GetName() == f'layer_{i}'

    def test_get_nonexistent_layer_by_index(self, valid_vector):
        """Test getting a nonexistent layer by index."""
        ds = open_vector(valid_vector)
        with pytest.raises(ValueError):
            _vector_get_layer(ds, 1)

    def test_get_nonexistent_layer_by_name(self, valid_vector):
        """Test getting a nonexistent layer by name."""
        ds = open_vector(valid_vector)
        with pytest.raises(ValueError):
            _vector_get_layer(ds, 'nonexistent')

    def test_get_layer_from_multitype_vector(self, multitype_vector):
        """Test getting layers from a multitype vector."""
        ds = open_vector(multitype_vector)
        layers = _vector_get_layer(ds, 'points')
        assert len(layers) == 1
        assert layers[0].GetName() == 'points'

        layers = _vector_get_layer(ds, 'lines')
        assert len(layers) == 1
        assert layers[0].GetName() == 'lines'

        layers = _vector_get_layer(ds, 'polygons')
        assert len(layers) == 1
        assert layers[0].GetName() == 'polygons'

        layers = _vector_get_layer(ds, 'table')
        assert len(layers) == 1
        assert layers[0].GetName() == 'table'
