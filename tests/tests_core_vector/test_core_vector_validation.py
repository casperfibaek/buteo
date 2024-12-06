# pylint: skip-file
# type: ignore

# Standard library
import sys; sys.path.append("../../")

import pytest
from osgeo import ogr, osr

from buteo.core_vector.core_vector_validation import (
    check_vector_has_geometry,
    check_vector_has_attributes,
    check_vector_has_crs,
    check_vector_is_geometry_type,
    check_vector_is_point_type,
    check_vector_is_line_type,
    check_vector_is_polygon_type,
    check_vector_has_invalid_geometry,
    check_vector_has_multiple_layers,
    check_vector_is_valid,
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

        with pytest.raises(ValueError):
            check_vector_has_geometry(valid_vector, layer_name_or_id="nonexistent")

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

        with pytest.raises(ValueError):
            check_vector_has_attributes(valid_vector, layer_name_or_id="nonexistent")

    def test_multiple_attributes(self, complex_vector):
        """Test checking multiple attributes."""
        assert not check_vector_has_attributes(complex_vector, attributes=["name", "value"])
        assert not check_vector_has_attributes(complex_vector, attributes=["area"])

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

        with pytest.raises(ValueError):
            check_vector_has_crs(valid_vector, layer_name_or_id="nonexistent")

    def test_multiple_layer_crs(self, multitype_vector):
        """Test CRS check across multiple layers."""
        assert check_vector_has_crs(multitype_vector, layer_name_or_id="points")
        assert check_vector_has_crs(multitype_vector, layer_name_or_id="lines")
        assert check_vector_has_crs(multitype_vector, layer_name_or_id="polygons")
        assert not check_vector_has_crs(multitype_vector, layer_name_or_id="table")

    def test_mixed_crs_layers(self, tmp_path):
        """Test with layers having different CRS settings."""
        vector_path = tmp_path / "mixed_crs.gpkg"
        driver = ogr.GetDriverByName('GPKG')
        ds = driver.CreateDataSource(str(vector_path))

        # Layer with CRS
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        _ = ds.CreateLayer('with_crs', srs, ogr.wkbPoint)

        # Layer without CRS
        _ = ds.CreateLayer('without_crs', None, ogr.wkbPoint)

        ds = None

        assert check_vector_has_crs(str(vector_path), layer_name_or_id="with_crs")
        assert not check_vector_has_crs(str(vector_path), layer_name_or_id="without_crs")


class TestCheckVectorIsGeometryType:
    def test_single_geometry_type(self, valid_vector):
        """Test checking single geometry type."""
        assert check_vector_is_geometry_type(valid_vector, "3D POINT") is True
        assert check_vector_is_geometry_type(valid_vector, "POLYGON") is False

    def test_multiple_geometry_types(self, valid_vector):
        """Test checking multiple geometry types."""
        assert check_vector_is_geometry_type(valid_vector, ["3D POINT", "POLYGON"]) is True
        assert check_vector_is_geometry_type(valid_vector, ["POLYGON", "LINE STRING"]) is False

    def test_specific_layer(self, complex_vector):
        """Test checking geometry type in specific layer."""
        assert check_vector_is_geometry_type(complex_vector, "3D POINT", layer_name_or_id="points") is True
        assert check_vector_is_geometry_type(complex_vector, "3D POLYGON", layer_name_or_id="polygons") is True

        with pytest.raises(ValueError):
            check_vector_is_geometry_type(complex_vector, "POINT", layer_name_or_id="nonexistent")

    def test_multitype_layers(self, multitype_vector):
        """Test geometry types across different layers."""
        assert check_vector_is_geometry_type(multitype_vector, "POINT", layer_name_or_id="points") is True
        assert check_vector_is_geometry_type(multitype_vector, "LINE STRING", layer_name_or_id="lines") is True
        assert check_vector_is_geometry_type(multitype_vector, "POLYGON", layer_name_or_id="polygons") is True
        assert check_vector_is_geometry_type(multitype_vector, "NONE", layer_name_or_id="table") is True

    def test_invalid_geometry_type(self, valid_vector):
        """Test with invalid geometry type."""
        with pytest.raises(ValueError):
            check_vector_is_geometry_type(valid_vector, "INVALID_TYPE")

    def test_multilayer_same_type(self, multilayer_vector):
        """Test multiple layers with same geometry type."""
        assert check_vector_is_geometry_type(multilayer_vector, "3D POINT") is True
        assert check_vector_is_geometry_type(multilayer_vector, ["3D POINT"]) is True
        assert check_vector_is_geometry_type(multilayer_vector, "3D POLYGON") is False

    def test_type_parameter_validation(self, valid_vector):
        """Test validation of geometry_type parameter."""
        with pytest.raises(TypeError):
            check_vector_is_geometry_type(valid_vector, 123)

        with pytest.raises(TypeError):
            check_vector_is_geometry_type(valid_vector, None)


class TestCheckVectorIsPointType:
    def test_point_type(self, valid_vector):
        """Test checking if vector is point type."""
        assert check_vector_is_point_type(valid_vector) is True

    def test_non_point_type(self, complex_vector):
        """Test checking if vector is not point type."""
        assert check_vector_is_point_type(complex_vector) is False

    def test_specific_layer(self, multitype_vector):
        """Test checking point type in specific layer."""
        assert check_vector_is_point_type(multitype_vector, layer_name_or_id="points") is True
        assert check_vector_is_point_type(multitype_vector, layer_name_or_id="lines") is False

    def test_invalid_layer(self, multitype_vector):
        """Test checking point type in invalid layer."""
        with pytest.raises(ValueError):
            check_vector_is_point_type(multitype_vector, layer_name_or_id="nonexistent")


class TestCheckVectorIsLineType:
    def test_line_type(self, multitype_vector):
        """Test checking if vector is line type."""
        assert check_vector_is_line_type(multitype_vector, layer_name_or_id="lines") is True

    def test_non_line_type(self, valid_vector):
        """Test checking if vector is not line type."""
        assert check_vector_is_line_type(valid_vector) is False

    def test_specific_layer(self, multitype_vector):
        """Test checking line type in specific layer."""
        assert check_vector_is_line_type(multitype_vector, layer_name_or_id="lines") is True
        assert check_vector_is_line_type(multitype_vector, layer_name_or_id="points") is False

    def test_invalid_layer(self, multitype_vector):
        """Test checking line type in invalid layer."""
        with pytest.raises(ValueError):
            check_vector_is_line_type(multitype_vector, layer_name_or_id="nonexistent")


class TestCheckVectorIsPolygonType:
    def test_polygon_type(self, complex_vector):
        """Test checking if vector is polygon type."""
        assert check_vector_is_polygon_type(complex_vector, layer_name_or_id="polygons") is True

    def test_non_polygon_type(self, valid_vector):
        """Test checking if vector is not polygon type."""
        assert check_vector_is_polygon_type(valid_vector) is False

    def test_specific_layer(self, multitype_vector):
        """Test checking polygon type in specific layer."""
        assert check_vector_is_polygon_type(multitype_vector, layer_name_or_id="polygons") is True
        assert check_vector_is_polygon_type(multitype_vector, layer_name_or_id="points") is False

    def test_invalid_layer(self, multitype_vector):
        """Test checking polygon type in invalid layer."""
        with pytest.raises(ValueError):
            check_vector_is_polygon_type(multitype_vector, layer_name_or_id="nonexistent")


class TestCheckVectorHasInvalidGeometry:
    def test_valid_geometry(self, valid_vector):
        """Test vector with valid geometry."""
        assert check_vector_has_invalid_geometry(valid_vector) is False

    def test_invalid_geometry(self, tmp_path):
        """Test vector with invalid geometry."""
        vector_path = tmp_path / "invalid_geometry.gpkg"
        driver = ogr.GetDriverByName('GPKG')
        ds = driver.CreateDataSource(str(vector_path))
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        layer = ds.CreateLayer('test', srs, ogr.wkbPolygon)

        # Add invalid geometry
        feature = ogr.Feature(layer.GetLayerDefn())
        ring = ogr.Geometry(ogr.wkbLinearRing)
        ring.AddPoint(0, 0)
        ring.AddPoint(0, 1)
        ring.AddPoint(1, 1)
        ring.AddPoint(1, 0)  # Invalid polygon (not closed)
        poly = ogr.Geometry(ogr.wkbPolygon)
        poly.AddGeometry(ring)
        feature.SetGeometry(poly)
        layer.CreateFeature(feature)

        ds.FlushCache()
        ds = None

        assert check_vector_has_invalid_geometry(str(vector_path)) is True

    def test_empty_geometry(self, tmp_path):
        """Test vector with empty geometry."""
        vector_path = tmp_path / "empty_geometry.gpkg"
        driver = ogr.GetDriverByName('GPKG')
        ds = driver.CreateDataSource(str(vector_path))
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        layer = ds.CreateLayer('test', srs, ogr.wkbPoint)

        # Add empty geometry
        feature = ogr.Feature(layer.GetLayerDefn())

        layer.CreateFeature(feature)

        ds.FlushCache()
        ds = None

        assert check_vector_has_invalid_geometry(str(vector_path)) is True
        assert check_vector_has_invalid_geometry(str(vector_path), allow_empty=True) is False

    def test_specific_layer(self, complex_vector):
        """Test checking invalid geometry in specific layer."""
        assert check_vector_has_invalid_geometry(complex_vector, layer_name_or_id="points") is False
        assert check_vector_has_invalid_geometry(complex_vector, layer_name_or_id="polygons") is False

        with pytest.raises(ValueError):
            check_vector_has_invalid_geometry(complex_vector, layer_name_or_id="nonexistent")


class TestCheckVectorHasMultipleLayers:
    def test_multiple_layers(self, multitype_vector):
        """Test vector with multiple layers."""
        assert check_vector_has_multiple_layers(multitype_vector) is True

    def test_single_layer(self, valid_vector):
        """Test vector with a single layer."""
        assert check_vector_has_multiple_layers(valid_vector) is False

    def test_multiple_layers_same_type(self, multilayer_vector):
        """Test vector with multiple layers of the same type."""
        assert check_vector_has_multiple_layers(multilayer_vector) is True


class TestVectorIsValid:
    def test_valid_vector(self, valid_vector):
        """Test a valid vector with default criteria."""
        assert check_vector_is_valid(valid_vector) is True

    def test_vector_without_geometry(self, vector_no_geometry):
        """Test a vector without geometry."""
        assert check_vector_is_valid(vector_no_geometry) is False

    def test_vector_without_crs(self, vector_no_geometry):
        """Test a vector without CRS."""
        assert check_vector_is_valid(vector_no_geometry, has_geometry=False) is False

    def test_vector_with_attributes(self, valid_vector):
        """Test a vector with attributes."""
        assert check_vector_is_valid(valid_vector, has_attributes=True) is True

    def test_vector_with_multiple_layers(self, multitype_vector):
        """Test a vector with multiple layers."""
        assert check_vector_is_valid(
            multitype_vector,
            layer_name_or_id="points",
            has_geometry=False,
            empty_geometries_valid=True
        ) is True

    def test_vector_with_mixed_criteria(self, valid_vector):
        """Test a vector with mixed validation criteria."""
        assert check_vector_is_valid(valid_vector, has_geometry=True, has_crs=True, has_attributes=True, has_valid_geometry=True) is True
