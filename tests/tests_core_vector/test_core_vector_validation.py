# pylint: skip-file
# type: ignore

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
