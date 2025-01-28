# pylint: skip-file
# type: ignore


import os
import pytest
from osgeo import ogr, osr
from buteo.core_vector.core_vector_characteristics import vector_add_shapes_in_place

import sys; sys.path.append("../../")


@pytest.fixture
def simple_polygon(tmp_path):
    """Create a simple square polygon for testing."""
    vector_path = tmp_path / "square.gpkg"
    driver = ogr.GetDriverByName('GPKG')
    ds = driver.CreateDataSource(str(vector_path))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    layer = ds.CreateLayer('test', srs, ogr.wkbPolygon)
    
    # Create a 1x1 square polygon
    square_wkt = "POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))"
    square = ogr.CreateGeometryFromWkt(square_wkt)
    
    feature = ogr.Feature(layer.GetLayerDefn())
    feature.SetGeometry(square)
    layer.CreateFeature(feature)
    
    ds = None
    return str(vector_path)

@pytest.fixture
def complex_polygon(tmp_path):
    """Create a more complex polygon for testing."""
    vector_path = tmp_path / "complex.gpkg"
    driver = ogr.GetDriverByName('GPKG')
    ds = driver.CreateDataSource(str(vector_path))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    layer = ds.CreateLayer('test', srs, ogr.wkbPolygon)
    
    # Create an L-shaped polygon
    l_shape_wkt = "POLYGON ((0 0, 0 2, 1 2, 1 1, 2 1, 2 0, 0 0))"
    l_shape = ogr.CreateGeometryFromWkt(l_shape_wkt)
    
    feature = ogr.Feature(layer.GetLayerDefn())
    feature.SetGeometry(l_shape)
    layer.CreateFeature(feature)
    
    ds = None
    return str(vector_path)


def _vector_get_fields(vector_layer: ogr.Layer) -> None:
    """Internal."""
    layer_defn = vector_layer.GetLayerDefn()

    field_names = []
    field_types = []

    # Iterate through fields
    for field_index in range(layer_defn.GetFieldCount()):
        field_defn = layer_defn.GetFieldDefn(field_index)
        field_name = field_defn.GetName()
        field_type = field_defn.GetFieldTypeName(field_defn.GetType())
        field_names.append(field_name)
        field_types.append(field_type)

    return field_names, field_types

class TestVectorAddShapes:
    def test_add_all_shapes(self, simple_polygon):
        """Test adding all shape calculations to a simple polygon."""
        result = vector_add_shapes_in_place(simple_polygon)
        
        ds = ogr.Open(result, 1)
        layer = ds.GetLayer()
        feature = layer.GetNextFeature()
        
        # Check if all fields were created
        field_names = _vector_get_fields(layer)[0]
        assert "area" in field_names
        assert "perimeter" in field_names
        assert "ipq" in field_names
        assert "hull_area" in field_names
        assert "hull_peri" in field_names
        assert "hull_ratio" in field_names
        assert "compactness" in field_names
        assert "centroid_x" in field_names
        assert "centroid_y" in field_names
        
        # Check values for square (1x1)
        assert feature.GetField("area") == pytest.approx(1.0)
        assert feature.GetField("perimeter") == pytest.approx(4.0)
        assert feature.GetField("ipq") == pytest.approx(0.785398, rel=1e-5)  # pi/4
        assert feature.GetField("centroid_x") == pytest.approx(0.5)
        assert feature.GetField("centroid_y") == pytest.approx(0.5)
        
        ds = None

    def test_add_specific_shapes(self, simple_polygon):
        """Test adding only specific shape calculations."""
        shapes = ["area", "perimeter"]
        result = vector_add_shapes_in_place(simple_polygon, shapes=shapes)
        
        ds = ogr.Open(result, 1)
        layer = ds.GetLayer()
        feature = layer.GetNextFeature()
        
        field_names = _vector_get_fields(layer)[0]

        # Check if only specified fields were created
        assert "area" in field_names
        assert "perimeter" in field_names
        assert "ipq" not in field_names
        assert "hull_area" not in field_names
        
        ds = None

    def test_add_shapes_with_prefix(self, simple_polygon):
        """Test adding shapes with a prefix."""
        result = vector_add_shapes_in_place(simple_polygon, prefix="test_")
        
        ds = ogr.Open(result, 1)
        layer = ds.GetLayer()
        feature = layer.GetNextFeature()
        
        field_names = _vector_get_fields(layer)[0]
        assert "test_area" in field_names
        assert "test_perimeter" in field_names
        assert "test_centroid_x" in field_names
        
        ds = None

    def test_complex_shape_calculations(self, complex_polygon):
        """Test shape calculations on a more complex polygon."""
        result = vector_add_shapes_in_place(complex_polygon)
        
        ds = ogr.Open(result, 1)
        layer = ds.GetLayer()
        feature = layer.GetNextFeature()
        
        # L-shaped polygon area should be 3 square units
        assert feature.GetField("area") == pytest.approx(3.0)
        
        # Convex hull area should be larger than polygon area
        assert feature.GetField("hull_area") > feature.GetField("area")
        
        # Hull ratio should be less than 1 (not a convex shape)
        assert feature.GetField("hull_ratio") < 1.0
        
        ds = None

    def test_invalid_shape_type(self, simple_polygon):
        """Test error handling for invalid shape type."""
        with pytest.raises(ValueError):
            vector_add_shapes_in_place(simple_polygon, shapes=["invalid_shape"])

    def test_multiple_vectors(self, simple_polygon, complex_polygon):
        """Test processing multiple vectors at once."""
        results = vector_add_shapes_in_place([simple_polygon, complex_polygon])
        
        assert isinstance(results, list)
        assert len(results) == 2
        
        for result in results:
            assert os.path.exists(result)
            ds = ogr.Open(result)
            assert ds is not None
            ds = None

    def test_disallow_lists(self, simple_polygon, complex_polygon):
        """Test disallowing lists when allow_lists is False."""
        with pytest.raises(ValueError):
            vector_add_shapes_in_place(
                [simple_polygon, complex_polygon],
                allow_lists=False
            )