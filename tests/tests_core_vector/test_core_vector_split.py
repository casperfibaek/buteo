# pylint: skip-file
# type: ignore

# Standard library
import sys; sys.path.append("../../")

import pytest
from osgeo import ogr, osr

from buteo.core_vector.core_vector_split import (
    vector_split_by_feature,
    vector_split_by_attribute,
)

@pytest.fixture
def test_vector(tmp_path):
    """Create a test vector with multiple features and attributes."""
    vector_path = tmp_path / "test.gpkg"
    driver = ogr.GetDriverByName('GPKG')
    ds = driver.CreateDataSource(str(vector_path))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    layer = ds.CreateLayer('test', srs, ogr.wkbPolygon)

    # Add fields
    layer.CreateField(ogr.FieldDefn('category', ogr.OFTString))
    layer.CreateField(ogr.FieldDefn('value', ogr.OFTInteger))

    # Add features
    for i in range(3):
        feature = ogr.Feature(layer.GetLayerDefn())
        ring = ogr.Geometry(ogr.wkbLinearRing)
        ring.AddPoint(i, i)
        ring.AddPoint(i+1, i)
        ring.AddPoint(i+1, i+1)
        ring.AddPoint(i, i+1)
        ring.AddPoint(i, i)
        poly = ogr.Geometry(ogr.wkbPolygon)
        poly.AddGeometry(ring)
        feature.SetGeometry(poly)
        feature.SetField('category', f'cat{i%2}')
        feature.SetField('value', i)
        layer.CreateFeature(feature)

    ds = None
    return str(vector_path)

@pytest.fixture 
def test_vector_type(tmp_path):
    """Create a test vector with type attribute [A,A,B]."""
    vector_path = tmp_path / "test_type.gpkg"
    driver = ogr.GetDriverByName('GPKG')
    ds = driver.CreateDataSource(str(vector_path))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    layer = ds.CreateLayer('test', srs, ogr.wkbPolygon)

    # Add fields
    layer.CreateField(ogr.FieldDefn('type', ogr.OFTString))
    
    # Add features
    types = ['A', 'A', 'B']
    for i, type_val in enumerate(types):
        feature = ogr.Feature(layer.GetLayerDefn())
        ring = ogr.Geometry(ogr.wkbLinearRing)
        ring.AddPoint(i, i)
        ring.AddPoint(i+1, i)
        ring.AddPoint(i+1, i+1)
        ring.AddPoint(i, i+1)
        ring.AddPoint(i, i)
        poly = ogr.Geometry(ogr.wkbPolygon)
        poly.AddGeometry(ring)
        feature.SetGeometry(poly)
        feature.SetField('type', type_val)
        layer.CreateFeature(feature)

    ds = None
    return str(vector_path)

class TestVectorSplitByFeature:
    def test_basic_split(self, test_vector, tmp_path):
        """Test basic feature splitting functionality."""
        out_folder = tmp_path / "split_features"
        out_folder.mkdir()
        results = vector_split_by_feature(test_vector, str(out_folder))
        assert len(results) == 3
        assert all(ogr.Open(path) is not None for path in results)

    def test_split_with_prefix_suffix(self, test_vector, tmp_path):
        """Test splitting with prefix and suffix."""
        out_folder = tmp_path / "split_prefix"
        out_folder.mkdir()
        results = vector_split_by_feature(test_vector, str(out_folder), prefix="pre_", suffix="_post")
        assert all("pre_" in path for path in results)
        assert all("_post" in path for path in results)

    def test_split_with_extension(self, test_vector, tmp_path):
        """Test splitting with different output format."""
        out_folder = tmp_path / "split_format"
        out_folder.mkdir()
        results = vector_split_by_feature(test_vector, str(out_folder), extension="geojson")
        assert all(path.endswith(".geojson") for path in results)
        
    def test_split_no_folder(self, test_vector):
        """Test splitting without output folder (temp files)."""
        results = vector_split_by_feature(test_vector)
        assert len(results) == 3
        assert all(ogr.Open(path) is not None for path in results)

class TestVectorSplitByAttribute:
    def test_basic_attribute_split(self, test_vector, tmp_path):
        """Test basic attribute splitting functionality."""
        out_folder = tmp_path / "split_attr"
        out_folder.mkdir()
        results = vector_split_by_attribute(test_vector, "category", str(out_folder))
        assert len(results) == 2  # Two unique categories
        assert all(ogr.Open(path) is not None for path in results)

    def test_split_numeric_attribute(self, test_vector, tmp_path):
        """Test splitting by numeric attribute."""
        out_folder = tmp_path / "split_numeric"
        out_folder.mkdir()
        results = vector_split_by_attribute(test_vector, "value", str(out_folder))
        assert len(results) == 3  # Three unique values
        assert all(ogr.Open(path) is not None for path in results)

    def test_split_with_extension(self, test_vector, tmp_path):
        """Test attribute splitting with different output format."""
        out_folder = tmp_path / "split_attr_format"
        out_folder.mkdir()
        results = vector_split_by_attribute(test_vector, "category", str(out_folder), extension="geojson")
        assert all(path.endswith(".geojson") for path in results)

    def test_split_by_type(self, test_vector_type, tmp_path):
        """Test splitting by type attribute [A,A,B]."""
        out_folder = tmp_path / "split_type"
        out_folder.mkdir()
        results = vector_split_by_attribute(test_vector_type, "type", str(out_folder))
        assert len(results) == 2  # Two unique types (A and B)
        
        # Check feature counts in output files
        for result in results:
            ds = ogr.Open(result)
            layer = ds.GetLayer()
            if "type_A" in result:
                assert layer.GetFeatureCount() == 2  # Should have 2 'A' features
            elif "type_B" in result:
                assert layer.GetFeatureCount() == 1  # Should have 1 'B' feature
            ds = None

    def test_invalid_attribute(self, test_vector_type, tmp_path):
        """Test splitting by non-existent attribute."""
        out_folder = tmp_path / "split_invalid"
        out_folder.mkdir()
        with pytest.raises(Exception):
            vector_split_by_attribute(test_vector_type, "invalid_field", str(out_folder))