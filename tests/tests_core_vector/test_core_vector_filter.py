# pylint: skip-file
# type: ignore

# Standard library
import sys; sys.path.append("../../")
import os

import pytest
from osgeo import ogr, osr

from buteo.core_vector.core_vector_filter import (
    vector_filter_by_function,
)



@pytest.fixture
def sample_vector(tmp_path):
    """Create a sample vector with multiple features and attributes."""
    vector_path = tmp_path / "sample.gpkg"
    driver = ogr.GetDriverByName('GPKG')
    ds = driver.CreateDataSource(str(vector_path))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    
    layer = ds.CreateLayer('test', srs, ogr.wkbPolygon)
    
    # Add fields
    field_defn = ogr.FieldDefn("value", ogr.OFTInteger)
    layer.CreateField(field_defn)
    field_defn = ogr.FieldDefn("name", ogr.OFTString)
    layer.CreateField(field_defn)

    # Create features with different values
    for i in range(4):
        wkt = f"POLYGON (({i} {i}, {i} {i+1}, {i+1} {i+1}, {i+1} {i}, {i} {i}))"
        poly = ogr.CreateGeometryFromWkt(wkt)
        
        feature = ogr.Feature(layer.GetLayerDefn())
        feature.SetField("value", i)
        feature.SetField("name", f"feature_{i}")
        feature.SetGeometry(poly)
        layer.CreateFeature(feature)
    
    ds = None
    return str(vector_path)

class TestVectorFilterByFunction:
    def test_filter_by_attribute(self, sample_vector, tmp_path):
        """Test filtering features based on attribute values."""
        out_path = str(tmp_path / "filtered_attr.gpkg")
        
        # Filter features where value > 1
        result = vector_filter_by_function(
            sample_vector,
            out_path,
            filter_function_attr=lambda x: x["value"] > 1
        )
        
        # Verify result
        ds = ogr.Open(result)
        layer = ds.GetLayer()
        assert layer.GetFeatureCount() == 2
        
        # Check filtered values
        for feature in layer:
            assert feature.GetField("value") > 1
        ds = None

    def test_filter_by_geometry(self, sample_vector, tmp_path):
        """Test filtering features based on geometry properties."""
        out_path = str(tmp_path / "filtered_geom.gpkg")
        
        # Filter features based on geometry area
        result = vector_filter_by_function(
            sample_vector,
            out_path,
            filter_function_geom=lambda geom: geom.Area() > 1.0
        )
        
        # Verify result
        ds = ogr.Open(result)
        layer = ds.GetLayer()
        for feature in layer:
            assert feature.GetGeometryRef().Area() > 1.0
        ds = None

    def test_filter_combined(self, sample_vector, tmp_path):
        """Test filtering using both attribute and geometry conditions."""
        out_path = str(tmp_path / "filtered_combined.gpkg")
        
        result = vector_filter_by_function(
            sample_vector,
            out_path,
            filter_function_attr=lambda x: x["value"] > 1,
            filter_function_geom=lambda geom: geom.Area() > 1.0
        )
        
        ds = ogr.Open(result)
        layer = ds.GetLayer()
        for feature in layer:
            assert feature.GetField("value") > 1
            assert feature.GetGeometryRef().Area() > 1.0
        ds = None

    def test_inplace_filter(self, sample_vector):
        """Test inplace filtering of features."""
        og_ds = ogr.Open(sample_vector)
        og_lyr = og_ds.GetLayer()
        og_lyr.ResetReading()
        original_count = og_lyr.GetFeatureCount()
        
        result = vector_filter_by_function(
            sample_vector,
            inplace=True,
            filter_function_attr=lambda x: x["value"] > 2
        )
        
        ds = ogr.Open(result)
        layer = ds.GetLayer()
        assert layer.GetFeatureCount() < original_count
        for feature in layer:
            assert feature.GetField("value") > 2
        ds = None

    def test_invalid_filter_input(self, sample_vector):
        """Test error handling for invalid filter inputs."""
        with pytest.raises(ValueError):
            vector_filter_by_function(
                sample_vector,
                filter_function_attr=None,
                filter_function_geom=None
            )

    def test_memory_output(self, sample_vector):
        """Test output to memory."""
        result = vector_filter_by_function(
            sample_vector,
            filter_function_attr=lambda x: x["value"] > 0
        )
        
        assert "/vsimem/" in result
        ds = ogr.Open(result)
        assert ds is not None
        ds = None

    def test_overwrite_existing(self, sample_vector, tmp_path):
        """Test overwrite functionality."""
        out_path = str(tmp_path / "overwrite.gpkg")
        
        # Create dummy file
        with open(out_path, 'w') as f:
            f.write("dummy")
        
        # Should raise error without overwrite
        with pytest.raises(FileExistsError):
            vector_filter_by_function(
                sample_vector,
                out_path,
                filter_function_attr=lambda x: x["value"] > 0,
                overwrite=False
            )
        
        # Should succeed with overwrite
        result = vector_filter_by_function(
            sample_vector,
            out_path,
            filter_function_attr=lambda x: x["value"] > 0,
            overwrite=True
        )
        
        assert os.path.exists(result)
        assert ogr.Open(result) is not None
