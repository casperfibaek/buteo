# pylint: skip-file
# type: ignore

# Standard library
import sys; sys.path.append("../../")
import os

import pytest
from osgeo import ogr, osr

from buteo.core_vector.core_vector_fixgeometry import (
    _vector_fix_geometry,
)


@pytest.fixture
def invalid_vector(tmp_path):
    """Create a vector with invalid geometry."""
    vector_path = tmp_path / "invalid.gpkg"
    driver = ogr.GetDriverByName('GPKG')
    ds = driver.CreateDataSource(str(vector_path))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    layer = ds.CreateLayer('test', srs, ogr.wkbPolygon)

    # Create an invalid polygon (self-intersecting)
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
def multilayer_invalid_vector(tmp_path):
    """Create a vector with multiple layers containing invalid geometries."""
    vector_path = tmp_path / "multilayer_invalid.gpkg"
    driver = ogr.GetDriverByName('GPKG')
    ds = driver.CreateDataSource(str(vector_path))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)

    # First layer with invalid polygon
    layer1 = ds.CreateLayer('layer1', srs, ogr.wkbPolygon)
    ring1 = ogr.Geometry(ogr.wkbLinearRing)
    ring1.AddPoint(0, 0)
    ring1.AddPoint(2, 2)
    ring1.AddPoint(0, 2)
    ring1.AddPoint(2, 0)
    ring1.AddPoint(0, 0)
    poly1 = ogr.Geometry(ogr.wkbPolygon)
    poly1.AddGeometry(ring1)
    feature1 = ogr.Feature(layer1.GetLayerDefn())
    feature1.SetGeometry(poly1)
    layer1.CreateFeature(feature1)

    # Second layer with valid polygon
    layer2 = ds.CreateLayer('layer2', srs, ogr.wkbPolygon)
    ring2 = ogr.Geometry(ogr.wkbLinearRing)
    ring2.AddPoint(0, 0)
    ring2.AddPoint(0, 1)
    ring2.AddPoint(1, 1)
    ring2.AddPoint(1, 0)
    ring2.AddPoint(0, 0)
    poly2 = ogr.Geometry(ogr.wkbPolygon)
    poly2.AddGeometry(ring2)
    feature2 = ogr.Feature(layer2.GetLayerDefn())
    feature2.SetGeometry(poly2)
    layer2.CreateFeature(feature2)
    
    ds = None
    return str(vector_path)

class TestVectorFixGeometry:
    def test_fix_invalid_geometry(self, invalid_vector, tmp_path):
        """Test fixing an invalid geometry."""
        out_path = str(tmp_path / "fixed.gpkg")
        result = _vector_fix_geometry(invalid_vector, out_path)

        # Check if output exists
        assert os.path.exists(result)
        
        # Check if geometry is valid
        ds = ogr.Open(result)
        layer = ds.GetLayer()
        feature = layer.GetNextFeature()
        assert feature.GetGeometryRef().IsValid()
        ds = None

    def test_fix_geometry_no_invalid(self, tmp_path):
        """Test fixing a vector with no invalid geometries."""
        # Create valid vector
        vector_path = tmp_path / "valid.gpkg"
        driver = ogr.GetDriverByName('GPKG')
        ds = driver.CreateDataSource(str(vector_path))
        layer = ds.CreateLayer('test', None, ogr.wkbPolygon)
        
        ring = ogr.Geometry(ogr.wkbLinearRing)
        ring.AddPoint(0, 0)
        ring.AddPoint(0, 1)
        ring.AddPoint(1, 1)
        ring.AddPoint(1, 0)
        ring.AddPoint(0, 0)
        
        poly = ogr.Geometry(ogr.wkbPolygon)
        poly.AddGeometry(ring)
        
        feature = ogr.Feature(layer.GetLayerDefn())
        feature.SetGeometry(poly)
        layer.CreateFeature(feature)
        ds = None

        # Should return input path since no fix needed
        result = _vector_fix_geometry(str(vector_path), check_invalid_first=True)
        assert result == str(vector_path)

    def test_fix_multiple_layers(self, multilayer_invalid_vector, tmp_path):
        """Test fixing multiple layers in a vector."""
        out_path = str(tmp_path / "fixed_multilayer.gpkg")
        result = _vector_fix_geometry(multilayer_invalid_vector, out_path)
        
        ds = ogr.Open(result)
        assert ds.GetLayerCount() == 2
        
        # Check both layers
        for i in range(2):
            layer = ds.GetLayer(i)
            feature = layer.GetNextFeature()
            assert feature.GetGeometryRef().IsValid()
        ds = None

    def test_fix_specific_layer(self, multilayer_invalid_vector, tmp_path):
        """Test fixing a specific layer by name."""
        out_path = str(tmp_path / "fixed_specific.gpkg")
        result = _vector_fix_geometry(
            multilayer_invalid_vector,
            out_path,
            layer_name_or_id="layer1"
        )
        
        ds = ogr.Open(result)
        layer = ds.GetLayer("layer1")
        feature = layer.GetNextFeature()
        assert feature.GetGeometryRef().IsValid()
        ds = None

    def test_fix_with_output_options(self, invalid_vector, tmp_path):
        """Test fixing with various output options."""
        result = _vector_fix_geometry(
            invalid_vector,
            prefix="test_",
            suffix="_fixed",
            add_uuid=True,
            add_timestamp=True
        )
        
        assert "test_" in result
        assert "_fixed" in result
        assert ogr.Open(result) is not None

    def test_fix_memory_output(self, invalid_vector):
        """Test fixing with memory output."""
        result = _vector_fix_geometry(invalid_vector, None)
        assert "/vsimem/" in result
        assert ogr.Open(result) is not None

    def test_fix_with_overwrite(self, invalid_vector, tmp_path):
        """Test fixing with overwrite option."""
        out_path = str(tmp_path / "overwrite.gpkg")
        
        # Create dummy file
        with open(out_path, 'w') as f:
            f.write("dummy")
        
        # Should raise error without overwrite
        with pytest.raises(FileExistsError):
            _vector_fix_geometry(invalid_vector, out_path, overwrite=False)
        
        # Should succeed with overwrite
        result = _vector_fix_geometry(invalid_vector, out_path, overwrite=True)
        assert ogr.Open(result) is not None
