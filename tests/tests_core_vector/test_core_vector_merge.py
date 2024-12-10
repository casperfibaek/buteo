# pylint: skip-file
# type: ignore

# Standard library
import sys; sys.path.append("../../")

import pytest
from osgeo import ogr, osr

from buteo.core_vector.core_vector_merge import vector_merge_features
@pytest.fixture
def create_test_vectors(tmp_path):
    """Create multiple test vector files."""
    vectors = []
    for i in range(3):
        vector_path = tmp_path / f"test_{i}.gpkg"
        driver = ogr.GetDriverByName('GPKG')
        ds = driver.CreateDataSource(str(vector_path))
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        layer = ds.CreateLayer('test', srs, ogr.wkbPolygon)
        
        # Add fields 
        layer.CreateField(ogr.FieldDefn('id', ogr.OFTInteger))
        layer.CreateField(ogr.FieldDefn('name', ogr.OFTString))

        # Add a feature
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
        feature.SetField('id', i)
        feature.SetField('name', f'feature_{i}')
        layer.CreateFeature(feature)
        vectors.append(str(vector_path))
        ds = None

    return vectors

class TestVectorMergeFeatures:
    def test_basic_merge(self, create_test_vectors, tmp_path):
        """Test basic merging functionality."""
        out_path = tmp_path / "merged.gpkg"
        result = vector_merge_features(create_test_vectors, str(out_path))
        
        ds = ogr.Open(result)
        layer = ds.GetLayer()
        assert layer.GetFeatureCount() == 3
        ds = None

    def test_merge_with_projection(self, create_test_vectors, tmp_path):
        """Test merging with specific projection."""
        out_path = tmp_path / "merged_proj.gpkg"
        result = vector_merge_features(
            create_test_vectors, 
            str(out_path),
            projection="EPSG:3857"
        )
        
        ds = ogr.Open(result)
        layer = ds.GetLayer()
        srs = layer.GetSpatialRef()
        assert srs.GetAuthorityCode(None) == "3857"
        ds = None

    def test_merge_no_output_path(self, create_test_vectors):
        """Test merging without specifying output path."""
        result = vector_merge_features(create_test_vectors)
        ds = ogr.Open(result)
        assert ds is not None
        ds = None

    def test_merge_empty_input(self):
        """Test merging with empty input."""
        with pytest.raises(ValueError):
            vector_merge_features([])

    def test_merge_invalid_path(self, tmp_path):
        """Test merging with invalid input path."""
        with pytest.raises(ValueError):
            vector_merge_features([str(tmp_path / "nonexistent.gpkg")])

    def test_merge_overwrite(self, create_test_vectors, tmp_path):
        """Test overwrite functionality."""
        out_path = tmp_path / "merged.gpkg"
        
        # Create first merge
        vector_merge_features(create_test_vectors, str(out_path))
        
        # Try overwriting with overwrite=False
        with pytest.raises(ValueError):
            vector_merge_features(
                create_test_vectors, 
                str(out_path), 
                overwrite=False
            )

        # Test successful overwrite
        result = vector_merge_features(
            create_test_vectors, 
            str(out_path), 
            overwrite=True
        )
        assert result == str(out_path)
