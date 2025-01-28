# pylint: skip-file
# type: ignore

# Standard library
import sys; sys.path.append("../../")
import pytest
import os

from osgeo import ogr, osr

from buteo.core_vector.core_vector_info import (
    _get_basic_info_vector,
    _get_bounds_info_vector,
    get_metadata_vector,
)

def paths_are_the_same(path1, path2):
    return os.path.normpath(path1) == os.path.normpath(path2)

# Fixtures
@pytest.fixture
def sample_vector():
    """Create a simple in-memory vector dataset."""
    driver = ogr.GetDriverByName('Memory')
    datasource = driver.CreateDataSource('memData')
    layer = datasource.CreateLayer('layer', geom_type=ogr.wkbPoint)
    layer.CreateField(ogr.FieldDefn('id', ogr.OFTInteger))
    layer.CreateField(ogr.FieldDefn('value', ogr.OFTReal))

    for i in range(10):
        feature = ogr.Feature(layer.GetLayerDefn())
        feature.SetField('id', i)
        feature.SetField('value', float(i) * 1.1)
        point = ogr.Geometry(ogr.wkbPoint)
        point.AddPoint(i, i)
        feature.SetGeometry(point)
        layer.CreateFeature(feature)
        feature = None

    return datasource

@pytest.fixture
def sample_file_vector(tmp_path):
    """Create a simple file-based vector dataset."""
    filename = str(tmp_path / "test.shp")
    driver = ogr.GetDriverByName('ESRI Shapefile')
    datasource = driver.CreateDataSource(filename)
    layer = datasource.CreateLayer('layer', geom_type=ogr.wkbPoint)
    layer.CreateField(ogr.FieldDefn('id', ogr.OFTInteger))
    layer.CreateField(ogr.FieldDefn('value', ogr.OFTReal))

    for i in range(10):
        feature = ogr.Feature(layer.GetLayerDefn())
        feature.SetField('id', i)
        feature.SetField('value', float(i) * 1.1)
        point = ogr.Geometry(ogr.wkbPoint)
        point.AddPoint(i, i)
        feature.SetGeometry(point)
        layer.CreateFeature(feature)
        feature = None

    return ogr.Open(filename), filename

class TestGetBasicInfoVector:
    def test_basic_info(self, sample_vector):
        """Test basic information extraction."""
        info = _get_basic_info_vector(sample_vector)
        
        assert isinstance(info, dict)
        assert info['layer_name'] == 'layer'
        assert info['feature_count'] == 10
        assert info['geom_type_name'] == 'point'
        assert isinstance(info['projection_wkt'], str)
        assert isinstance(info['projection_osr'], osr.SpatialReference)

    def test_invalid_input(self):
        """Test error handling for invalid input."""
        with pytest.raises(TypeError):
            _get_basic_info_vector("not_a_datasource")

class TestGetBoundsInfoVector:
    def test_bounds_info(self, sample_vector):
        """Test bounds information extraction."""
        info = _get_basic_info_vector(sample_vector)
        bounds = _get_bounds_info_vector(sample_vector, info['projection_osr'])
        
        assert isinstance(bounds, dict)
        assert 'bbox' in bounds
        assert 'bbox_latlng' in bounds
        assert 'bounds' in bounds
        assert 'bounds_latlng' in bounds
        assert 'centroid' in bounds
        assert 'centroid_latlng' in bounds
        assert 'area_bbox' in bounds
        assert 'area_latlng' in bounds

    def test_invalid_input(self):
        """Test error handling for invalid input."""
        with pytest.raises(TypeError):
            _get_bounds_info_vector("not_a_datasource", osr.SpatialReference())

class TestGetMetadataVector:
    def test_file_metadata(self, sample_file_vector):
        """Test metadata extraction from file dataset."""
        datasource, filename = sample_file_vector
        metadata = get_metadata_vector(datasource)
        
        assert isinstance(metadata, dict)

        assert paths_are_the_same(metadata['path'], filename)
        assert metadata['basename'] == os.path.basename(filename)
        assert metadata['ext'] == '.shp'
        assert metadata['in_memory'] is False
        assert metadata['driver'] == 'ESRI Shapefile'
        assert metadata['layer_count'] == 1

    def test_memory_metadata(self, sample_vector):
        """Test metadata extraction from memory dataset."""
        metadata = get_metadata_vector(sample_vector)
        
        assert isinstance(metadata, dict)
        assert metadata['in_memory'] is True
        assert metadata['driver'] == 'Memory'
        assert metadata['layer_count'] == 1

    def test_invalid_input(self):
        """Test error handling for invalid input."""
        with pytest.raises(TypeError):
            get_metadata_vector(123)

    def test_metadata_consistency(self, sample_file_vector):
        """Test consistency between file and dataset metadata."""
        datasource, filename = sample_file_vector
        
        file_metadata = get_metadata_vector(filename)
        dataset_metadata = get_metadata_vector(datasource)
        
        # Compare key attributes
        assert file_metadata['layer_count'] == dataset_metadata['layer_count']
        assert file_metadata['driver'] == dataset_metadata['driver']
        assert file_metadata['basename'] == dataset_metadata['basename']
