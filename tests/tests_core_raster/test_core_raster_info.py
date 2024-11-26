# pylint: skip-file
# type: ignore

# Standard library
import sys; sys.path.append("../../")

import pytest
from osgeo import gdal, osr
import numpy as np
import os

from buteo.core_raster.core_raster_info import (
    _get_basic_info_raster,
    _get_bounds_info_raster,
    get_metadata_raster,
)



def same_path(path1: str, path2: str) -> bool:
    """Check if two paths point to the same location regardless of path separator style."""
    try:
        return Path(path1).resolve() == Path(path2).resolve()
    except Exception:
        return os.path.normpath(path1) == os.path.normpath(path2)

# Fixtures
@pytest.fixture
def sample_dataset():
    """Create a simple in-memory raster dataset."""
    dataset = gdal.GetDriverByName('MEM').Create('', 100, 200, 3, gdal.GDT_Float32)
    dataset.SetGeoTransform([0, 1, 0, 0, 0, -1])
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    dataset.SetProjection(srs.ExportToWkt())
    return dataset

@pytest.fixture
def sample_file_dataset(tmp_path):
    """Create a GeoTIFF file dataset."""
    filename = str(tmp_path / "test.tif")
    dataset = gdal.GetDriverByName('GTiff').Create(filename, 100, 200, 3, gdal.GDT_Float32)
    dataset.SetGeoTransform([0, 1, 0, 0, 0, -1])
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    dataset.SetProjection(srs.ExportToWkt())
    
    # Set nodata value for first band
    band = dataset.GetRasterBand(1)
    band.SetNoDataValue(-9999)
    
    dataset.FlushCache()
    dataset = None
    
    return gdal.Open(filename), filename

@pytest.fixture
def projected_dataset():
    """Create a dataset with Web Mercator projection."""
    dataset = gdal.GetDriverByName('MEM').Create('', 100, 200, 1, gdal.GDT_Float32)
    dataset.SetGeoTransform([0, 100, 0, 0, 0, -100])
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(3857)
    dataset.SetProjection(srs.ExportToWkt())
    return dataset

class TestGetBasicInfoRaster:
    def test_basic_info(self, sample_dataset):
        """Test basic information extraction."""
        info = _get_basic_info_raster(sample_dataset)
        
        assert isinstance(info, dict)
        assert info['size'] == (100, 200)
        assert info['bands'] == 3
        assert info['dtype'] == gdal.GDT_Float32
        assert isinstance(info['projection_wkt'], str)
        assert isinstance(info['projection_osr'], osr.SpatialReference)
        assert info['transform'] == (0, 1, 0, 0, 0, -1)

    def test_invalid_input(self):
        """Test error handling for invalid input."""
        with pytest.raises(TypeError):
            _get_basic_info_raster("not_a_dataset")

class TestGetBoundsInfoRaster:
    def test_bounds_info(self, sample_dataset):
        """Test bounds information extraction."""
        info = _get_basic_info_raster(sample_dataset)
        bounds = _get_bounds_info_raster(sample_dataset, info['projection_osr'])
        
        assert isinstance(bounds, dict)
        assert 'bbox' in bounds
        assert 'bbox_latlng' in bounds
        assert 'bbox_gdal' in bounds
        assert 'bbox_gdal_latlng' in bounds
        assert 'bounds_latlng' in bounds
        assert 'bounds_raster' in bounds
        assert 'centroid' in bounds
        assert 'centroid_latlng' in bounds
        assert 'area' in bounds
        assert 'area_latlng' in bounds

    def test_projected_bounds(self, projected_dataset):
        """Test bounds calculation with projected coordinates."""
        info = _get_basic_info_raster(projected_dataset)
        bounds = _get_bounds_info_raster(projected_dataset, info['projection_osr'])

        assert isinstance(bounds['bbox'], list)
        assert isinstance(bounds['area'], float)
        assert bounds['area'] > 0

class TestGetMetadataRaster:
    def test_file_metadata(self, sample_file_dataset):
        """Test metadata extraction from file dataset."""
        dataset, filename = sample_file_dataset
        metadata = get_metadata_raster(filename)

        
        assert isinstance(metadata, dict)
        assert same_path(metadata['path'], filename)
        assert metadata['basename'] == os.path.basename(filename)
        assert metadata['ext'] == '.tif'
        assert metadata['in_memory'] is False
        assert metadata['width'] == 100
        assert metadata['height'] == 200
        assert metadata['bands'] == 3
        assert metadata['dtype'] == np.float32
        assert metadata['nodata'] is True
        assert metadata['nodata_value'] == -9999

    def test_memory_metadata(self, sample_dataset):
        """Test metadata extraction from memory dataset."""
        metadata = get_metadata_raster(sample_dataset)
        
        assert isinstance(metadata, dict)
        assert metadata['in_memory'] is True
        assert metadata['driver'] == 'MEM'
        assert metadata['width'] == 100
        assert metadata['height'] == 200

    def test_invalid_input(self):
        """Test error handling for invalid input."""
        with pytest.raises(TypeError):
            get_metadata_raster(123)

    def test_metadata_consistency(self, sample_file_dataset):
        """Test consistency between file and dataset metadata."""
        dataset, filename = sample_file_dataset
        
        file_metadata = get_metadata_raster(filename)
        dataset_metadata = get_metadata_raster(dataset)
        
        # Compare key attributes
        assert file_metadata['size'] == dataset_metadata['size']
        assert file_metadata['bands'] == dataset_metadata['bands']
        assert file_metadata['dtype'] == dataset_metadata['dtype']
        assert file_metadata['projection_wkt'] == dataset_metadata['projection_wkt']

    def test_projection_metadata(self, projected_dataset):
        """Test metadata for projected dataset."""
        metadata = get_metadata_raster(projected_dataset)
        
        assert 'projection_wkt' in metadata
        assert 'projection_osr' in metadata
        assert metadata['pixel_width'] == 100
        assert metadata['pixel_height'] == 100