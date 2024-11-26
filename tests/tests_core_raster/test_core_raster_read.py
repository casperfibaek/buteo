# pylint: skip-file
# type: ignore

# Standard library
import sys; sys.path.append("../../")

import pytest
import numpy as np
from osgeo import gdal
from buteo.core_raster import core_raster_read



# Fixtures
@pytest.fixture
def test_raster(tmp_path):
    """Create a test raster file"""
    filename = str(tmp_path / "test.tif")
    driver = gdal.GetDriverByName('GTiff')
    ds = driver.Create(filename, 10, 10, 3, gdal.GDT_Float32)
    ds.SetGeoTransform([0, 1, 0, 0, 0, -1])
    
    # Add some test data
    for band in range(3):
        data = np.ones((10,10)) * (band + 1)
        ds.GetRasterBand(band + 1).WriteArray(data)
    
    ds = None
    return filename

@pytest.fixture
def mem_raster():
    """Create an in-memory raster"""
    ds = gdal.GetDriverByName('MEM').Create('', 5, 5, 1, gdal.GDT_Byte)
    data = np.ones((5,5), dtype=np.uint8)
    ds.GetRasterBand(1).WriteArray(data)
    return ds

class TestReadRasterBand:
    """Test _read_raster_band function"""

    def test_basic_read(self, test_raster):
        """Test basic band reading"""
        ds = gdal.Open(test_raster)
        data = core_raster_read._read_raster_band(ds, 1)
        
        assert isinstance(data, np.ndarray)
        assert data.shape == (10, 10)
        assert np.all(data == 1)
        ds = None

    def test_offset_read(self, test_raster):
        """Test reading with offsets"""
        ds = gdal.Open(test_raster)
        data = core_raster_read._read_raster_band(ds, 1, (2, 2, 3, 3))
        
        assert data.shape == (3, 3)
        assert np.all(data == 1)
        ds = None

    def test_invalid_band_index(self, test_raster):
        """Test invalid band index"""
        ds = gdal.Open(test_raster)
        with pytest.raises(ValueError):
            core_raster_read._read_raster_band(ds, 4)
        ds = None

class TestValidateRasterDataset:
    """Test _validate_raster_dataset function"""

    def test_no_projection(self, mem_raster):
        """Test handling of raster with no projection"""
        with pytest.warns(UserWarning):
            core_raster_read._validate_raster_dataset(mem_raster, "test", None)
        assert mem_raster.GetProjectionRef() != ""

    def test_custom_projection(self, mem_raster):
        """Test setting custom projection"""
        with pytest.warns(UserWarning):
            core_raster_read._validate_raster_dataset(mem_raster, "test", 3857)
        assert "3857" in mem_raster.GetProjectionRef()

    def test_invalid_projection(self, mem_raster):
        """Test invalid projection"""
        with pytest.raises(ValueError):
            core_raster_read._validate_raster_dataset(mem_raster, "test", "invalid_proj")

class TestOpenRaster:
    """Test open_raster function"""

    def test_single_raster(self, test_raster):
        """Test opening single raster"""
        ds = core_raster_read.open_raster(test_raster)
        assert isinstance(ds, gdal.Dataset)
        assert ds.RasterCount == 3
        ds = None

    def test_multiple_rasters(self, test_raster):
        """Test opening multiple rasters"""
        datasets = core_raster_read.open_raster([test_raster, test_raster])
        assert isinstance(datasets, list)
        assert len(datasets) == 2
        assert all(isinstance(ds, gdal.Dataset) for ds in datasets)
        for ds in datasets:
            ds = None

    def test_writeable_mode(self, test_raster):
        """Test opening in write mode"""
        ds = core_raster_read.open_raster(test_raster, writeable=True)
        assert isinstance(ds, gdal.Dataset)
        ds = None

    def test_invalid_path(self):
        """Test invalid raster path"""
        with pytest.raises(ValueError):
            core_raster_read.open_raster("nonexistent.tif")
