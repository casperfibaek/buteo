# pylint: skip-file
# type: ignore

# Standard library
import sys; sys.path.append("../")

import pytest
import os
import numpy as np
from osgeo import gdal, ogr, osr
from typing import List
import threading
import time
import psutil
from concurrent.futures import ThreadPoolExecutor

from buteo.utils import utils_gdal

# Fixtures
@pytest.fixture
def sample_geotiff(tmp_path):
    """Create a sample GeoTIFF file"""
    filename = str(tmp_path / "test.tif")
    driver = gdal.GetDriverByName('GTiff')
    ds = driver.Create(filename, 10, 10, 1, gdal.GDT_Byte)
    ds.SetGeoTransform([0, 1, 0, 0, 0, -1])
    ds = None
    return filename

@pytest.fixture
def sample_vector(tmp_path):
    """Create a sample vector file"""
    filename = str(tmp_path / "test.gpkg")
    driver = ogr.GetDriverByName('GPKG')
    ds = driver.CreateDataSource(filename)
    layer = ds.CreateLayer('test')
    ds = None
    return filename

@pytest.fixture
def memory_dataset():
    """Create a sample in-memory dataset"""
    return gdal.GetDriverByName('MEM').Create('', 10, 10, 1, gdal.GDT_Byte)

class TestCreationOptions:
    def test_default_creation_options(self):
        options = utils_gdal._get_default_creation_options()
        assert isinstance(options, list)
        assert "TILED=YES" in options
        assert "COMPRESS=LZW" in options

    def test_custom_creation_options(self):
        custom = ["COMPRESS=JPEG"]
        options = utils_gdal._get_default_creation_options(custom)
        assert "COMPRESS=JPEG" in options
        assert "TILED=YES" in options

    def test_invalid_options(self):
        with pytest.raises(TypeError):
            utils_gdal._get_default_creation_options("not a list")
        with pytest.raises(TypeError):
            utils_gdal._get_default_creation_options([1, 2, 3])

class TestMemoryManagement:
    def test_get_gdal_memory(self):
        result = utils_gdal.get_gdal_memory()
        assert isinstance(result, list)

    def test_clear_gdal_memory(self):
        result = utils_gdal.clear_gdal_memory()
        assert isinstance(result, bool)

    @pytest.mark.parametrize("path", [
        "/vsimem/test.tif",
        "/vsimem/test.shp",
    ])
    def test_check_is_dataset_in_memory(self, path):
        assert not utils_gdal._check_is_dataset_in_memory(path)

class TestExtensionValidation:
    @pytest.mark.parametrize("ext,expected", [
        ("tif", True),
        ("shp", True),
        ("invalid", False),
        ("", False),
        (None, False),
    ])
    def test_check_is_valid_ext(self, ext, expected):
        assert utils_gdal._check_is_valid_ext(ext) == expected

    def test_invalid_ext_type(self):
        with pytest.raises(TypeError):
            utils_gdal._check_is_valid_ext(123)

class TestDriverHandling:
    def test_get_default_driver_raster(self):
        driver = utils_gdal._get_default_driver_raster()
        assert isinstance(driver, gdal.Driver)
        assert driver.ShortName == "GTiff"

    def test_get_default_driver_vector(self):
        driver = utils_gdal._get_default_driver_vector()
        assert isinstance(driver, ogr.Driver)
        assert driver.name == "GPKG"

    @pytest.mark.parametrize("path,expected", [
        ("test.tif", "GTiff"),
        ("test.shp", "ESRI Shapefile"),
        ("test.gpkg", "GPKG"),
    ])
    def test_get_driver_name_from_path(self, path, expected):
        assert utils_gdal._get_driver_name_from_path(path) == expected

class TestDatasetValidation:
    def test_check_is_raster_empty(self, memory_dataset):
        assert not utils_gdal._check_is_raster_empty(memory_dataset)
        assert utils_gdal._check_is_raster_empty(None)

    def test_check_is_vector_empty(self, sample_vector):
        ds = ogr.Open(sample_vector)
        assert utils_gdal._check_is_vector_empty(ds)
        ds = None

    def test_invalid_inputs(self):
        with pytest.raises(TypeError):
            utils_gdal._check_is_raster_empty("not a dataset")
        with pytest.raises(TypeError):
            utils_gdal._check_is_vector_empty("not a dataset")

class TestDatasetChecks:
    def test_check_is_raster(self, sample_geotiff):
        assert utils_gdal._check_is_raster(sample_geotiff)
        assert not utils_gdal._check_is_raster("nonexistent.tif")
        assert not utils_gdal._check_is_raster(None)

    def test_check_is_vector(self, sample_vector):
        assert utils_gdal._check_is_vector(sample_vector)
        assert not utils_gdal._check_is_vector("nonexistent.shp")
        assert not utils_gdal._check_is_vector(None)

    def test_check_is_raster_or_vector(self, sample_geotiff, sample_vector):
        assert utils_gdal._check_is_raster_or_vector(sample_geotiff)
        assert utils_gdal._check_is_raster_or_vector(sample_vector)
        assert not utils_gdal._check_is_raster_or_vector("nonexistent.tif")

class TestDatasetOperations:
    def test_delete_dataset_if_in_memory(self, memory_dataset):
        path = memory_dataset.GetDescription()
        assert utils_gdal.delete_dataset_if_in_memory(memory_dataset)
        memory_dataset = None

    def test_get_path_from_dataset(self, sample_geotiff):
        ds = gdal.Open(sample_geotiff)
        path = utils_gdal._get_path_from_dataset(ds)
        assert isinstance(path, str)
        assert path == sample_geotiff
        ds = None


class TestComplexDatasets:
    @pytest.fixture
    def multiband_geotiff(self, tmp_path):
        """Create a complex multiband GeoTIFF"""
        filename = str(tmp_path / "complex.tif")
        driver = gdal.GetDriverByName('GTiff')
        ds = driver.Create(filename, 100, 100, 3, gdal.GDT_Float32)
        ds.SetGeoTransform([0, 0.5, 0, 50, 0, -0.5])
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        ds.SetProjection(srs.ExportToWkt())
        
        for band in range(3):
            data = np.random.rand(100, 100)
            ds.GetRasterBand(band + 1).WriteArray(data)
            ds.GetRasterBand(band + 1).SetNoDataValue(-9999)
        
        ds = None
        return filename

    def test_complex_raster_handling(self, multiband_geotiff):
        """Test handling of complex multiband raster"""
        ds = gdal.Open(multiband_geotiff)
        assert ds is not None
        assert ds.RasterCount == 3
        assert ds.GetRasterBand(1).DataType == gdal.GDT_Float32
        assert ds.GetProjection() != ""
        ds = None

class TestPerformance:
    @pytest.mark.benchmark
    def test_large_dataset_performance(self, tmp_path):
        """Test performance with large datasets"""
        filename = str(tmp_path / "large.tif")
        driver = gdal.GetDriverByName('GTiff')
        
        start_time = time.time()
        ds = driver.Create(filename, 5000, 5000, 1, gdal.GDT_Byte)
        data = np.random.randint(0, 255, (5000, 5000), dtype=np.uint8)
        ds.GetRasterBand(1).WriteArray(data)
        ds = None
        
        duration = time.time() - start_time
        assert duration < 10.0  # Should complete within 10 seconds

class TestMemoryLeaks:
    def get_memory_usage():
        """Helper to get current memory usage"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # MB

    def test_memory_leak_check(self):
        """Test for memory leaks in dataset operations"""
        initial_memory = self.get_memory_usage()
        
        for _ in range(100):
            ds = gdal.GetDriverByName('MEM').Create('', 1000, 1000, 1, gdal.GDT_Byte)
            utils_gdal.delete_dataset_if_in_memory(ds)
            ds = None
        
        final_memory = self.get_memory_usage()
        memory_growth = final_memory - initial_memory
        assert memory_growth < 10  # Less than 10MB growth

class TestThreadSafety:
    def test_concurrent_dataset_access(self, sample_geotiff):
        """Test thread-safe dataset operations"""
        def read_dataset(path):
            ds = gdal.Open(path)
            data = ds.GetRasterBand(1).ReadAsArray()
            ds = None
            return data.shape

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(read_dataset, sample_geotiff) for _ in range(10)]
            results = [f.result() for f in futures]
            
        assert all(shape == (10, 10) for shape in results)

class TestErrorRecovery:
    def test_corrupt_file_handling(self, tmp_path):
        """Test handling of corrupt files"""
        corrupt_file = tmp_path / "corrupt.tif"
        corrupt_file.write_bytes(b'Not a valid TIFF file')
        
        with pytest.raises(RuntimeError):
            gdal.Open(str(corrupt_file))
            
    def test_interrupted_write(self, tmp_path):
        """Test recovery from interrupted write operations"""
        filename = str(tmp_path / "interrupted.tif")
        ds = gdal.GetDriverByName('GTiff').Create(filename, 100, 100, 1, gdal.GDT_Byte)
        
        try:
            # Simulate interrupt during write
            raise KeyboardInterrupt
        except KeyboardInterrupt:
            ds = None
            
        # Verify file is properly closed
        assert os.path.exists(filename)
        new_ds = gdal.Open(filename)
        assert new_ds is not None
        new_ds = None

class TestEdgeCases:
    @pytest.mark.parametrize("invalid_path", [
        "",
        "nonexistent/path/file.tif",
        "http://invalid.url/file.tif",
        "\x00invalid.tif",
    ])
    def test_invalid_paths(self, invalid_path):
        """Test handling of invalid file paths"""
        with pytest.raises((RuntimeError, OSError)):
            gdal.Open(invalid_path)

    def test_zero_size_raster(self):
        """Test handling of zero-size rasters"""
        ds = gdal.GetDriverByName('MEM').Create('', 0, 0, 1, gdal.GDT_Byte)
        assert utils_gdal._check_is_raster_empty(ds)
        ds = None

    def test_extreme_values(self, tmp_path):
        """Test handling of extreme values"""
        filename = str(tmp_path / "extreme.tif")
        ds = gdal.GetDriverByName('GTiff').Create(filename, 10, 10, 1, gdal.GDT_Float64)
        data = np.array([[np.inf, -np.inf, np.nan]])
        ds.GetRasterBand(1).WriteArray(data)
        ds = None
        
        result = gdal.Open(filename)
        assert result is not None
        result = None
