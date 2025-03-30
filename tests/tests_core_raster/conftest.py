"""Fixtures for core_raster tests."""

import pytest
import numpy as np
import os
from osgeo import gdal, osr

@pytest.fixture
def sample_raster_array_2d():
    """Create a sample 2D array for raster tests.
    
    Returns:
        numpy.ndarray: 2D numpy array with sample data
    """
    return np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12]
    ], dtype=np.int32)

@pytest.fixture
def sample_raster_array_3d():
    """Create a sample 3D array for raster tests.
    
    Returns:
        numpy.ndarray: 3D numpy array with sample data (3 bands)
    """
    return np.array([
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
        [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]],
        [[25, 26, 27, 28], [29, 30, 31, 32], [33, 34, 35, 36]]
    ], dtype=np.int32)

@pytest.fixture
def sample_raster_file(tmp_path, sample_raster_array_2d):
    """Create a sample GeoTIFF raster file.
    
    Returns:
        str: Path to the created raster file
    """
    raster_path = str(tmp_path / "sample_raster.tif")
    
    # Create the raster file
    driver = gdal.GetDriverByName('GTiff')
    rows, cols = sample_raster_array_2d.shape
    dataset = driver.Create(raster_path, cols, rows, 1, gdal.GDT_Int32)
    
    # Set geotransform and projection
    dataset.SetGeoTransform((0, 10, 0, 0, 0, -10))  # 10m pixel size
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    dataset.SetProjection(srs.ExportToWkt())
    
    # Write data
    band = dataset.GetRasterBand(1)
    band.WriteArray(sample_raster_array_2d)
    band.SetNoDataValue(-9999)
    
    # Close the dataset to write to disk
    dataset = None
    
    return raster_path

@pytest.fixture
def sample_multiband_raster_file(tmp_path, sample_raster_array_3d):
    """Create a sample multiband GeoTIFF raster file.
    
    Returns:
        str: Path to the created multiband raster file
    """
    raster_path = str(tmp_path / "sample_multiband_raster.tif")
    
    # Create the raster file
    bands, rows, cols = sample_raster_array_3d.shape
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(raster_path, cols, rows, bands, gdal.GDT_Int32)
    
    # Set geotransform and projection
    dataset.SetGeoTransform((0, 10, 0, 0, 0, -10))  # 10m pixel size
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    dataset.SetProjection(srs.ExportToWkt())
    
    # Write data for each band
    for i in range(bands):
        band = dataset.GetRasterBand(i + 1)
        band.WriteArray(sample_raster_array_3d[i])
        band.SetNoDataValue(-9999)
    
    # Close the dataset to write to disk
    dataset = None
    
    return raster_path

@pytest.fixture
def raster_with_nodata(tmp_path):
    """Create a raster with NoData values.
    
    Returns:
        str: Path to the created raster with NoData values
    """
    raster_path = str(tmp_path / "raster_with_nodata.tif")
    
    # Create sample array with NoData values
    arr = np.array([
        [1, 2, -9999, 4],
        [5, -9999, 7, 8],
        [-9999, 10, 11, -9999]
    ], dtype=np.int32)
    
    # Create the raster file
    rows, cols = arr.shape
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(raster_path, cols, rows, 1, gdal.GDT_Int32)
    
    # Set geotransform and projection
    dataset.SetGeoTransform((0, 10, 0, 0, 0, -10))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    dataset.SetProjection(srs.ExportToWkt())
    
    # Write data
    band = dataset.GetRasterBand(1)
    band.WriteArray(arr)
    band.SetNoDataValue(-9999)
    
    # Close the dataset to write to disk
    dataset = None
    
    return raster_path
