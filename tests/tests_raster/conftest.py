"""Fixtures for raster tests."""

import pytest
import os
import numpy as np
from osgeo import gdal, osr

@pytest.fixture
def sample_dem(tmp_path):
    """Create a sample DEM raster for testing.
    
    Returns:
        str: Path to a DEM raster file
    """
    raster_path = str(tmp_path / "sample_dem.tif")
    
    # Create a simple elevation model (a hill)
    size = 100
    x = np.linspace(-5, 5, size)
    y = np.linspace(-5, 5, size)
    xx, yy = np.meshgrid(x, y)
    zz = 100 - np.sqrt(xx**2 + yy**2) * 10
    zz[zz < 0] = 0
    
    # Create the raster file
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(raster_path, size, size, 1, gdal.GDT_Float32)
    
    # Set geotransform and projection
    dataset.SetGeoTransform((0, 10, 0, 0, 0, -10))  # 10m pixel size
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    dataset.SetProjection(srs.ExportToWkt())
    
    # Write data
    band = dataset.GetRasterBand(1)
    band.WriteArray(zz)
    band.SetNoDataValue(-9999)
    
    # Close the dataset to write to disk
    dataset = None
    
    return raster_path

@pytest.fixture
def sample_orthophoto(tmp_path):
    """Create a sample RGB orthophoto for testing.
    
    Returns:
        str: Path to an RGB orthophoto raster file
    """
    raster_path = str(tmp_path / "sample_orthophoto.tif")
    
    # Create a sample RGB image with a simple pattern
    size = 50
    red = np.zeros((size, size), dtype=np.uint8)
    green = np.zeros((size, size), dtype=np.uint8)
    blue = np.zeros((size, size), dtype=np.uint8)
    
    # Create a gradient pattern
    for i in range(size):
        red[:, i] = int(255 * i / size)
        green[i, :] = int(255 * i / size)
        blue[i, i] = 255
    
    # Create the raster file
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(raster_path, size, size, 3, gdal.GDT_Byte)
    
    # Set geotransform and projection
    dataset.SetGeoTransform((0, 1, 0, 0, 0, -1))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    dataset.SetProjection(srs.ExportToWkt())
    
    # Write data
    dataset.GetRasterBand(1).WriteArray(red)  # Red band
    dataset.GetRasterBand(2).WriteArray(green)  # Green band
    dataset.GetRasterBand(3).WriteArray(blue)  # Blue band
    
    # Close the dataset to write to disk
    dataset = None
    
    return raster_path

@pytest.fixture
def unaligned_rasters(tmp_path):
    """Create a set of unaligned rasters for testing alignment functions.
    
    Returns:
        tuple: (raster1_path, raster2_path) - Paths to two unaligned rasters
    """
    raster1_path = str(tmp_path / "raster1.tif")
    raster2_path = str(tmp_path / "raster2.tif")
    
    # Create data
    data1 = np.ones((10, 10), dtype=np.float32)
    data2 = np.ones((15, 15), dtype=np.float32) * 2
    
    # Create first raster
    driver = gdal.GetDriverByName('GTiff')
    ds1 = driver.Create(raster1_path, 10, 10, 1, gdal.GDT_Float32)
    ds1.SetGeoTransform((0, 1, 0, 0, 0, -1))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    ds1.SetProjection(srs.ExportToWkt())
    ds1.GetRasterBand(1).WriteArray(data1)
    ds1 = None
    
    # Create second raster with different pixel size and origin
    ds2 = driver.Create(raster2_path, 15, 15, 1, gdal.GDT_Float32)
    ds2.SetGeoTransform((0.5, 0.5, 0, 0.5, 0, -0.5))  # Different origin and pixel size
    ds2.SetProjection(srs.ExportToWkt())
    ds2.GetRasterBand(1).WriteArray(data2)
    ds2 = None
    
    return (raster1_path, raster2_path)

@pytest.fixture
def multi_resolution_rasters(tmp_path):
    """Create a set of rasters with different resolutions for testing resampling functions.
    
    Returns:
        tuple: (high_res_path, low_res_path) - Paths to high and low resolution rasters
    """
    high_res_path = str(tmp_path / "high_res.tif")
    low_res_path = str(tmp_path / "low_res.tif")
    
    # Create data
    high_res_data = np.ones((100, 100), dtype=np.float32)
    low_res_data = np.ones((25, 25), dtype=np.float32) * 2
    
    # Create high resolution raster
    driver = gdal.GetDriverByName('GTiff')
    ds_high = driver.Create(high_res_path, 100, 100, 1, gdal.GDT_Float32)
    ds_high.SetGeoTransform((0, 1, 0, 0, 0, -1))  # 1m pixel size
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    ds_high.SetProjection(srs.ExportToWkt())
    ds_high.GetRasterBand(1).WriteArray(high_res_data)
    ds_high = None
    
    # Create low resolution raster covering the same area
    ds_low = driver.Create(low_res_path, 25, 25, 1, gdal.GDT_Float32)
    ds_low.SetGeoTransform((0, 4, 0, 0, 0, -4))  # 4m pixel size
    ds_low.SetProjection(srs.ExportToWkt())
    ds_low.GetRasterBand(1).WriteArray(low_res_data)
    ds_low = None
    
    return (high_res_path, low_res_path)
