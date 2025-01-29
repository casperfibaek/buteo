# pylint: skip-file
# type: ignore

import pytest
import numpy as np
from osgeo import gdal, osr
from buteo.raster.resample import raster_resample, resample_array


@pytest.fixture
def test_raster(tmp_path):
    """Create a test raster in EPSG:3857."""
    raster_path = tmp_path / "test_3857.tif"

    # Create a simple raster in WGS84
    driver = gdal.GetDriverByName('GTiff')
    ds = driver.Create(str(raster_path), 10, 10, 1, gdal.GDT_Float32)

    # Set projection to WGS84
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(3857)
    ds.SetProjection(srs.ExportToWkt())

    # Set geotransform (roughly covering Denmark - if it had been latlng)
    ds.SetGeoTransform([8.0, 0.1, 0, 57.0, 0, -0.1])

    # Fill with sample data
    data = np.ones((10, 10))
    ds.GetRasterBand(1).WriteArray(data)
    ds.GetRasterBand(1).SetNoDataValue(-9999)

    ds = None
    return str(raster_path)


@pytest.fixture 
def test_array():
    return np.ones((100, 100))

def test_raster_resample_size(test_raster):
    # Test resampling to specific size
    resampled = raster_resample(test_raster, [50, 50], target_in_pixels=True)
    ds = gdal.Open(resampled)
    assert ds.RasterXSize == 50
    assert ds.RasterYSize == 50

def test_raster_resample_resolution(test_raster):
    # Test resampling to specific resolution
    resampled = raster_resample(test_raster, [0.2, 0.2])
    ds = gdal.Open(resampled)
    assert ds.RasterXSize == 5
    assert ds.RasterYSize == 5

def test_raster_resample_methods(test_raster):
    # Test different resampling methods
    methods = ['nearest', 'bilinear', 'cubic', 'average']
    for method in methods:
        resampled = raster_resample(test_raster, [50, 50], 
                                  target_in_pixels=True,
                                  resample_alg=method)
        assert gdal.Open(resampled) is not None

def test_raster_resample_nodata(test_raster):
    # Test nodata handling
    resampled = raster_resample(test_raster, [50, 50],
                              target_in_pixels=True,
                              dst_nodata=-9999)
    ds = gdal.Open(resampled)
    assert ds.GetRasterBand(1).GetNoDataValue() == -9999

def test_resample_array_2d(test_array):
    # Test array resampling 2D
    resampled = resample_array(test_array, (50, 50))
    assert resampled.shape == (1, 50, 50)

def test_resample_array_3d():
    # Test array resampling 3D
    array = np.ones((3, 100, 100))
    resampled = resample_array(array, (50, 50))
    assert resampled.shape == (3, 50, 50)

def test_invalid_inputs():
    # Test invalid inputs raise appropriate errors
    with pytest.raises(ValueError):
        raster_resample("not_a_raster", [50, 50])
    
    with pytest.raises(AssertionError):
        resample_array(np.ones(4), (50, 50))

def test_raster_resample_dtype(test_raster):
    # Test output dtype specification
    resampled = raster_resample(test_raster, [50, 50],
                              target_in_pixels=True,
                              dtype='uint8')
    ds = gdal.Open(resampled)
    assert ds.GetRasterBand(1).DataType == gdal.GDT_Byte
