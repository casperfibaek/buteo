# pylint: skip-file
# type: ignore

import pytest
import numpy as np
from osgeo import gdal, osr

from buteo.raster.coordinates import raster_create_grid_with_coordinates


@pytest.fixture
def test_raster(tmp_path):
    """Create a test raster in EPSG:4326 (WGS84)."""
    raster_path = tmp_path / "test_4326.tif"

    # Create a simple raster in WGS84
    driver = gdal.GetDriverByName('GTiff')
    ds = driver.Create(str(raster_path), 10, 10, 1, gdal.GDT_Float32)

    # Set projection to WGS84
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    ds.SetProjection(srs.ExportToWkt())

    # Set geotransform (roughly covering Denmark)
    ds.SetGeoTransform([8.0, 0.1, 0, 57.0, 0, -0.1])

    # Fill with sample data
    data = np.ones((10, 10))
    ds.GetRasterBand(1).WriteArray(data)
    ds.GetRasterBand(1).SetNoDataValue(-9999)

    ds = None
    return str(raster_path)


def test_raster_create_grid_with_coordinates(test_raster):
    """Test creating a grid with coordinates from a raster."""
    grid = raster_create_grid_with_coordinates(test_raster)
    
    # Check the shape of the grid
    assert grid.shape == (10, 10, 2)
    
    # Check that coordinates are within the expected range
    assert np.min(grid[:, :, 0]) >= 8.0  # x min
    assert np.max(grid[:, :, 0]) <= 9.0  # x max
    assert np.min(grid[:, :, 1]) >= 55.9  # y min
    assert np.max(grid[:, :, 1]) <= 57.1  # y max
    
    # Check that the grid has the expected coordinate pattern
    # X coordinates should increase from left to right
    assert np.all(np.diff(grid[0, :, 0]) > 0)
    
    # Y coordinates should decrease from top to bottom
    assert np.all(np.diff(grid[:, 0, 1]) < 0)
