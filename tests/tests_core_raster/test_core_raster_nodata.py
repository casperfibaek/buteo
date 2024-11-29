# pylint: skip-file
# type: ignore

# Standard library
import sys; sys.path.append("../../")

import pytest
import numpy as np
from osgeo import gdal
from uuid import uuid4

from buteo.core_raster.core_raster_nodata import (
    check_rasters_have_same_nodata,
    raster_has_nodata,
    raster_get_nodata,
    raster_set_nodata
)



@pytest.fixture
def raster_without_nodata(tmp_path):
    """Create a sample raster without nodata values."""
    raster_path = tmp_path / f"no_nodata_{uuid4().hex}.tif"
    driver = gdal.GetDriverByName('GTiff')
    ds = driver.Create(str(raster_path), 10, 10, 1, gdal.GDT_Float32)
    ds.SetGeoTransform([0, 1, 0, 0, 0, -1])
    band = ds.GetRasterBand(1)
    data = np.ones((10, 10), dtype=np.float32)
    band.WriteArray(data)
    ds.FlushCache()
    ds = None
    return str(raster_path)

@pytest.fixture
def raster_with_nodata(tmp_path):
    """Create a sample raster with nodata values."""
    raster_path = tmp_path / f"with_nodata_{uuid4().hex}.tif"
    driver = gdal.GetDriverByName('GTiff')
    ds = driver.Create(str(raster_path), 10, 10, 1, gdal.GDT_Float32)
    ds.SetGeoTransform([0, 1, 0, 0, 0, -1])
    band = ds.GetRasterBand(1)
    data = np.ones((10, 10), dtype=np.float32)
    data[0, 0] = -9999
    band.WriteArray(data)
    band.SetNoDataValue(-9999)
    ds.FlushCache()
    ds = None
    return str(raster_path)

@pytest.fixture
def raster_with_different_nodata(tmp_path):
    """Create a sample raster with different nodata value."""
    raster_path = tmp_path / f"different_nodata_{uuid4().hex}.tif"
    driver = gdal.GetDriverByName('GTiff')
    ds = driver.Create(str(raster_path), 10, 10, 1, gdal.GDT_Float32)
    ds.SetGeoTransform([0, 1, 0, 0, 0, -1])
    band = ds.GetRasterBand(1)
    data = np.ones((10, 10), dtype=np.float32)
    data[0, 0] = -1
    band.WriteArray(data)
    band.SetNoDataValue(-1)
    ds.FlushCache()
    ds = None
    return str(raster_path)

class TestCheckRastersHaveSameNodata:
    def test_single_raster(self, raster_with_nodata):
        """Test checking nodata values for a single raster."""
        assert check_rasters_have_same_nodata([raster_with_nodata]) is True

    def test_same_nodata(self, raster_with_nodata):
        """Test checking nodata values for multiple rasters with same nodata."""
        assert check_rasters_have_same_nodata([raster_with_nodata, raster_with_nodata]) is True

    def test_different_nodata(self, raster_with_nodata, raster_with_different_nodata):
        """Test checking nodata values for rasters with different nodata values."""
        assert check_rasters_have_same_nodata([raster_with_nodata, raster_with_different_nodata]) is False

    def test_empty_list(self):
        """Test with empty list."""
        with pytest.raises(ValueError):
            check_rasters_have_same_nodata([])

    def test_invalid_input(self):
        """Test with invalid input."""
        with pytest.raises(TypeError):
            check_rasters_have_same_nodata("not_a_list")

class TestRasterHasNodata:
    def test_raster_with_nodata(self, raster_with_nodata):
        """Test raster with nodata values."""
        assert raster_has_nodata(raster_with_nodata) is True

    def test_raster_without_nodata(self, raster_without_nodata):
        """Test raster without nodata values."""
        assert raster_has_nodata(raster_without_nodata) is False

    def test_multiple_rasters(self, raster_with_nodata, raster_without_nodata):
        """Test multiple rasters."""
        results = raster_has_nodata([raster_with_nodata, raster_without_nodata])
        assert results == [True, False]

    def test_invalid_input(self):
        """Test invalid input."""
        with pytest.raises(TypeError):
            raster_has_nodata(123)

class TestRasterGetNodata:
    def test_get_nodata_value(self, raster_with_nodata):
        """Test getting nodata value."""
        assert raster_get_nodata(raster_with_nodata) == -9999

    def test_get_different_nodata(self, raster_with_different_nodata):
        """Test getting different nodata value."""
        assert raster_get_nodata(raster_with_different_nodata) == -1

    def test_multiple_rasters(self, raster_with_nodata, raster_with_different_nodata):
        """Test getting nodata values from multiple rasters."""
        results = raster_get_nodata([raster_with_nodata, raster_with_different_nodata])
        assert results == [-9999, -1]

    def test_invalid_input(self):
        """Test invalid input."""
        with pytest.raises(TypeError):
            raster_get_nodata(123)

class TestRasterSetNodata:
    def test_set_nodata(self, raster_without_nodata, tmp_path):
        """Test setting nodata value."""
        output_path = tmp_path / "set_nodata.tif"
        result = raster_set_nodata(
            raster_without_nodata,
            -9999,
            out_path=str(output_path),
            in_place=False
        )
        ds = gdal.Open(result)
        assert ds.GetRasterBand(1).GetNoDataValue() == -9999
        ds = None

    def test_remove_nodata(self, raster_with_nodata, tmp_path):
        """Test removing nodata value."""
        output_path = tmp_path / "remove_nodata.tif"
        result = raster_set_nodata(
            raster_with_nodata,
            None,
            out_path=str(output_path),
            in_place=False
        )
        ds = gdal.Open(result)
        assert ds.GetRasterBand(1).GetNoDataValue() is None
        ds = None

    def test_in_place_modification(self, raster_without_nodata):
        """Test in-place nodata modification."""
        result = raster_set_nodata(raster_without_nodata, -9999, in_place=True)
        ds = gdal.Open(result)
        assert ds.GetRasterBand(1).GetNoDataValue() == -9999
        ds = None

    def test_multiple_rasters(self, raster_without_nodata, raster_with_nodata, tmp_path):
        """Test setting nodata for multiple rasters."""
        output_paths = [
            tmp_path / "multi1.tif",
            tmp_path / "multi2.tif"
        ]
        results = raster_set_nodata(
            [raster_without_nodata, raster_with_nodata],
            -9999,
            out_path=[str(p) for p in output_paths],
            in_place=False
        )
        for path in results:
            ds = gdal.Open(path)
            assert ds.GetRasterBand(1).GetNoDataValue() == -9999
            ds = None

    def test_invalid_inputs(self, raster_without_nodata):
        """Test invalid inputs."""
        with pytest.raises(TypeError):
            raster_set_nodata(123, -9999)
        with pytest.raises(TypeError):
            raster_set_nodata(raster_without_nodata, "invalid")
