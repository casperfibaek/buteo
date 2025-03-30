# pylint: skip-file
# type: ignore

import pytest
from osgeo import gdal, osr
import numpy as np
import os

from buteo.raster.reproject import (
    _find_common_projection,
    _raster_reproject,
    raster_reproject,
)

@pytest.fixture
def wgs84_raster(tmp_path):
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

@pytest.fixture
def utm_raster(tmp_path):
    """Create a test raster in EPSG:32632 (UTM 32N)."""
    raster_path = tmp_path / "test_32632.tif"

    driver = gdal.GetDriverByName('GTiff')
    ds = driver.Create(str(raster_path), 10, 10, 1, gdal.GDT_Float32)

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32632)
    ds.SetProjection(srs.ExportToWkt())

    # Set geotransform (roughly same area in UTM)
    ds.SetGeoTransform([500000, 1000, 0, 6300000, 0, -1000])

    data = np.ones((10, 10))
    ds.GetRasterBand(1).WriteArray(data)
    ds.GetRasterBand(1).SetNoDataValue(-9999)

    ds = None
    return str(raster_path)

@pytest.fixture
def rgb_raster(tmp_path):
    """Create a test RGB raster in EPSG:4326."""
    raster_path = tmp_path / "rgb_4326.tif"
    
    # Create a simple RGB raster
    driver = gdal.GetDriverByName('GTiff')
    ds = driver.Create(str(raster_path), 10, 10, 3, gdal.GDT_Byte)
    
    # Set projection to WGS84
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    ds.SetProjection(srs.ExportToWkt())
    
    # Set geotransform
    ds.SetGeoTransform([8.0, 0.1, 0, 57.0, 0, -0.1])
    
    # Fill with sample data for each band
    red = np.ones((10, 10), dtype=np.uint8) * 255
    green = np.ones((10, 10), dtype=np.uint8) * 128
    blue = np.ones((10, 10), dtype=np.uint8) * 64
    
    ds.GetRasterBand(1).WriteArray(red)
    ds.GetRasterBand(2).WriteArray(green)
    ds.GetRasterBand(3).WriteArray(blue)
    
    ds = None
    return str(raster_path)

class TestRasterReproject:
    def test_find_common_projection(self, wgs84_raster, utm_raster):
        """Test finding common projection from multiple rasters."""
        # Test single raster
        proj = _find_common_projection(wgs84_raster)
        assert isinstance(proj, osr.SpatialReference)
        assert proj.GetAuthorityCode(None) == "4326"

        # Test multiple rasters with same projection
        proj = _find_common_projection([wgs84_raster, wgs84_raster])
        assert proj.GetAuthorityCode(None) == "4326"

        # Test multiple rasters with different projections
        proj = _find_common_projection([wgs84_raster, utm_raster])
        assert proj.GetAuthorityCode(None) in ["4326", "32632"]

    def test_raster_reproject_basic(self, wgs84_raster, tmp_path):
        """Test basic raster reprojection to UTM."""
        out_path = str(tmp_path / "reprojected.tif")

        # Reproject to UTM 32N
        result = _raster_reproject(
            wgs84_raster,
            32632,
            out_path=out_path,
            resample_alg="nearest"
        )

        # Verify output exists and has correct projection
        assert os.path.exists(result)
        ds = gdal.Open(result)
        assert ds is not None

        srs = osr.SpatialReference(wkt=ds.GetProjection())
        assert srs.GetAuthorityCode(None) == "32632"
        ds = None

    def test_raster_reproject_multiple(self, wgs84_raster, tmp_path):
        """Test reprojecting multiple rasters."""
        result = raster_reproject(
            [wgs84_raster, wgs84_raster],
            4326,
            out_path=[
                str(tmp_path / "reproj1.tif"),
                str(tmp_path / "reproj2.tif")
            ]
        )

        assert isinstance(result, list)
        assert len(result) == 2
        for path in result:
            assert os.path.exists(path)
            ds = gdal.Open(path)
            srs = osr.SpatialReference(wkt=ds.GetProjection())
            assert srs.GetAuthorityCode(None) == "4326"
            ds = None

    def test_reproject_different_methods(self, wgs84_raster, tmp_path):
        """Test different resampling methods."""
        methods = ["nearest", "bilinear", "cubic", "cubicspline", "lanczos"]

        for method in methods:
            out_path = str(tmp_path / f"reproj_{method}.tif")
            result = _raster_reproject(
                wgs84_raster,
                32632,
                out_path=out_path,
                resample_alg=method
            )
            assert os.path.exists(result)

    def test_reproject_with_nodata(self, wgs84_raster, tmp_path):
        """Test reprojection with different nodata handling."""
        out_path = str(tmp_path / "reproj_nodata.tif")

        result = _raster_reproject(
            wgs84_raster,
            32632,
            out_path=out_path,
            dst_nodata=-9999
        )

        ds = gdal.Open(result)
        assert ds.GetRasterBand(1).GetNoDataValue() == -9999
        ds = None

    def test_reproject_with_creation_options(self, wgs84_raster, tmp_path):
        """Test reprojection with specific creation options."""
        out_path = str(tmp_path / "reproj_compress.tif")

        result = _raster_reproject(
            wgs84_raster,
            32632,
            out_path=out_path,
            creation_options=["COMPRESS=LZW", "TILED=YES"]
        )

        ds = gdal.Open(result)
        assert ds is not None
        ds = None

    def test_error_invalid_projection(self, wgs84_raster, tmp_path):
        """Test error handling for invalid projection."""
        out_path = str(tmp_path / "invalid.tif")

        with pytest.raises(ValueError):
            _raster_reproject(
                wgs84_raster,
                "invalid_proj",
                out_path=out_path
            )

    def test_error_invalid_resample(self, wgs84_raster, tmp_path):
        """Test error handling for invalid resampling method."""
        out_path = str(tmp_path / "invalid.tif")

        with pytest.raises(ValueError):
            _raster_reproject(
                wgs84_raster,
                4326,
                out_path=out_path,
                resample_alg="invalid_method"
            )
            
    def test_rgb_raster_reproject(self, rgb_raster, tmp_path):
        """Test reprojection of RGB raster."""
        out_path = str(tmp_path / "rgb_reprojected.tif")
        
        result = _raster_reproject(
            rgb_raster,
            3857,  # Web Mercator
            out_path=out_path
        )
        
        # Verify output exists and has 3 bands
        assert os.path.exists(result)
        ds = gdal.Open(result)
        assert ds.RasterCount == 3
        
        # Check projection
        srs = osr.SpatialReference(wkt=ds.GetProjection())
        assert srs.GetAuthorityCode(None) == "3857"
        
        # Verify band values are preserved through reprojection
        band1 = ds.GetRasterBand(1).ReadAsArray()
        band2 = ds.GetRasterBand(2).ReadAsArray()
        band3 = ds.GetRasterBand(3).ReadAsArray()
        
        # We should find 255, 128, and 64 values in bands 1, 2, and 3
        assert np.any(band1 == 255)
        assert np.any(band2 == 128) 
        assert np.any(band3 == 64)
        
        ds = None
