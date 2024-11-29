# pylint: skip-file
# type: ignore

# Standard library
import sys; sys.path.append("../../")

import pytest
import numpy as np
from osgeo import gdal
from pathlib import Path

from buteo.core_raster.core_raster_stack import (
    raster_stack_list,
    raster_stack_vrt_list
)



@pytest.fixture
def aligned_rasters(tmp_path):
    """Create a set of aligned single-band rasters."""
    raster_paths = []
    for i in range(3):
        path = tmp_path / f"aligned_{i}.tif"
        driver = gdal.GetDriverByName('GTiff')
        ds = driver.Create(str(path), 10, 10, 1, gdal.GDT_Float32)
        ds.SetGeoTransform([0, 1, 0, 0, 0, -1])
        band = ds.GetRasterBand(1)
        data = np.full((10, 10), i, dtype=np.float32)
        band.WriteArray(data)
        ds.FlushCache()
        ds = None
        raster_paths.append(str(path))
    return raster_paths

@pytest.fixture
def multiband_aligned_rasters(tmp_path):
    """Create aligned multi-band rasters."""
    raster_paths = []
    for i in range(2):
        path = tmp_path / f"multiband_{i}.tif"
        driver = gdal.GetDriverByName('GTiff')
        ds = driver.Create(str(path), 10, 10, 3, gdal.GDT_Float32)
        ds.SetGeoTransform([0, 1, 0, 0, 0, -1])
        for b in range(3):
            band = ds.GetRasterBand(b + 1)
            data = np.full((10, 10), i * 3 + b, dtype=np.float32)
            band.WriteArray(data)
        ds.FlushCache()
        ds = None
        raster_paths.append(str(path))
    return raster_paths

class TestRasterStackList:
    def test_basic_stack(self, aligned_rasters, tmp_path):
        """Test basic raster stacking."""
        out_path = tmp_path / "stacked.tif"
        result = raster_stack_list(
            aligned_rasters,
            str(out_path)
        )
        
        assert Path(result).exists()
        ds = gdal.Open(result)
        assert ds.RasterCount == len(aligned_rasters)
        # Verify values in each band
        for i in range(ds.RasterCount):
            data = ds.GetRasterBand(i + 1).ReadAsArray()
            assert np.all(data == i)
        ds = None

    def test_stack_with_dtype(self, aligned_rasters, tmp_path):
        """Test stacking with dtype conversion."""
        out_path = tmp_path / "stacked_int16.tif"
        result = raster_stack_list(
            aligned_rasters,
            str(out_path),
            dtype="int16"
        )
        
        ds = gdal.Open(result)
        assert ds.GetRasterBand(1).DataType == gdal.GDT_Int16
        ds = None

    def test_stack_with_creation_options(self, aligned_rasters, tmp_path):
        """Test stacking with GDAL creation options."""
        out_path = tmp_path / "stacked_compressed.tif"
        result = raster_stack_list(
            aligned_rasters,
            str(out_path),
            creation_options=["COMPRESS=LZW"]
        )
        
        ds = gdal.Open(result)
        assert ds.GetMetadataItem("COMPRESSION", "IMAGE_STRUCTURE") == "LZW"
        ds = None

    def test_invalid_inputs(self, tmp_path):
        """Test invalid inputs."""
        with pytest.raises(ValueError):
            raster_stack_list(
                [],
                str(tmp_path / "empty.tif")
            )
        with pytest.raises(TypeError):
            raster_stack_list(
                123,
                str(tmp_path / "invalid.tif")
            )

class TestRasterStackVrtList:
    def test_basic_vrt_stack(self, aligned_rasters, tmp_path):
        """Test basic VRT stacking."""
        out_path = tmp_path / "stacked.vrt"
        result = raster_stack_vrt_list(
            aligned_rasters,
            str(out_path)
        )
        
        assert Path(result).exists()
        ds = gdal.Open(result)
        assert ds.RasterCount == len(aligned_rasters)
        ds = None

    def test_vrt_stack_separate_bands(self, multiband_aligned_rasters, tmp_path):
        """Test VRT stacking with separate bands."""
        out_path = tmp_path / "stacked_separate.vrt"
        result = raster_stack_vrt_list(
            multiband_aligned_rasters,
            str(out_path),
            separate=True
        )
        
        ds = gdal.Open(result)
        assert ds.RasterCount == len(multiband_aligned_rasters) * 3
        ds = None

    def test_vrt_stack_merged_bands(self, multiband_aligned_rasters, tmp_path):
        """Test VRT stacking with merged bands."""
        out_path = tmp_path / "stacked_merged.vrt"
        result = raster_stack_vrt_list(
            multiband_aligned_rasters,
            str(out_path),
            separate=False
        )
        
        ds = gdal.Open(result)
        assert ds.RasterCount == 3
        ds = None

    def test_vrt_stack_with_resampling(self, aligned_rasters, tmp_path):
        """Test VRT stacking with resampling algorithm."""
        out_path = tmp_path / "stacked_resampled.vrt"
        result = raster_stack_vrt_list(
            aligned_rasters,
            str(out_path),
            resample_alg="bilinear"
        )
        
        assert Path(result).exists()

    def test_vrt_stack_with_nodata(self, aligned_rasters, tmp_path):
        """Test VRT stacking with nodata handling."""
        # Add nodata to first raster
        ds = gdal.Open(aligned_rasters[0], gdal.GA_Update)
        band = ds.GetRasterBand(1)
        band.SetNoDataValue(-9999)
        ds = None

        out_path = tmp_path / "stacked_nodata.vrt"
        result = raster_stack_vrt_list(
            aligned_rasters,
            str(out_path),
            nodata_VRT=-9999
        )
        
        ds = gdal.Open(result)
        band = ds.GetRasterBand(1)
        assert band.GetNoDataValue() == -9999
        ds = None

    def test_vrt_stack_with_reference(self, aligned_rasters, tmp_path):
        """Test VRT stacking with reference raster."""
        reference = aligned_rasters[0]
        out_path = tmp_path / "stacked_reference.vrt"
        result = raster_stack_vrt_list(
            aligned_rasters[1:],
            str(out_path),
            reference=reference
        )
        
        assert Path(result).exists()
        ref_ds = gdal.Open(reference)
        result_ds = gdal.Open(result)
        assert ref_ds.GetGeoTransform() == result_ds.GetGeoTransform()
        ref_ds = None
        result_ds = None

    def test_invalid_vrt_inputs(self, tmp_path):
        """Test invalid inputs for VRT stacking."""
        with pytest.raises(ValueError):
            raster_stack_vrt_list(
                [],
                str(tmp_path / "empty.vrt")
            )
        
        with pytest.raises(TypeError):
            raster_stack_vrt_list(
                123,
                str(tmp_path / "invalid.vrt")
            )
