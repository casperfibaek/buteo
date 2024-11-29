# pylint: skip-file
# type: ignore

# Standard library
import sys; sys.path.append("../../")

import pytest
import numpy as np
from osgeo import gdal
from pathlib import Path

from buteo.core_raster.core_raster_subset import raster_extract_bands



@pytest.fixture
def sample_multiband_raster(tmp_path):
    """Creates a sample multiband raster dataset for testing."""
    raster_path = tmp_path / "multiband.tif"
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(str(raster_path), 10, 10, 3, gdal.GDT_Float32)
    ds.SetGeoTransform((0, 1, 0, 0, 0, -1))
    
    # Create distinct data for each band
    for i in range(1, 4):
        band = ds.GetRasterBand(i)
        data = np.full((10, 10), i, dtype=np.float32)
        band.WriteArray(data)
    
    ds.FlushCache()
    ds = None
    return str(raster_path)

class TestRasterExtractBands:
    def test_extract_single_band(self, sample_multiband_raster, tmp_path):
        """Test extracting a single band from a multiband raster."""
        out_path = tmp_path / "single_band.tif"
        result = raster_extract_bands(
            sample_multiband_raster,
            band=1,
            out_path=str(out_path)
        )
        
        ds = gdal.Open(result)
        assert ds is not None
        assert ds.RasterCount == 1
        data = ds.GetRasterBand(1).ReadAsArray()
        assert np.all(data == 1)  # First band should all be 1s
        ds = None

    def test_extract_multiple_bands(self, sample_multiband_raster, tmp_path):
        """Test extracting multiple bands from a multiband raster."""
        out_path = tmp_path / "multi_band.tif"
        result = raster_extract_bands(
            sample_multiband_raster,
            band=[1, 3],
            out_path=str(out_path)
        )
        
        assert Path(result).exists()
        ds = gdal.Open(result)
        assert ds.RasterCount == 2
        band1_data = ds.GetRasterBand(1).ReadAsArray()
        band2_data = ds.GetRasterBand(2).ReadAsArray()
        assert np.all(band1_data == 1)  # First band should be 1s
        assert np.all(band2_data == 3)  # Third band should be 3s
        ds = None

    def test_no_output_path(self, sample_multiband_raster):
        """Test extraction without specifying output path."""
        result = raster_extract_bands(
            sample_multiband_raster,
            band=1
        )
        ds = gdal.Open(result)
        assert ds is not None
        assert ds.RasterCount == 1
        # Clean up temporary file
        gdal.Unlink(result)
        ds = None

    def test_creation_options(self, sample_multiband_raster, tmp_path):
        """Test band extraction with specific creation options."""
        out_path = tmp_path / "compressed.tif"
        result = raster_extract_bands(
            sample_multiband_raster,
            band=1,
            out_path=str(out_path),
            creation_options=["COMPRESS=LZW"]
        )
        
        ds = gdal.Open(result)
        assert ds.GetMetadataItem("COMPRESSION", "IMAGE_STRUCTURE") == "LZW"
        ds = None

    def test_invalid_band_number(self, sample_multiband_raster, tmp_path):
        """Test extraction with invalid band number."""
        out_path = tmp_path / "invalid.tif"
        with pytest.raises(ValueError):
            raster_extract_bands(
                sample_multiband_raster,
                band=4,  # Only 3 bands exist
                out_path=str(out_path)
            )

    def test_invalid_band_type(self, sample_multiband_raster):
        """Test extraction with invalid band type."""
        with pytest.raises(AssertionError):
            raster_extract_bands(
                sample_multiband_raster,
                band="1"  # Should be int or list of ints
            )

    def test_overwrite_existing(self, sample_multiband_raster, tmp_path):
        """Test overwriting existing output file."""
        out_path = tmp_path / "overwrite.tif"
        
        # Create first version
        raster_extract_bands(
            sample_multiband_raster,
            band=1,
            out_path=str(out_path)
        )
        
        # Overwrite with different band
        result = raster_extract_bands(
            sample_multiband_raster,
            band=2,
            out_path=str(out_path),
            overwrite=True
        )
        
        ds = gdal.Open(result)
        data = ds.GetRasterBand(1).ReadAsArray()
        assert np.all(data == 2)  # Should now contain band 2 data
        ds = None

    def test_no_overwrite_existing(self, sample_multiband_raster, tmp_path):
        """Test failing when not allowing overwrite of existing file."""
        out_path = tmp_path / "no_overwrite.tif"
        
        # Create first version
        raster_extract_bands(
            sample_multiband_raster,
            band=1,
            out_path=str(out_path)
        )
        
        # Attempt to overwrite without permission
        with pytest.raises(AssertionError):
            raster_extract_bands(
                sample_multiband_raster,
                band=2,
                out_path=str(out_path),
                overwrite=False
            )

    def test_invalid_raster_input(self, tmp_path):
        """Test extraction with invalid raster input."""
        out_path = tmp_path / "invalid.tif"
        with pytest.raises(AssertionError):
            raster_extract_bands(
                "nonexistent.tif",
                band=1,
                out_path=str(out_path)
            )