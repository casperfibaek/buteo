# pylint: skip-file
# type: ignore

# Standard library
import sys; sys.path.append("../../")

import pytest
import numpy as np
from osgeo import gdal
from pathlib import Path

from buteo.core_raster.core_raster_datatypes import (
    raster_get_datatype,
    raster_set_datatype
)




@pytest.fixture
def sample_raster_float32(tmp_path):
    """Creates a sample Float32 raster dataset for testing."""
    raster_path = tmp_path / "sample_float32.tif"
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(str(raster_path), 20, 10, 1, gdal.GDT_Float32)
    data = np.random.rand(10, 20).astype(np.float32)
    ds.GetRasterBand(1).WriteArray(data)
    ds.FlushCache()
    ds = None
    return str(raster_path)

@pytest.fixture
def sample_raster_uint8(tmp_path):
    """Creates a sample UInt8 raster dataset for testing."""
    raster_path = tmp_path / "sample_uint8.tif"
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(str(raster_path), 20, 10, 1, gdal.GDT_Byte)
    data = (np.random.rand(10, 20) * 255).astype(np.uint8)
    ds.GetRasterBand(1).WriteArray(data)
    ds.FlushCache()
    ds = None
    return str(raster_path)

class TestRasterGetDatatype:
    def test_get_datatype_single(self, sample_raster_float32):
        """Test raster_get_datatype with a single raster."""
        dtype = raster_get_datatype(sample_raster_float32)
        assert dtype == 'float32'

    def test_get_datatype_list(self, sample_raster_float32, sample_raster_uint8):
        """Test raster_get_datatype with a list of rasters."""
        rasters = [sample_raster_float32, sample_raster_uint8]
        dtypes = raster_get_datatype(rasters)
        assert dtypes == ['float32', 'uint8']

    def test_invalid_input(self):
        """Test raster_get_datatype with invalid input."""
        with pytest.raises(ValueError):
            raster_get_datatype("non_existent.tif")

class TestRasterSetDatatype:
    def test_set_datatype_float32_to_uint16(self, sample_raster_float32, tmp_path):
        """Test changing data type from Float32 to UInt16."""
        output_path = tmp_path / "converted_uint16.tif"
        result = raster_set_datatype(
            sample_raster_float32,
            'uint16',
            out_path=str(output_path)
        )
        assert Path(result).exists()
        ds = gdal.Open(result)
        band = ds.GetRasterBand(1)
        dtype = gdal.GetDataTypeName(band.DataType)
        assert dtype == 'UInt16'
        ds = None

    def test_set_datatype_uint8_to_float64(self, sample_raster_uint8, tmp_path):
        """Test changing data type from UInt8 to Float64."""
        output_path = tmp_path / "converted_float64.tif"
        result = raster_set_datatype(
            sample_raster_uint8,
            'float64',
            out_path=str(output_path)
        )
        assert Path(result).exists()
        ds = gdal.Open(result)
        band = ds.GetRasterBand(1)
        dtype = gdal.GetDataTypeName(band.DataType)
        assert dtype == 'Float64'
        ds = None

    def test_set_datatype_with_creation_options(self, sample_raster_float32, tmp_path):
        """Test setting data type with GDAL creation options."""
        output_path = tmp_path / "converted_with_options.tif"
        result = raster_set_datatype(
            sample_raster_float32,
            'uint8',
            out_path=str(output_path),
            creation_options=["COMPRESS=LZW"]
        )
        assert Path(result).exists()
        ds = gdal.Open(result)
        compression = ds.GetMetadataItem('COMPRESSION', 'IMAGE_STRUCTURE')
        assert compression == 'LZW'
        ds = None

    def test_set_datatype_overwrite(self, sample_raster_uint8, tmp_path):
        """Test overwriting an existing output file."""
        output_path = tmp_path / "overwrite_test.tif"
        raster_set_datatype(
            sample_raster_uint8,
            'float32',
            out_path=str(output_path)
        )

        # Attempt to overwrite without setting overwrite=True
        with pytest.raises(Exception):
            raster_set_datatype(
                sample_raster_uint8,
                'float32',
                out_path=str(output_path),
                overwrite=False
            )

        # Overwrite with overwrite=True
        result = raster_set_datatype(
            sample_raster_uint8,
            'float32',
            out_path=str(output_path),
            overwrite=True
        )
        assert Path(result).exists()
        ds = gdal.Open(result)
        dtype = gdal.GetDataTypeName(ds.GetRasterBand(1).DataType)
        assert dtype == 'Float32'
        ds = None

    def test_set_datatype_invalid_dtype(self, sample_raster_float32):
        """Test setting data type with an invalid dtype."""
        with pytest.raises(TypeError):
            raster_set_datatype(
                sample_raster_float32,
                12345  # Invalid dtype
            )

    def test_set_datatype_list(self, sample_raster_float32, sample_raster_uint8, tmp_path):
        """Test setting data type for a list of rasters."""
        rasters = [sample_raster_float32, sample_raster_uint8]
        output_paths = [
            tmp_path / "converted1.tif",
            tmp_path / "converted2.tif"
        ]
        results = raster_set_datatype(
            rasters,
            'int16',
            out_path=[str(output_paths[0]), str(output_paths[1])]
        )
        for result_path in results:
            assert Path(result_path).exists()
            ds = gdal.Open(result_path)
            dtype = gdal.GetDataTypeName(ds.GetRasterBand(1).DataType)
            assert dtype == 'Int16'
            ds = None

    def test_set_datatype_add_suffix(self, sample_raster_float32, tmp_path):
        """Test adding a suffix to the output file name."""
        result = raster_set_datatype(
            sample_raster_float32,
            'uint8',
            out_path=None,
            suffix="_uint8",
            overwrite=True
        )
        assert result.endswith("_uint8.tif")
        assert result.startswith('/vsimem/')
        ds = gdal.Open(result)
        assert ds is not None
        dtype = gdal.GetDataTypeName(ds.GetRasterBand(1).DataType)
        assert dtype == 'Byte'
        ds = None
        # Clean up the generated file
        assert gdal.Unlink(result) == 0

    def test_set_datatype_no_overwrite_existing_file(self, sample_raster_uint8, tmp_path):
        """Test attempting to overwrite an existing file without overwrite flag."""
        output_path = tmp_path / "existing_file.tif"
        # Create an initial file
        raster_set_datatype(
            sample_raster_uint8,
            'float32',
            out_path=str(output_path)
        )
        # Attempt to overwrite
        with pytest.raises(Exception):
            raster_set_datatype(
                sample_raster_uint8,
                'float32',
                out_path=str(output_path),
                overwrite=False
            )

    def test_invalid_raster_input(self):
        """Test raster_set_datatype with invalid raster input."""
        with pytest.raises(ValueError):
            raster_set_datatype(
                "non_existent_file.tif",
                'uint8'
            )

    def test_raster_set_datatype_no_output_path(self, sample_raster_uint8):
        """Test raster_set_datatype without specifying out_path."""
        result = raster_set_datatype(
            sample_raster_uint8,
            'float32',
            out_path=None
        )
        assert result.startswith('/vsimem/')
        ds = gdal.Open(result)

        assert ds is not None
        
        dtype = gdal.GetDataTypeName(ds.GetRasterBand(1).DataType)
        assert dtype == 'Float32'
        ds = None
        # Clean up the generated file
        assert gdal.Unlink(result) == 0
