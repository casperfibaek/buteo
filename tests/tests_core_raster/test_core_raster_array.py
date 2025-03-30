# pylint: skip-file
# type: ignore

import pytest
import numpy as np
from osgeo import gdal
from pathlib import Path

from buteo.core_raster.core_raster_array import (
    raster_to_array,
    raster_to_array_random_patches,
    array_to_raster
)

@pytest.fixture
def sample_raster_dataset(tmp_path):
    """Creates a sample raster dataset with random values for testing."""
    raster_path = tmp_path / "sample_raster_random.tif"
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(str(raster_path), 20, 10, 3, gdal.GDT_Float32)
    ds.SetProjection('GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]')
    ds.SetGeoTransform((0, 1, 0, 0, 0, -1))
    for i in range(1, 4):
        band = ds.GetRasterBand(i)
        data = np.random.rand(10, 20).astype(np.float32)
        band.WriteArray(data)
    ds.FlushCache()
    ds = None
    return str(raster_path)

class TestRasterToArray:
    def test_basic_read(self, sample_raster_dataset):
        """Test basic reading of raster to array."""
        arr = raster_to_array(sample_raster_dataset)
        assert arr.shape == (3, 10, 20)
        assert arr.dtype == np.float32

    def test_band_selection(self, sample_raster_dataset):
        """Test band selection."""
        # Single band
        arr = raster_to_array(sample_raster_dataset, bands=1)
        assert arr.shape == (1, 10, 20)
        # Multiple bands
        arr = raster_to_array(sample_raster_dataset, bands=[1, 3])
        assert arr.shape == (2, 10, 20)

    def test_pixel_offsets(self, sample_raster_dataset):
        """Test reading with pixel offsets."""
        arr = raster_to_array(
            sample_raster_dataset,
            pixel_offsets=(2, 2, 5, 5)
        )
        assert arr.shape == (3, 5, 5)

    def test_bbox_reading(self, sample_raster_dataset):
        """Test reading with bbox."""
        ds = gdal.Open(sample_raster_dataset)
        gt = ds.GetGeoTransform()
        xmin = gt[0]
        ymax = gt[3]
        bbox = [
            xmin, xmin + 1,
            ymax, ymax + 1,
        ]
        arr = raster_to_array(sample_raster_dataset, bbox=bbox)
        assert arr.shape == (3, 1, 1)
        ds = None

    def test_filled_nodata(self, sample_raster_dataset):
        """Test nodata filling."""
        # Set nodata value and create nodata pixel
        ds = gdal.Open(sample_raster_dataset, gdal.GA_Update)
        band = ds.GetRasterBand(1)
        band.SetNoDataValue(-9999)
        data = band.ReadAsArray()
        data[0, 0] = -9999
        band.WriteArray(data)
        ds.FlushCache()
        ds = None
        # Test filling nodata
        arr = raster_to_array(
            sample_raster_dataset,
            filled=True,
            fill_value=0
        )
        assert arr[0, 0, 0] == 0

    def test_invalid_inputs(self, sample_raster_dataset):
        """Test invalid inputs."""
        with pytest.raises(TypeError):
            raster_to_array(123)
        with pytest.raises(ValueError):
            raster_to_array(sample_raster_dataset, bands=4)

    def test_cast_dtype(self, sample_raster_dataset):
        """Test casting to different dtype."""
        arr = raster_to_array(sample_raster_dataset, cast=np.int32)
        assert arr.dtype == np.int32

    def test_bbox_with_srs(self, sample_raster_dataset):
        """Test reading with bbox and spatial reference system."""
        ds = gdal.Open(sample_raster_dataset)
        gt = ds.GetGeoTransform()
        xmin = gt[0]
        ymax = gt[3]
        bbox = [
            xmin, xmin + 1.00001,
            ymax, ymax + 1.00001,
        ]
        arr = raster_to_array(sample_raster_dataset, bbox=bbox)
        assert arr.shape == (3, 1, 1)
        ds = None

class TestArrayToRaster:
    def test_basic_conversion(self, sample_raster_dataset, tmp_path):
        """Test basic array to raster conversion."""
        arr = np.ones((3, 10, 20), dtype=np.float32)
        out_path = tmp_path / "output.tif"
        result = array_to_raster(arr, reference=sample_raster_dataset, out_path=str(out_path))
        assert Path(result).exists()
        ds = gdal.Open(str(out_path))
        assert ds.RasterXSize == 20
        assert ds.RasterYSize == 10
        assert ds.RasterCount == 3
        data = ds.ReadAsArray()
        assert np.array_equal(data, arr)
        ds = None

    def test_2d_array_conversion(self, sample_raster_dataset, tmp_path):
        """Test converting a 2D array to raster."""
        arr = np.full((10, 20), 42, dtype=np.float32)
        output_path = tmp_path / "output_2d.tif"
        array_to_raster(arr, reference=sample_raster_dataset, out_path=str(output_path))
        assert output_path.exists()
        ds = gdal.Open(str(output_path))
        assert ds.RasterCount == 1
        data = ds.ReadAsArray()
        assert data.shape == (10, 20)
        assert np.all(data == 42)
        ds = None

    def test_set_nodata(self, sample_raster_dataset, tmp_path):
        """Test setting nodata value."""
        arr = np.ma.array(
            np.ones((3, 10, 20), dtype=np.float32),
            mask=False,
            fill_value=-9999
        )
        arr.mask[0, 0, 0] = True
        out_path = tmp_path / "output_nodata.tif"
        result = array_to_raster(
            arr,
            reference=sample_raster_dataset,
            out_path=str(out_path),
            set_nodata="arr"
        )
        ds = gdal.Open(str(out_path))
        band = ds.GetRasterBand(1)
        nodata = band.GetNoDataValue()

        assert nodata == -9999
        data = band.ReadAsArray()
        assert data[0, 0] == -9999
        ds = None

    def test_invalid_inputs(self, sample_raster_dataset, tmp_path):
        """Test invalid inputs."""
        arr = np.ones((3, 10, 20), dtype=np.float32)
        with pytest.raises(ValueError):
            array_to_raster(
                arr,
                reference=sample_raster_dataset,
                out_path=str(tmp_path / "output_invalid.tif"),
                set_nodata="invalid_value"
            )
        with pytest.raises(ValueError):
            array_to_raster(
                arr,
                reference="non_existent.tif",
                out_path=str(tmp_path / "output_invalid.tif")
            )
        with pytest.raises(ValueError):
            array_to_raster(
                arr,
                reference=sample_raster_dataset,
                out_path=str(tmp_path / "output_invalid.tif"),
                pixel_offsets=(0, 0, 5)
            )

    def test_creation_options(self, sample_raster_dataset, tmp_path):
        """Test raster creation with GDAL creation options."""
        arr = np.ones((3, 10, 20), dtype=np.float32)
        out_path = tmp_path / "output_creation_options.tif"
        result = array_to_raster(
            arr,
            reference=sample_raster_dataset,
            out_path=str(out_path),
            creation_options=["COMPRESS=LZW"]
        )
        assert Path(result).exists()
        ds = gdal.Open(str(out_path))
        assert ds.GetMetadataItem("COMPRESSION", "IMAGE_STRUCTURE") == "LZW"
        ds = None

    def test_overwrite_existing_file(self, sample_raster_dataset, tmp_path):
        """Test overwriting an existing file."""
        arr = np.ones((3, 10, 20), dtype=np.float32)
        out_path = tmp_path / "output_overwrite.tif"
        array_to_raster(arr, reference=sample_raster_dataset, out_path=str(out_path))
        # Overwrite with new data
        new_arr = np.full((3, 10, 20), 42, dtype=np.float32)
        result = array_to_raster(new_arr, reference=sample_raster_dataset, out_path=str(out_path), overwrite=True)
        assert Path(result).exists()
        ds = gdal.Open(str(out_path))
        data = ds.ReadAsArray()
        assert np.array_equal(data, new_arr)
        ds = None


class TestRasterToArrayRandomPatches:
    def test_random_patches_basic(self, sample_raster_dataset):
        """Test basic functionality of raster_to_array_random_patches."""
        patch_size = (5, 5)
        num_patches = 2
        arr = raster_to_array_random_patches(
            sample_raster_dataset,
            patch_size=patch_size,
            num_patches=num_patches
        )
        assert arr.shape == (num_patches, 3, patch_size[0], patch_size[1])
        assert arr.dtype == np.float32

    def test_random_patches_filled_nodata(self, sample_raster_dataset):
        """Test raster_to_array_random_patches with filled nodata values."""
        # Set nodata value and create nodata pixels
        ds = gdal.Open(sample_raster_dataset, gdal.GA_Update)
        band = ds.GetRasterBand(1)
        band.SetNoDataValue(-9999)
        data = band.ReadAsArray()
        # Set every second pixel to nodata:
        data[::2, ::2] = -9999
        band.WriteArray(data)
        ds.FlushCache()
        ds = None
        # Test filled nodata
        patch_size = (5, 5)
        num_patches = 3
        arr = raster_to_array_random_patches(
            sample_raster_dataset,
            patch_size=patch_size,
            num_patches=num_patches,
            filled=True,
            fill_value=0
        )
        assert arr.shape == (num_patches, 3, patch_size[0], patch_size[1])
        assert arr.dtype == np.float32

        # Since its every second pixel is nodata, it should be safe.
        assert arr[0, :, :, :].min() == 0
        assert arr[1, :, :, :].min() == 0
        assert arr[2, :, :, :].min() == 0

    def test_random_patches_invalid_inputs(self, sample_raster_dataset):
        """Test invalid inputs for raster_to_array_random_patches."""
        with pytest.raises(TypeError):
            raster_to_array_random_patches(123, patch_size=(5, 5), num_patches=2)
        with pytest.raises(ValueError):
            raster_to_array_random_patches(sample_raster_dataset, patch_size=(0, 5), num_patches=2)
        with pytest.raises(ValueError):
            raster_to_array_random_patches(sample_raster_dataset, patch_size=(5, 5), num_patches=-1)
        with pytest.raises(ValueError):
            raster_to_array_random_patches(sample_raster_dataset, patch_size=(5,), num_patches=2)

    def test_random_patches_with_specific_bands(self, sample_raster_dataset):
        """Test raster_to_array_random_patches with specific band selection."""
        patch_size = (4, 4)
        num_patches = 4
        arr = raster_to_array_random_patches(
            sample_raster_dataset,
            patch_size=patch_size,
            num_patches=num_patches,
            bands=[1, 3]
        )
        assert arr.shape == (num_patches, 2, patch_size[0], patch_size[1])
        assert arr.dtype == np.float32

    def test_random_patches_cast_dtype(self, sample_raster_dataset):
        """Test casting dtype in raster_to_array_random_patches."""
        patch_size = (3, 3)
        num_patches = 1
        arr = raster_to_array_random_patches(
            sample_raster_dataset,
            patch_size=patch_size,
            num_patches=num_patches,
            cast=np.int32
        )
        assert arr.shape == (num_patches, 3, patch_size[0], patch_size[1])
        assert arr.dtype == np.int32
