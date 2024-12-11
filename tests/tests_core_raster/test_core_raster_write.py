# pylint: skip-file
# type: ignore

# Standard library
import sys; sys.path.append("../../")
import os

import pytest
import numpy as np
from osgeo import gdal, ogr, osr
from pathlib import Path

from buteo.core_raster.core_raster_write import (
    raster_create_copy,
    raster_create_empty,
    raster_create_from_array,
    raster_set_band_descriptions,
    raster_set_crs,
)



@pytest.fixture
def sample_raster(tmp_path):
    """Create a sample raster dataset."""
    raster_path = tmp_path / "sample.tif"
    driver = gdal.GetDriverByName('GTiff')
    ds = driver.Create(str(raster_path), 10, 10, 3, gdal.GDT_Float32)
    ds.SetGeoTransform([0, 1, 0, 0, 0, -1])
    for i in range(3):
        band = ds.GetRasterBand(i + 1)
        data = np.full((10, 10), i + 1, dtype=np.float32)
        band.WriteArray(data)
    ds.FlushCache()
    ds = None
    return str(raster_path)

@pytest.fixture
def sample_vector(tmp_path):
    """Create a sample vector dataset."""
    vector_path = tmp_path / "sample.gpkg"
    driver = ogr.GetDriverByName('GPKG')
    ds = driver.CreateDataSource(str(vector_path))
    layer = ds.CreateLayer('test')
    feature = ogr.Feature(layer.GetLayerDefn())
    point = ogr.Geometry(ogr.wkbPoint)
    point.AddPoint(0, 0)
    feature.SetGeometry(point)
    layer.CreateFeature(feature)
    ds = None
    return str(vector_path)


class TestRasterCreateCopy:
    def test_save_raster(self, sample_raster, tmp_path):
        """Test saving a raster dataset."""
        out_path = tmp_path / "output.tif"
        result = raster_create_copy(sample_raster, str(out_path))
        assert Path(result).exists()
        ds = gdal.Open(result)
        assert ds.RasterCount == 3
        ds = None

    def test_save_multiple_datasets(self, sample_raster, tmp_path):
        """Test saving multiple datasets."""
        out_paths = [tmp_path / "out1.tif", tmp_path / "out2.tif"]
        results = raster_create_copy(
            [sample_raster, sample_raster],
            [str(p) for p in out_paths]
        )
        assert all(Path(p).exists() for p in results)

    def test_save_with_creation_options(self, sample_raster, tmp_path):
        """Test saving with creation options."""
        out_path = tmp_path / "compressed.tif"
        result = raster_create_copy(
            sample_raster,
            str(out_path),
            creation_options=["COMPRESS=LZW"]
        )
        ds = gdal.Open(result)
        assert ds.GetMetadataItem("COMPRESSION", "IMAGE_STRUCTURE") == "LZW"
        ds = None

    def test_save_specific_bands(self, sample_raster, tmp_path):
        """Test saving specific bands from the raster."""
        out_path = tmp_path / "band_subset.tif"
        result = raster_create_copy(
            sample_raster,
            str(out_path),
            bands=[1, 3]  # Only copy first and third bands
        )
        ds = gdal.Open(result)
        assert ds.RasterCount == 2
        # Check if the band data is correct
        band1 = ds.GetRasterBand(1).ReadAsArray()
        band2 = ds.GetRasterBand(2).ReadAsArray()
        assert np.all(band1 == 1)  # First band should be all 1's
        assert np.all(band2 == 3)  # Second band should be all 3's
        ds = None

    def test_save_single_band(self, sample_raster, tmp_path):
        """Test saving a single band from the raster."""
        out_path = tmp_path / "single_band.tif"
        result = raster_create_copy(
            sample_raster,
            str(out_path),
            bands=[2]  # Only copy second band
        )
        ds = gdal.Open(result)
        assert ds.RasterCount == 1
        band = ds.GetRasterBand(1).ReadAsArray()
        assert np.all(band == 2)  # Second band should be all 2's
        ds = None

    def test_save_invalid_band(self, sample_raster, tmp_path):
        """Test saving with invalid band number."""
        out_path = tmp_path / "invalid_band.tif"
        with pytest.raises(RuntimeError):
            raster_create_copy(
                sample_raster,
                str(out_path),
                bands=[4]  # Invalid band number (sample has 3 bands)
            )

    def test_save_with_prefix_suffix(self, sample_raster, tmp_path):
        """Test saving with prefix and suffix."""
        out_path = tmp_path / "output.tif"
        result = raster_create_copy(
            sample_raster,
            str(out_path),
            prefix="test_",
            suffix="_processed"
        )
        assert Path(result).name.startswith("test_")
        assert "_processed" in Path(result).name

    def test_save_with_uuid_timestamp(self, sample_raster, tmp_path):
        """Test saving with UUID and timestamp."""
        out_path = tmp_path / "output.tif"
        result = raster_create_copy(
            sample_raster,
            str(out_path),
            add_uuid=True,
            add_timestamp=True
        )
        # Basic check that filename got longer due to UUID and timestamp
        assert len(Path(result).name) > len("output.tif")

    def test_overwrite_protection(self, sample_raster, tmp_path):
        """Test overwrite protection."""
        out_path = tmp_path / "protected.tif"
        
        # Create the file first
        raster_create_copy(sample_raster, str(out_path))
        
        # Try to overwrite with overwrite=False
        with pytest.raises(FileExistsError):
            raster_create_copy(
                sample_raster,
                str(out_path),
                overwrite=False
            )

    def test_save_invalid_input(self, sample_raster, tmp_path):
        """Test saving with invalid input."""
        out_path = tmp_path / "output.tif"
        with pytest.raises(ValueError):
            raster_create_copy(
                "nonexistent.tif",
                str(out_path)
            )

    def test_save_mixed_band_list(self, sample_raster, tmp_path):
        """Test saving with mixed order of bands."""
        out_path = tmp_path / "mixed_bands.tif"
        result = raster_create_copy(
            sample_raster,
            str(out_path),
            bands=[3, 1, 2]  # Reorder bands
        )
        with gdal.Open(result) as ds:
            assert ds.RasterCount == 3
            band1 = ds.GetRasterBand(1).ReadAsArray()
            band2 = ds.GetRasterBand(2).ReadAsArray()
            band3 = ds.GetRasterBand(3).ReadAsArray()
            assert np.all(band1 == 3)  # First band should be all 3's
            assert np.all(band2 == 1)  # Second band should be all 1's
            assert np.all(band3 == 2)  # Third band should be all 2's

        # if os.path.exists("./subset_bands"):
        #     os.remove("./subset_bands")


class TestRasterCreateEmpty:
    def test_basic_creation(self, tmp_path):
        """Test basic empty raster creation."""
        out_path = tmp_path / "empty.tif"
        result = raster_create_empty(
            str(out_path),
            width=100,
            height=100,
            bands=3
        )
        assert Path(result).exists()
        ds = gdal.Open(result)
        assert ds.RasterXSize == 100
        assert ds.RasterYSize == 100
        assert ds.RasterCount == 3
        ds = None

    def test_with_nodata(self, tmp_path):
        """Test creation with nodata value."""
        out_path = tmp_path / "with_nodata.tif"
        result = raster_create_empty(
            str(out_path),
            width=50,
            height=50,
            nodata_value=-9999,
            fill_value=0
        )
        ds = gdal.Open(result)
        band = ds.GetRasterBand(1)
        assert band.GetNoDataValue() == -9999
        data = band.ReadAsArray()
        assert np.all(data == 0)
        ds = None

    def test_with_projection(self, tmp_path):
        """Test creation with specific projection."""
        out_path = tmp_path / "projected.tif"
        result = raster_create_empty(
            str(out_path),
            projection="EPSG:4326"
        )
        ds = gdal.Open(result)
        assert "4326" in ds.GetProjection()
        ds = None

class TestRasterCreateFromArray:
    def test_2d_array(self, tmp_path):
        """Test creating raster from 2D array."""
        arr = np.ones((100, 100), dtype=np.float32)
        out_path = tmp_path / "from_2d.tif"
        result = raster_create_from_array(arr, str(out_path))
        ds = gdal.Open(result)
        assert ds.RasterCount == 1
        data = ds.ReadAsArray()
        assert np.array_equal(data, arr)
        ds = None

    def test_3d_array(self, tmp_path):
        """Test creating raster from 3D array."""
        arr = np.ones((100, 100, 3), dtype=np.float32)
        out_path = tmp_path / "from_3d.tif"
        result = raster_create_from_array(arr, str(out_path))
        ds = gdal.Open(result)
        assert ds.RasterCount == 3
        data = ds.ReadAsArray()
        assert np.array_equal(data, arr.transpose(2, 0, 1))
        ds = None

    def test_with_masked_array(self, tmp_path):
        """Test creating raster from masked array."""
        arr = np.ma.array(
            np.ones((50, 50)),
            mask=np.zeros((50, 50), dtype=bool)
        )
        arr.mask[0, 0] = True
        out_path = tmp_path / "from_masked.tif"
        result = raster_create_from_array(arr, str(out_path))
        
        ds = gdal.Open(result)
        band = ds.GetRasterBand(1)
        assert band.GetNoDataValue() is not None
        data = band.ReadAsArray()
        assert data[0, 0] == arr.fill_value
        ds = None

    def test_with_projection(self, tmp_path):
        """Test creating raster with specific projection."""
        arr = np.ones((30, 30), dtype=np.float32)
        out_path = tmp_path / "projected.tif"
        result = raster_create_from_array(
            arr,
            str(out_path),
            projection="EPSG:4326"
        )
        
        ds = gdal.Open(result)
        assert "4326" in ds.GetProjection()
        ds = None

    def test_with_pixel_size(self, tmp_path):
        """Test creating raster with specific pixel size."""
        arr = np.ones((20, 20), dtype=np.float32)
        out_path = tmp_path / "pixel_size.tif"
        pixel_size = 0.5
        result = raster_create_from_array(
            arr,
            str(out_path),
            pixel_size=pixel_size
        )
        
        ds = gdal.Open(result)
        gt = ds.GetGeoTransform()
        assert gt[1] == pixel_size  # pixel width
        assert abs(gt[5]) == pixel_size  # pixel height
        ds = None

    def test_with_origin(self, tmp_path):
        """Test creating raster with specific origin coordinates."""
        arr = np.ones((25, 25), dtype=np.float32)
        out_path = tmp_path / "origin.tif"
        x_min, y_max = 100.0, 500.0
        result = raster_create_from_array(
            arr,
            str(out_path),
            x_min=x_min,
            y_max=y_max
        )
        
        ds = gdal.Open(result)
        gt = ds.GetGeoTransform()
        assert gt[0] == x_min
        assert gt[3] == y_max
        ds = None

class TestRasterSetBandDescriptions:
    def test_set_single_band_description(self, sample_raster):
        """Test setting description for a single band."""
        result = raster_set_band_descriptions(
            sample_raster,
            bands=[1],
            descriptions=["Red"]
        )
        
        ds = gdal.Open(result)
        band = ds.GetRasterBand(1)
        assert band.GetDescription() == "Red"
        ds = None

    def test_set_multiple_band_descriptions(self, sample_raster):
        """Test setting descriptions for multiple bands."""
        result = raster_set_band_descriptions(
            sample_raster,
            bands=[1, 2, 3],
            descriptions=["Red", "Green", "Blue"]
        )
        
        ds = gdal.Open(result)
        for i, desc in enumerate(["Red", "Green", "Blue"], 1):
            band = ds.GetRasterBand(i)
            assert band.GetDescription() == desc
        ds = None

    def test_mismatched_bands_descriptions(self, sample_raster):
        """Test error when bands and descriptions lists have different lengths."""
        with pytest.raises(AssertionError):
            raster_set_band_descriptions(
                sample_raster,
                bands=[1, 2],
                descriptions=["Red"]
            )

    def test_invalid_band_number(self, sample_raster):
        """Test error when band number is invalid."""
        with pytest.raises(RuntimeError):
            raster_set_band_descriptions(
                sample_raster,
                bands=[4],  # Assuming sample_raster has 3 bands
                descriptions=["Invalid"]
            )

    def test_invalid_inputs(self, sample_raster):
        """Test various invalid inputs."""
        with pytest.raises(AssertionError):
            raster_set_band_descriptions(
                sample_raster,
                bands=["1"],  # Should be integer
                descriptions=["Red"]
            )
        
        with pytest.raises(AssertionError):
            raster_set_band_descriptions(
                sample_raster,
                bands=[1],
                descriptions=[1]  # Should be string
            )

class TestRasterSetCRS:
    def test_set_crs_with_epsg_code(self, sample_raster):
        """Test setting CRS using an EPSG code."""
        result = raster_set_crs(sample_raster, 4326)
        ds = gdal.Open(result)
        spatial_ref = osr.SpatialReference(wkt=ds.GetProjection())
        assert spatial_ref.GetAuthorityCode(None) == '4326'
        ds = None

    def test_set_crs_with_epsg_string(self, sample_raster):
        """Test setting CRS using an EPSG string."""
        result = raster_set_crs(sample_raster, "EPSG:3857")
        ds = gdal.Open(result)
        spatial_ref = osr.SpatialReference(wkt=ds.GetProjection())
        assert spatial_ref.GetAuthorityCode(None) == '3857'
        ds = None

    def test_set_crs_with_spatial_reference(self, sample_raster):
        """Test setting CRS using an osr.SpatialReference object."""
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(32633)  # WGS 84 / UTM zone 33N
        result = raster_set_crs(sample_raster, srs)
        ds = gdal.Open(result)
        spatial_ref = osr.SpatialReference(wkt=ds.GetProjection())
        assert spatial_ref.GetAuthorityCode(None) == '32633'
        ds = None

    def test_set_crs_with_gdal_dataset(self, sample_raster, tmp_path):
        """Test setting CRS using another GDAL Dataset."""
        # Create a reference raster with known CRS
        ref_path = tmp_path / "reference.tif"
        raster_create_empty(
            str(ref_path),
            projection="EPSG:26915"  # NAD83 / UTM zone 15N
        )
        result = raster_set_crs(sample_raster, str(ref_path))
        ds = gdal.Open(result)
        spatial_ref = osr.SpatialReference(wkt=ds.GetProjection())
        assert spatial_ref.GetAuthorityCode(None) == '26915'
        ds = None

    def test_set_crs_invalid_input(self, sample_raster):
        """Test setting CRS with invalid input."""
        with pytest.raises(ValueError):
            raster_set_crs(sample_raster, "INVALID_CRS")

    def test_set_crs_in_place(self, sample_raster):
        """Test that raster_set_crs modifies the raster in place."""
        original_ds = gdal.Open(sample_raster)
        original_proj = original_ds.GetProjection()
        original_ds = None

        raster_set_crs(sample_raster, 4326)

        ds = gdal.Open(sample_raster)
        new_proj = ds.GetProjection()
        ds = None

        assert original_proj != new_proj
