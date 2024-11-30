# pylint: skip-file
# type: ignore

# Standard library
import sys; sys.path.append("../../")

import pytest
from osgeo import gdal, osr, ogr
from pathlib import Path

from buteo.core_raster.core_raster_extent import (
    get_raster_overlap_fraction,
    get_raster_intersection,
    check_rasters_intersect,
    check_rasters_are_aligned,
    raster_to_vector_extent,
    _raster_to_vector_extent,
)

@pytest.fixture
def reference_raster(tmp_path):
    """Create a reference raster at (0,0) with size 10x10."""
    raster_path = tmp_path / "reference.tif"
    driver = gdal.GetDriverByName('GTiff')
    ds = driver.Create(str(raster_path), 10, 10, 1, gdal.GDT_Byte)
    ds.SetGeoTransform([0, 1, 0, 0, 0, -1])
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    ds.SetProjection(srs.ExportToWkt())
    ds.GetRasterBand(1).Fill(1)
    ds = None
    return str(raster_path)

@pytest.fixture
def overlapping_raster(tmp_path):
    """Create a raster overlapping with reference at (5,-5) with size 10x10."""
    raster_path = tmp_path / "overlapping.tif"
    driver = gdal.GetDriverByName('GTiff')
    ds = driver.Create(str(raster_path), 10, 10, 1, gdal.GDT_Byte)
    ds.SetGeoTransform([5, 1, 0, 5, 0, -1])
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    ds.SetProjection(srs.ExportToWkt())
    ds.GetRasterBand(1).Fill(2)
    ds = None
    return str(raster_path)

@pytest.fixture
def non_overlapping_raster(tmp_path):
    """Create a raster not overlapping with reference at (20,20)."""
    raster_path = tmp_path / "non_overlapping.tif"
    driver = gdal.GetDriverByName('GTiff')
    ds = driver.Create(str(raster_path), 10, 10, 1, gdal.GDT_Byte)
    ds.SetGeoTransform([20, 1, 0, 20, 0, -1])
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    ds.SetProjection(srs.ExportToWkt())
    ds.GetRasterBand(1).Fill(3)
    ds = None
    return str(raster_path)

class TestGetRasterOverlapFraction:
    def test_full_overlap(self, reference_raster):
        """Test overlap fraction of a raster with itself."""
        fraction = get_raster_overlap_fraction(reference_raster, reference_raster)
        assert fraction == 1.0

    def test_partial_overlap(self, reference_raster, overlapping_raster):
        """Test partial overlap between two rasters."""
        fraction = get_raster_overlap_fraction(reference_raster, overlapping_raster)
        assert 0.0 < fraction < 1.0

    def test_no_overlap(self, reference_raster, non_overlapping_raster):
        """Test overlap fraction when rasters don't overlap."""
        fraction = get_raster_overlap_fraction(reference_raster, non_overlapping_raster)
        assert fraction == 0.0

    def test_invalid_input(self, reference_raster):
        """Test with invalid input."""
        with pytest.raises(TypeError):
            get_raster_overlap_fraction(123, reference_raster)
        with pytest.raises(ValueError):
            get_raster_overlap_fraction(reference_raster, "nonexistent.tif")

class TestRasterToExtent:
    def test_basic_extent(self, reference_raster, tmp_path):
        """Test basic extent extraction."""
        out_path = tmp_path / "extent.gpkg"
        result = _raster_to_vector_extent(reference_raster, str(out_path))
        assert Path(result).exists()

        # Verify extent geometry
        ds = gdal.OpenEx(result, gdal.OF_VECTOR)
        layer = ds.GetLayer(0)
        feature = layer.GetNextFeature()
        geom = feature.GetGeometryRef()
        env = geom.GetEnvelope()
        
        assert env[0] == 0  # xmin
        assert env[1] == 10  # xmax
        assert env[2] == -10  # ymin
        assert env[3] == 0  # ymax
        ds = None

    def test_latlng_conversion(self, reference_raster, tmp_path):
        """Test extent extraction with lat/lng conversion."""
        out_path = tmp_path / "extent_latlng.gpkg"
        result = _raster_to_vector_extent(reference_raster, str(out_path), latlng=True)
        assert Path(result).exists()

        ds = gdal.OpenEx(result, gdal.OF_VECTOR)
        layer = ds.GetLayer(0)
        srs = layer.GetSpatialRef()
        assert srs.GetAuthorityCode(None) == "4326"
        ds = None

    def test_invalid_inputs(self, reference_raster):
        """Test with invalid inputs."""
        with pytest.raises(TypeError):
            _raster_to_vector_extent(123)
        with pytest.raises(ValueError):
            _raster_to_vector_extent(reference_raster, "invalid/path/extent.gpkg")

class TestGetRasterIntersection:
    def test_self_intersection(self, reference_raster):
        """Test intersection of a raster with itself."""
        intersection = get_raster_intersection(reference_raster, reference_raster)
        env = intersection.GetEnvelope()
        assert env[0] == 0  # xmin
        assert env[1] == 10  # xmax
        assert env[2] == -10  # ymin
        assert env[3] == 0  # ymax

    def test_partial_intersection(self, reference_raster, overlapping_raster):
        """Test partial intersection between two rasters."""
        intersection = get_raster_intersection(reference_raster, overlapping_raster)
        env = intersection.GetEnvelope()
        assert env[0] == 5  # xmin
        assert env[1] == 10  # xmax
        assert env[2] == -5  # ymin
        assert env[3] == 0  # ymax

    def test_no_intersection(self, reference_raster, non_overlapping_raster):
        """Test when rasters don't intersect."""
        with pytest.raises(ValueError):
            get_raster_intersection(reference_raster, non_overlapping_raster)

class TestCheckRastersIntersect:
    def test_self_intersection(self, reference_raster):
        """Test if raster intersects with itself."""
        assert check_rasters_intersect(reference_raster, reference_raster)

    def test_partial_intersection(self, reference_raster, overlapping_raster):
        """Test if partially overlapping rasters intersect."""
        assert check_rasters_intersect(reference_raster, overlapping_raster)

    def test_no_intersection(self, reference_raster, non_overlapping_raster):
        """Test if non-overlapping rasters don't intersect."""
        assert not check_rasters_intersect(reference_raster, non_overlapping_raster)

class TestCheckRastersAreAligned:
    def test_self_alignment(self, reference_raster):
        """Test if raster is aligned with itself."""
        assert check_rasters_are_aligned([reference_raster])

    def test_unaligned_rasters(self, reference_raster, overlapping_raster):
        """Test if unaligned rasters are detected."""
        assert not check_rasters_are_aligned([reference_raster, overlapping_raster])

    def test_same_dtype_check(self, reference_raster, tmp_path):
        """Test alignment check with dtype verification."""
        # Create raster with different dtype
        diff_dtype_path = tmp_path / "diff_dtype.tif"
        driver = gdal.GetDriverByName('GTiff')
        ds = driver.Create(str(diff_dtype_path), 10, 10, 1, gdal.GDT_Float32)
        ds.SetGeoTransform([0, 1, 0, 0, 0, -1])
        ds.SetProjection(gdal.Open(reference_raster).GetProjection())
        ds = None

        assert not check_rasters_are_aligned(
            [reference_raster, str(diff_dtype_path)],
            same_dtype=True
        )

    def test_same_nodata_check(self, reference_raster, tmp_path):
        """Test alignment check with nodata verification."""
        # Create raster with different nodata
        diff_nodata_path = tmp_path / "diff_nodata.tif"
        driver = gdal.GetDriverByName('GTiff')
        ds = driver.Create(str(diff_nodata_path), 10, 10, 1, gdal.GDT_Byte)
        ds.SetGeoTransform([0, 1, 0, 0, 0, -1])
        ds.SetProjection(gdal.Open(reference_raster).GetProjection())
        band = ds.GetRasterBand(1)
        band.SetNoDataValue(255)
        ds = None

        assert not check_rasters_are_aligned(
            [reference_raster, str(diff_nodata_path)],
            same_nodata=True
        )

    def test_invalid_inputs(self, reference_raster):
        """Test with invalid inputs."""
        with pytest.raises(TypeError):
            check_rasters_are_aligned([123])
        with pytest.raises(ValueError):
            check_rasters_are_aligned([])

class TestRasterToVectorExtent:
    def test_basic_vector_extent(self, reference_raster, tmp_path):
        """Test basic vector extent creation."""
        out_path = tmp_path / "vector_extent.gpkg"
        result = _raster_to_vector_extent(reference_raster, str(out_path))
        
        assert Path(result).exists()
        ds = ogr.Open(str(out_path))
        layer = ds.GetLayer(0)
        assert layer.GetFeatureCount() == 1
        
        # Check geometry
        feature = layer.GetNextFeature()
        geom = feature.GetGeometryRef()
        env = geom.GetEnvelope()
        assert env[0] == 0  # xmin
        assert env[1] == 10  # xmax
        assert env[2] == -10  # ymin
        assert env[3] == 0  # ymax
        ds = None

    def test_with_metadata(self, reference_raster, tmp_path):
        """Test vector extent creation with metadata."""
        out_path = tmp_path / "vector_extent_metadata.gpkg"
        result = _raster_to_vector_extent(
            reference_raster,
            str(out_path),
            metadata=True
        )
        
        ds = ogr.Open(str(out_path))
        layer = ds.GetLayer(0)
        feature = layer.GetNextFeature()
        
        # Check metadata fields exist and have values
        assert feature.GetField("width") == 10
        assert feature.GetField("height") == 10
        assert feature.GetField("bands") == 1
        assert feature.GetField("dtype") is not None
        assert feature.GetField("pixel_width") == 1.0
        ds = None

    def test_with_latlng(self, reference_raster, tmp_path):
        """Test vector extent creation with lat/lng coordinates."""
        out_path = tmp_path / "vector_extent_latlng.gpkg"
        result = _raster_to_vector_extent(
            reference_raster,
            str(out_path),
            latlng=True
        )
        
        ds = ogr.Open(str(out_path))
        layer = ds.GetLayer(0)
        srs = layer.GetSpatialRef()
        assert srs.GetAuthorityCode(None) == "4326"
        ds = None

    def test_different_output_format(self, reference_raster, tmp_path):
        """Test vector extent creation with different output format."""
        out_path = tmp_path / "vector_extent.shp"
        result = _raster_to_vector_extent(
            reference_raster,
            str(out_path),
            out_format="shp"
        )
        
        assert Path(result).exists()
        assert result.endswith(".shp")
        ds = ogr.Open(str(out_path))
        assert ds is not None
        ds = None

    def test_with_prefix_suffix(self, reference_raster, tmp_path):
        """Test vector extent creation with prefix and suffix."""
        out_path = tmp_path / "extent.gpkg"
        result = _raster_to_vector_extent(
            reference_raster,
            str(out_path),
            prefix="test_",
            suffix="_extent"
        )
        
        assert "test_" in Path(result).name
        assert "_extent" in Path(result).stem

class TestRasterGetFootprints:
    def test_single_raster_footprint(self, reference_raster, tmp_path):
        """Test getting footprint for single raster."""
        out_path = tmp_path / "footprint.gpkg"
        result = raster_to_vector_extent(
            reference_raster,
            str(out_path)
        )
        
        assert Path(result).exists()
        ds = ogr.Open(str(result))
        layer = ds.GetLayer(0)
        assert layer.GetFeatureCount() == 1
        ds = None

    def test_multiple_raster_footprints(self, reference_raster, overlapping_raster, tmp_path):
        """Test getting footprints for multiple rasters."""
        out_paths = [
            tmp_path / "footprint1.gpkg",
            tmp_path / "footprint2.gpkg"
        ]
        results = raster_to_vector_extent(
            [reference_raster, overlapping_raster],
            [str(p) for p in out_paths]
        )
        
        assert len(results) == 2
        assert all(Path(p).exists() for p in results)
        
        # Check each footprint
        for result in results:
            ds = ogr.Open(str(result))
            layer = ds.GetLayer(0)
            assert layer.GetFeatureCount() == 1
            ds = None

    def test_footprints_with_metadata(self, reference_raster, tmp_path):
        """Test getting footprints with metadata."""
        out_path = tmp_path / "footprint_metadata.gpkg"
        result = raster_to_vector_extent(
            reference_raster,
            str(out_path),
            metadata=True
        )
        
        ds = ogr.Open(str(result))
        layer = ds.GetLayer(0)
        feature = layer.GetNextFeature()
        
        # Check metadata fields
        assert feature.GetField("width") is not None
        assert feature.GetField("height") is not None
        assert feature.GetField("bands") is not None
        ds = None

    def test_footprints_latlng(self, reference_raster, tmp_path):
        """Test getting footprints in lat/lng coordinates."""
        out_path = tmp_path / "footprint_latlng.gpkg"
        result = raster_to_vector_extent(
            reference_raster,
            str(out_path),
            latlng=True
        )
        
        ds = ogr.Open(str(result))
        layer = ds.GetLayer(0)
        srs = layer.GetSpatialRef()
        assert srs.GetAuthorityCode(None) == "4326"
        ds = None

    def test_footprints_with_uuid(self, reference_raster, tmp_path):
        """Test getting footprints with UUID in filename."""
        out_path = tmp_path / "footprint.gpkg"
        result = raster_to_vector_extent(
            reference_raster,
            str(out_path),
            add_uuid=True
        )
        
        # Check that filename contains a UUID pattern
        assert len(Path(result).stem) > len(Path(out_path).stem)
        assert Path(result).exists()

    def test_invalid_inputs(self, reference_raster):
        """Test invalid inputs for getting footprints."""
        with pytest.raises(TypeError):
            raster_to_vector_extent(123)
        
        with pytest.raises(ValueError):
            raster_to_vector_extent(
                [reference_raster, reference_raster],
                "single_output.gpkg"  # Should be list for multiple inputs
            )

    def test_footprints_different_formats(self, reference_raster, tmp_path):
        """Test getting footprints in different output formats."""
        out_path = tmp_path / "footprint.shp"
        result = raster_to_vector_extent(
            reference_raster,
            str(out_path),
            out_format="shp"
        )
        
        assert Path(result).exists()
        assert result.endswith(".shp")
        ds = ogr.Open(str(result))
        assert ds is not None
        ds = None