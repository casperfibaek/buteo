""" Tests for core_raster.py """


# Standard library
import sys; sys.path.append("../")

# External
import numpy as np
import pytest
from osgeo import gdal

# Internal
from utils_tests import create_sample_raster
from buteo.raster import core_raster


# Test functions
def test_open_raster_single():
    """ Test: Open raster file. """
    raster_1 = create_sample_raster()
    raster = core_raster.raster_open(raster_1, writeable=False)
    assert isinstance(raster, gdal.Dataset)

    gdal.Unlink(raster_1)

def test_open_raster_list():
    """ Test: Open list of raster files. """
    raster_1 = create_sample_raster()
    raster_2 = create_sample_raster(nodata=0.0)
    rasters = core_raster.raster_open([raster_1, raster_2], writeable=False)
    assert isinstance(rasters, list) and len(rasters) == 2
    assert all(isinstance(r, gdal.Dataset) for r in rasters)

    gdal.Unlink(raster_1)
    gdal.Unlink(raster_2)

def test_open_raster_invalid_input():
    """ Test: Open raster file - invalid. """
    with pytest.raises(ValueError):
        core_raster.raster_open("non_existent_file.tif", writeable=False)

def test_open_raster_write_mode():
    """ Test: Open raster file in write mode. """
    raster_1 = create_sample_raster()
    raster = core_raster.raster_open(raster_1, writeable=True)
    assert isinstance(raster, gdal.Dataset)

    error_code = raster.GetRasterBand(1).WriteArray(np.zeros((10, 10)))
    is_writeable = error_code == 0

    assert is_writeable

    gdal.Unlink(raster_1)

def test_open_raster_read_mode():
    """ Test: Open raster file in read mode. """
    raster_1 = create_sample_raster()
    raster = core_raster.raster_open(raster_1, writeable=False)
    assert isinstance(raster, gdal.Dataset)

    with pytest.raises(RuntimeError):
        raster.GetRasterBand(1).WriteArray(np.zeros((10, 10)))

    gdal.Unlink(raster_1)


def test_raster_to_metadata():
    """ Test: raster to metadata. """
    raster_1 = create_sample_raster()
    metadata = core_raster._get_basic_metadata_raster(raster_1)

    assert isinstance(metadata, dict)
    assert metadata["width"] == 10
    assert metadata["height"] == 10
    assert metadata["bands"] == 1
    assert metadata["pixel_width"] == 1
    assert metadata["pixel_height"] == 1
    assert metadata["x_min"] == 0
    assert metadata["x_max"] == 10
    assert metadata["y_min"] == 0
    assert metadata["y_max"] == 10
    assert metadata["dtype_name"] == "uint8"
    assert metadata["nodata"] is False
    assert metadata["in_memory"] is True

    gdal.Unlink(raster_1)

def test_raster_to_metadata_nodata():
    """ Test: raster to metadata. """
    raster_2 = create_sample_raster(nodata=0.0)
    metadata = core_raster._get_basic_metadata_raster(raster_2)

    assert isinstance(metadata, dict)
    assert metadata["width"] == 10
    assert metadata["height"] == 10
    assert metadata["bands"] == 1
    assert metadata["pixel_width"] == 1
    assert metadata["pixel_height"] == 1
    assert metadata["x_min"] == 0
    assert metadata["x_max"] == 10
    assert metadata["y_min"] == 0
    assert metadata["y_max"] == 10
    assert metadata["dtype_name"] == "uint8"
    assert metadata["nodata"] is True

    gdal.Unlink(raster_2)

def test_raster_to_metadata_single_raster():
    """ Test: raster to metadata. Single raster. """
    raster_1 = create_sample_raster()
    metadata = core_raster._get_basic_metadata_raster(raster_1)

    assert isinstance(metadata, dict)
    assert metadata["width"] == 10
    assert metadata["height"] == 10
    assert metadata["bands"] == 1

    gdal.Unlink(raster_1)

def test_rasters_are_aligned_same_projection():
    """ Test: rasters are aligned. Same projection. """
    raster_1 = create_sample_raster()
    raster_2 = create_sample_raster()
    assert core_raster.check_rasters_are_aligned([raster_1, raster_2])

    gdal.Unlink(raster_1)
    gdal.Unlink(raster_2)

def test_rasters_are_aligned_different_projection():
    """ Test: rasters are aligned. Different projection. """
    raster_1 = create_sample_raster(epsg_code=4326)
    raster_2 = create_sample_raster(epsg_code=32632)

    assert not core_raster.check_rasters_are_aligned([raster_1, raster_2])

    gdal.Unlink(raster_1)
    gdal.Unlink(raster_2)

def test_rasters_are_aligned_same_extent():
    """ Test: rasters are aligned. Same extent. """
    raster_1 = create_sample_raster()
    raster_2 = create_sample_raster()
    assert core_raster.check_rasters_are_aligned([raster_1, raster_2])

    gdal.Unlink(raster_1)
    gdal.Unlink(raster_2)

def test_rasters_are_aligned_different_extent():
    """ Test: rasters are aligned. Different extent. """
    raster_1 = create_sample_raster()
    raster_2 = create_sample_raster(width=20, height=20)
    assert not core_raster.check_rasters_are_aligned([raster_1, raster_2])

    gdal.Unlink(raster_1)
    gdal.Unlink(raster_2)

def test_rasters_are_aligned_same_dtype():
    """ Test: rasters are aligned. Same dtype. """
    raster_1 = create_sample_raster()
    raster_2 = create_sample_raster()
    assert core_raster.check_rasters_are_aligned([raster_1, raster_2], same_dtype=True)

    gdal.Unlink(raster_1)
    gdal.Unlink(raster_2)

def test_rasters_are_aligned_different_dtype():
    """ Test: rasters are aligned. Different dtype. """
    raster_1 = create_sample_raster(datatype=gdal.GDT_Byte)
    raster_2 = create_sample_raster(datatype=gdal.GDT_UInt16)
    assert not core_raster.check_rasters_are_aligned([raster_1, raster_2], same_dtype=True)

    gdal.Unlink(raster_1)
    gdal.Unlink(raster_2)

def test_raster_has_nodata_true():
    """ Test: raster has nodata. True case. """
    raster_1 = create_sample_raster(nodata=0)
    assert core_raster._check_raster_has_nodata(raster_1)

    gdal.Unlink(raster_1)

def test_raster_has_nodata_false():
    """ Test: raster has nodata. False case. """
    raster_1 = create_sample_raster()
    assert not core_raster._check_raster_has_nodata(raster_1)

    gdal.Unlink(raster_1)

def test_rasters_have_nodata_true():
    """ Test: rasters have nodata. True case. """
    raster_1 = create_sample_raster(nodata=0)
    raster_2 = create_sample_raster()
    assert core_raster._check_raster_has_nodata_list([raster_1, raster_2])

    gdal.Unlink(raster_1)
    gdal.Unlink(raster_2)

def test_rasters_have_nodata_false():
    """ Test: rasters have nodata. False case. """
    raster_1 = create_sample_raster()
    raster_2 = create_sample_raster()
    assert not core_raster._check_raster_has_nodata_list([raster_1, raster_2])

    gdal.Unlink(raster_1)
    gdal.Unlink(raster_2)

def test_rasters_have_same_nodata_true():
    """ Test: rasters have same nodata. True case. """
    raster_1 = create_sample_raster(nodata=0)
    raster_2 = create_sample_raster(nodata=0)
    assert core_raster.check_rasters_have_same_nodata([raster_1, raster_2])

    gdal.Unlink(raster_1)
    gdal.Unlink(raster_2)

def test_rasters_have_same_nodata_false():
    """ Test: rasters have same nodata. False case. """
    raster_1 = create_sample_raster(nodata=0)
    raster_2 = create_sample_raster(nodata=1)
    assert not core_raster.check_rasters_have_same_nodata([raster_1, raster_2])

    gdal.Unlink(raster_1)
    gdal.Unlink(raster_2)

def test_get_first_nodata_value():
    """ Test: get first nodata value. """
    raster_1 = create_sample_raster(nodata=0)
    assert core_raster._get_first_nodata_value(raster_1) == 0

    gdal.Unlink(raster_1)

def test_count_bands_in_rasters_single_band_rasters():
    """ Test: count bands in rasters. Single band rasters. """
    raster1 = create_sample_raster(width=10, height=10, bands=1)
    raster2 = create_sample_raster(width=10, height=10, bands=1)
    raster3 = create_sample_raster(width=10, height=10, bands=1)

    rasters = [raster1, raster2, raster3]
    bands = core_raster._raster_count_bands_list(rasters)

    assert bands == 3

    gdal.Unlink(raster1)
    gdal.Unlink(raster2)
    gdal.Unlink(raster3)

def test_count_bands_in_rasters_multiple_band_rasters():
    """ Test: count bands in rasters. Multiple band rasters. """
    raster1 = create_sample_raster(width=10, height=10, bands=2)
    raster2 = create_sample_raster(width=10, height=10, bands=3)
    raster3 = create_sample_raster(width=10, height=10, bands=1)

    rasters = [raster1, raster2, raster3]
    bands = core_raster._raster_count_bands_list(rasters)

    assert bands == 6

    gdal.Unlink(raster1)
    gdal.Unlink(raster2)
    gdal.Unlink(raster3)

def test_rasters_intersect_true():
    """ Test: Rasters intersect. True Case """
    raster1 = create_sample_raster(height=10, width=10, pixel_height=1, pixel_width=1)
    raster2 = create_sample_raster(height=10, width=10, pixel_height=1, pixel_width=1, x_min=5, y_max=15)

    assert core_raster.check_rasters_intersect(raster1, raster2) is True

    gdal.Unlink(raster1)
    gdal.Unlink(raster2)

def test_rasters_intersect_false():
    """ Test: Rasters intersect. False Case """
    raster1 = create_sample_raster(height=10, width=10, pixel_height=1, pixel_width=1)
    raster2 = create_sample_raster(height=10, width=10, pixel_height=1, pixel_width=1, x_min=15, y_max=25)

    assert core_raster.check_rasters_intersect(raster1, raster2) is False

    gdal.Unlink(raster1)
    gdal.Unlink(raster2)

def test_rasters_intersect_same():
    """ Test: Rasters intersect. Same Case """
    raster1 = create_sample_raster(width=10, height=10, x_min=0, y_max=10)
    raster2 = create_sample_raster(width=10, height=10, x_min=0, y_max=10)

    assert core_raster.check_rasters_intersect(raster1, raster2) is True

    gdal.Unlink(raster1)
    gdal.Unlink(raster2)

def test_rasters_intersect_different_pixelsizes():
    """ Test: Rasters intersect. Different pixel sizes """
    raster1 = create_sample_raster(width=10, height=10, pixel_width=1, pixel_height=1, x_min=0, y_max=10)
    raster2 = create_sample_raster(width=20, height=20, pixel_width=0.5, pixel_height=0.5, x_min=5, y_max=15)

    assert core_raster.check_rasters_intersect(raster1, raster2) is True

    gdal.Unlink(raster1)
    gdal.Unlink(raster2)

def rasters_intersect_non_intersecting_difpixel():
    """ Test: Rasters intersect. Different pixel sizes, non-intersecting """
    raster1 = create_sample_raster(width=10, height=10, pixel_width=1, pixel_height=1, x_min=0, y_max=10)
    raster2 = create_sample_raster(width=40, height=40, pixel_width=0.5, pixel_height=0.5, x_min=20, y_max=30)

    assert core_raster.check_rasters_intersect(raster1, raster2) is False

    gdal.Unlink(raster1)
    gdal.Unlink(raster2)

def test_rasters_intersection_true():
    """ Test: Rasters intersection. True Case """ 
    raster1 = create_sample_raster(width=10, height=10, x_min=0, y_max=10)
    raster2 = create_sample_raster(width=10, height=10, x_min=5, y_max=15)
    intersection1 = core_raster.get_raster_intersection(raster1, raster2)

    assert intersection1 is not None

    gdal.Unlink(raster1)
    gdal.Unlink(raster2)

def test_rasters_intersection_false():
    """ Test: Rasters intersection. False Case """
    raster1 = create_sample_raster(width=10, height=10, x_min=0, y_max=10)
    raster2 = create_sample_raster(width=10, height=10, x_min=20, y_max=30)

    with pytest.raises(ValueError):
        core_raster.get_raster_intersection(raster1, raster2)

    gdal.Unlink(raster1)
    gdal.Unlink(raster2)

def test_get_overlap_fraction():
    """ Test: Get overlap fraction. Complete overlap. """
    raster1 = create_sample_raster(width=10, height=10, x_min=0, y_max=10)
    raster2 = create_sample_raster(width=10, height=10, x_min=0, y_max=10)
    overlap1 = core_raster.get_raster_overlap_fraction(raster1, raster2)

    assert overlap1 == 1.0

    gdal.Unlink(raster1)
    gdal.Unlink(raster2)

def test_get_overlap_fraction_partial():
    """ Test: Get overlap fraction. Partial overlap."""
    raster1 = create_sample_raster(width=10, height=10, x_min=0, y_max=10)
    raster2 = create_sample_raster(width=10, height=10, x_min=5, y_max=15)
    overlap = core_raster.get_raster_overlap_fraction(raster1, raster2)

    assert 0 < overlap < 1

    gdal.Unlink(raster1)
    gdal.Unlink(raster2)

def test_get_overlap_fraction_no_overlap():
    """ Test: Get overlap fraction. No overlap."""
    raster1 = create_sample_raster(width=10, height=10, x_min=0, y_max=10)
    raster2 = create_sample_raster(width=10, height=10, x_min=20, y_max=30)
    overlap = core_raster.get_raster_overlap_fraction(raster1, raster2)

    assert overlap == 0

    gdal.Unlink(raster1)
    gdal.Unlink(raster2)
