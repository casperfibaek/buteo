""" Tests for core_raster.py """


# Standard library
import sys; sys.path.append("../")
from uuid import uuid4

# External
import numpy as np
import pytest
from osgeo import gdal, osr

# Internal
from buteo.raster.core_raster import (
    raster_to_metadata,
    open_raster,
    raster_to_array,
    rasters_are_aligned,
    raster_has_nodata,
    rasters_have_nodata,
    rasters_have_same_nodata,
    get_first_nodata_value,
    count_bands_in_rasters,
    array_to_raster,
    raster_set_datatype,
    stack_rasters,
    stack_rasters_vrt,
    rasters_intersect,
    rasters_intersection,
    get_overlap_fraction,
    create_raster_from_array,
    create_grid_with_coordinates,
)
from buteo.utils import gdal_enums, gdal_utils

DEFAULT_ARR = np.array([
    [178, 250, 117,  23, 250,  42, 166, 164,  84, 175],
    [ 22,  87, 142, 176,  33, 212,  39,  66, 183, 172],
    [104,  94,  82, 229,  98, 110, 122,  22, 163, 217],
    [251,  21,  40,   2, 227, 142,   9,  63,  66, 207],
    [149,  27, 137,   8,  66, 228, 105,  39, 113, 210],
    [170, 120,  37,  80,  70,  51, 254,  23, 204, 233],
    [116, 242,  73, 133, 151,  55, 231, 193,  24,  34],
    [249,  74,   3, 143, 148,  60, 169, 161, 154,  17],
    [ 39, 198,  61,  40, 249, 232, 198,   3,  30, 180],
    [199, 179, 237, 103, 197, 215, 237, 157,  38, 117]
], dtype=np.uint8)


def get_sample_raster():
    """ Create a sample raster file for testing purposes. """
    filename = f"/vsimem/tmp_raster_{uuid4().int}.tif"
    driver = gdal.GetDriverByName("GTiff")
    raster = driver.Create(filename, 10, 10, 1, gdal.GDT_Byte)
    raster.SetGeoTransform([0, 1, 0, 0, 0, -1])
    raster.GetRasterBand(1).WriteArray(DEFAULT_ARR)
    raster.FlushCache()
    raster = None

    return filename


def get_sample_raster_with_nodata():
    """ Create a sample raster file for testing purposes. """
    filename = f"/vsimem/tmp_raster_nodata_{uuid4().int}.tif"
    driver = gdal.GetDriverByName("GTiff")
    raster = driver.Create(filename, 10, 10, 1, gdal.GDT_Byte)
    raster.SetGeoTransform([0, 1, 0, 0, 0, -1])
    raster.GetRasterBand(1).WriteArray(DEFAULT_ARR)
    raster.GetRasterBand(1).SetNoDataValue(0)
    raster.FlushCache()
    raster = None

    return filename

def create_sample_raster(
        width=10,
        height=10,
        bands=1,
        pixel_width=1,
        pixel_height=1,
        x_min=None,
        y_max=None,
        epsg_code=4326,
        datatype=gdal.GDT_Byte,
        nodata=None,
    ):
    """ Create a sample raster file for testing purposes. """
    raster_path = f"/vsimem/mem_raster_{uuid4().int}.tif"
    driver = gdal.GetDriverByName("GTiff")
    raster = driver.Create(raster_path, width, height, bands, datatype)

    if y_max is None:
        y_max = height * pixel_height
    if x_min is None:
        x_min = 0

    raster.SetGeoTransform((x_min, pixel_width, 0, y_max, 0, -pixel_height))

    for band in range(1, bands + 1):
        raster.GetRasterBand(band).WriteArray(np.random.randint(0, 255, (height, width), dtype=np.uint8))

    if nodata is not None:
        for band in range(1, bands + 1):
            raster.GetRasterBand(band).SetNoDataValue(float(nodata))

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg_code)
    raster.SetProjection(srs.ExportToWkt())
    raster.FlushCache()
    raster = None

    return raster_path


def test_create_sample_raster():
    """ Test creating a sample raster with default arguments. """
    # Test creating a sample raster with default arguments
    raster_path = create_sample_raster()

    ds = gdal.Open(raster_path)
    assert ds.RasterCount == 1
    assert ds.GetRasterBand(1).DataType == gdal.GDT_Byte
    assert ds.GetRasterBand(1).GetNoDataValue() is None
    assert ds.RasterXSize == 10
    assert ds.RasterYSize == 10

    gt = ds.GetGeoTransform()
    assert gt[0] == 0
    assert gt[1] == 1
    assert gt[2] == 0
    assert gt[3] == 10
    assert gt[4] == 0
    assert gt[5] == -1

    proj = ds.GetProjection()
    assert 'GEOGCS["WGS 84"' in proj
    assert ds.GetProjectionRef() == proj

    ds = None
    gdal.Unlink(raster_path)

    raster_path = create_sample_raster(
        width=20,
        height=30,
        bands=3,
        pixel_width=2,
        pixel_height=3,
        x_min=-10,
        y_max=50,
        epsg_code=3857,
        datatype=gdal.GDT_Float32,
        nodata=-9999,
    )
    ds = gdal.Open(raster_path)
    assert ds.RasterCount == 3
    assert ds.GetRasterBand(1).DataType == gdal.GDT_Float32
    assert ds.GetRasterBand(1).GetNoDataValue() == -9999.0
    assert ds.GetRasterBand(2).DataType == gdal.GDT_Float32
    assert ds.GetRasterBand(2).GetNoDataValue() == -9999.0
    assert ds.GetRasterBand(3).DataType == gdal.GDT_Float32
    assert ds.GetRasterBand(3).GetNoDataValue() == -9999.0
    assert ds.RasterXSize == 20
    assert ds.RasterYSize == 30

    gt = ds.GetGeoTransform()
    assert gt[0] == -10
    assert gt[1] == 2
    assert gt[2] == 0
    assert gt[3] == 50
    assert gt[4] == 0
    assert gt[5] == -3

    proj = ds.GetProjection()
    assert 'PROJCS["WGS 84 / Pseudo-Mercator"' in proj
    assert ds.GetProjectionRef() == proj

    ds = None
    gdal.Unlink(raster_path)


# Test functions
def test_open_raster_single():
    """ Test: Open raster file. """
    raster_1 = get_sample_raster()
    raster = open_raster(raster_1, writeable=False, allow_lists=False)
    assert isinstance(raster, gdal.Dataset)

    gdal.Unlink(raster_1)

def test_open_raster_list():
    """ Test: Open list of raster files. """
    raster_1 = get_sample_raster()
    raster_2 = get_sample_raster_with_nodata()
    rasters = open_raster([raster_1, raster_2], writeable=False)
    assert isinstance(rasters, list) and len(rasters) == 2
    assert all(isinstance(r, gdal.Dataset) for r in rasters)

    gdal.Unlink(raster_1)
    gdal.Unlink(raster_2)

def test_open_raster_invalid_input():
    """ Test: Open raster file - invalid. """
    with pytest.raises(ValueError):
        open_raster("non_existent_file.tif", writeable=False)

def test_open_raster_list_not_allowed():
    """ Test: Open list of raster files - not allowed. """
    raster_1 = get_sample_raster()
    raster_2 = get_sample_raster_with_nodata()
    with pytest.raises(ValueError):
        open_raster([raster_1, raster_2], allow_lists=False)

    gdal.Unlink(raster_1)
    gdal.Unlink(raster_2)

def test_open_raster_write_mode():
    """ Test: Open raster file in write mode. """
    raster_1 = get_sample_raster()
    raster = open_raster(raster_1, writeable=True)
    assert isinstance(raster, gdal.Dataset)

    error_code = raster.GetRasterBand(1).WriteArray(np.zeros((10, 10)))
    is_writeable = error_code == 0

    assert is_writeable

    gdal.Unlink(raster_1)

def test_open_raster_read_mode():
    """ Test: Open raster file in read mode. """
    raster_1 = get_sample_raster()
    raster = open_raster(raster_1, writeable=False)
    assert isinstance(raster, gdal.Dataset)

    error_code = raster.GetRasterBand(1).WriteArray(np.zeros((10, 10)))
    is_writeable = error_code == 0

    assert not is_writeable


    gdal.Unlink(raster_1)

def test_raster_to_metadata():
    """ Test: raster to metadata. """
    raster_1 = get_sample_raster()
    metadata = raster_to_metadata(raster_1)

    assert isinstance(metadata, dict)
    assert metadata["width"] == 10
    assert metadata["height"] == 10
    assert metadata["band_count"] == 1
    assert metadata["pixel_width"] == 1
    assert metadata["pixel_height"] == 1
    assert metadata["x_min"] == 0
    assert metadata["y_max"] == 0
    assert metadata["x_max"] == 10
    assert metadata["y_min"] == -10
    assert metadata["dtype"] == "uint8"
    assert metadata["has_nodata"] is False
    assert metadata["get_bbox_vector"] is not None
    assert metadata["get_bbox_vector_latlng"] is not None

    gdal.Unlink(raster_1)

def test_raster_to_metadata_nodata():
    """ Test: raster to metadata. """
    raster_2 = get_sample_raster_with_nodata()
    metadata = raster_to_metadata(raster_2)

    assert isinstance(metadata, dict)
    assert metadata["width"] == 10
    assert metadata["height"] == 10
    assert metadata["band_count"] == 1
    assert metadata["pixel_width"] == 1
    assert metadata["pixel_height"] == 1
    assert metadata["x_min"] == 0
    assert metadata["y_max"] == 0
    assert metadata["x_max"] == 10
    assert metadata["y_min"] == -10
    assert metadata["dtype"] == "uint8"
    assert metadata["has_nodata"] is True
    assert metadata["get_bbox_vector"] is not None
    assert metadata["get_bbox_vector_latlng"] is not None

    gdal.Unlink(raster_2)

def test_raster_to_metadata_single_raster():
    """ Test: raster to metadata. Single raster. """
    raster_1 = get_sample_raster()
    metadata = raster_to_metadata(raster_1, allow_lists=False)

    assert isinstance(metadata, dict)
    assert metadata["width"] == 10
    assert metadata["height"] == 10
    assert metadata["band_count"] == 1

    gdal.Unlink(raster_1)

def test_raster_to_metadata_list_of_rasters():
    """ Test: raster to metadata. List of rasters. """
    raster_1 = get_sample_raster()
    raster_2 = get_sample_raster_with_nodata()
    metadata_list = raster_to_metadata([raster_1, raster_2])

    assert isinstance(metadata_list, list)
    assert len(metadata_list) == 2
    for metadata in metadata_list:
        assert metadata["width"] == 10
        assert metadata["height"] == 10
        assert metadata["band_count"] == 1

    gdal.Unlink(raster_1)
    gdal.Unlink(raster_2)

def test_raster_to_metadata_disallow_lists():
    """ Test: raster to metadata. Disallow lists. """
    raster_1 = get_sample_raster()
    raster_2 = get_sample_raster()

    with pytest.raises(ValueError):
        raster_to_metadata([raster_1, raster_2], allow_lists=False)

    gdal.Unlink(raster_1)
    gdal.Unlink(raster_2)

def test_rasters_are_aligned_single_raster():
    """ Test: rasters are aligned. Single raster. """
    raster_1 = create_sample_raster()
    assert rasters_are_aligned([raster_1])

    gdal.Unlink(raster_1)

def test_rasters_are_aligned_same_projection():
    """ Test: rasters are aligned. Same projection. """
    raster_1 = create_sample_raster()
    raster_2 = create_sample_raster()
    assert rasters_are_aligned([raster_1, raster_2])

    gdal.Unlink(raster_1)
    gdal.Unlink(raster_2)

def test_rasters_are_aligned_different_projection():
    """ Test: rasters are aligned. Different projection. """
    raster_1 = create_sample_raster(epsg_code=4326)
    raster_2 = create_sample_raster(epsg_code=32632)

    assert not rasters_are_aligned([raster_1, raster_2])

    gdal.Unlink(raster_1)
    gdal.Unlink(raster_2)

def test_rasters_are_aligned_same_extent():
    """ Test: rasters are aligned. Same extent. """
    raster_1 = create_sample_raster()
    raster_2 = create_sample_raster()
    assert rasters_are_aligned([raster_1, raster_2], same_extent=True)

    gdal.Unlink(raster_1)
    gdal.Unlink(raster_2)

def test_rasters_are_aligned_different_extent():
    """ Test: rasters are aligned. Different extent. """
    raster_1 = create_sample_raster()
    raster_2 = create_sample_raster(width=20, height=20)
    assert not rasters_are_aligned([raster_1, raster_2], same_extent=True)

    gdal.Unlink(raster_1)
    gdal.Unlink(raster_2)

def test_rasters_are_aligned_same_dtype():
    """ Test: rasters are aligned. Same dtype. """
    raster_1 = create_sample_raster()
    raster_2 = create_sample_raster()
    assert rasters_are_aligned([raster_1, raster_2], same_dtype=True)

    gdal.Unlink(raster_1)
    gdal.Unlink(raster_2)

def test_rasters_are_aligned_different_dtype():
    """ Test: rasters are aligned. Different dtype. """
    raster_1 = create_sample_raster(datatype=gdal.GDT_Byte)
    raster_2 = create_sample_raster(datatype=gdal.GDT_UInt16)
    assert not rasters_are_aligned([raster_1, raster_2], same_dtype=True)

    gdal.Unlink(raster_1)
    gdal.Unlink(raster_2)

def test_raster_has_nodata_true():
    """ Test: raster has nodata. True case. """
    raster_1 = create_sample_raster(nodata=0)
    assert raster_has_nodata(raster_1)

    gdal.Unlink(raster_1)

def test_raster_has_nodata_false():
    """ Test: raster has nodata. False case. """
    raster_1 = create_sample_raster()
    assert not raster_has_nodata(raster_1)

    gdal.Unlink(raster_1)

def test_rasters_have_nodata_true():
    """ Test: rasters have nodata. True case. """
    raster_1 = create_sample_raster(nodata=0)
    raster_2 = create_sample_raster()
    assert rasters_have_nodata([raster_1, raster_2])

    gdal.Unlink(raster_1)
    gdal.Unlink(raster_2)

def test_rasters_have_nodata_false():
    """ Test: rasters have nodata. False case. """
    raster_1 = create_sample_raster()
    raster_2 = create_sample_raster()
    assert not rasters_have_nodata([raster_1, raster_2])

    gdal.Unlink(raster_1)
    gdal.Unlink(raster_2)

def test_rasters_have_same_nodata_true():
    """ Test: rasters have same nodata. True case. """
    raster_1 = create_sample_raster(nodata=0)
    raster_2 = create_sample_raster(nodata=0)
    assert rasters_have_same_nodata([raster_1, raster_2])

    gdal.Unlink(raster_1)
    gdal.Unlink(raster_2)

def test_rasters_have_same_nodata_false():
    """ Test: rasters have same nodata. False case. """
    raster_1 = create_sample_raster(nodata=0)
    raster_2 = create_sample_raster(nodata=1)
    assert not rasters_have_same_nodata([raster_1, raster_2])

    gdal.Unlink(raster_1)
    gdal.Unlink(raster_2)

def test_get_first_nodata_value():
    """ Test: get first nodata value. """
    raster_1 = create_sample_raster(nodata=0)
    assert get_first_nodata_value(raster_1) == 0

    gdal.Unlink(raster_1)

def test_count_bands_in_rasters_single_band_rasters():
    """ Test: count bands in rasters. Single band rasters. """
    raster1 = create_sample_raster(width=10, height=10, bands=1)
    raster2 = create_sample_raster(width=10, height=10, bands=1)
    raster3 = create_sample_raster(width=10, height=10, bands=1)

    rasters = [raster1, raster2, raster3]
    band_count = count_bands_in_rasters(rasters)

    assert band_count == 3

    gdal.Unlink(raster1)
    gdal.Unlink(raster2)
    gdal.Unlink(raster3)

def test_count_bands_in_rasters_multiple_band_rasters():
    """ Test: count bands in rasters. Multiple band rasters. """
    raster1 = create_sample_raster(width=10, height=10, bands=2)
    raster2 = create_sample_raster(width=10, height=10, bands=3)
    raster3 = create_sample_raster(width=10, height=10, bands=1)

    rasters = [raster1, raster2, raster3]
    band_count = count_bands_in_rasters(rasters)

    assert band_count == 6

    gdal.Unlink(raster1)
    gdal.Unlink(raster2)
    gdal.Unlink(raster3)

def test_raster_to_array_shape():
    """ Test: Convert raster to array """
    raster = get_sample_raster()

    array = raster_to_array(raster)

    gdal.Unlink(raster)
    assert array.shape == (10, 10, 1)

def test_raster_to_array_dtype():
    """ Test: Convert raster to array. """
    raster = get_sample_raster()

    array = raster_to_array(raster)

    gdal.Unlink(raster)
    assert array.dtype == np.uint8

def test_raster_to_array_invalid_raster():
    """ Test: Convert raster to array. (invalid) """
    with pytest.raises(ValueError):
        raster_to_array("non_existent_file.tif")

def test_raster_to_array_bbox():
    """ Test: raster to array with bbox. """
    x_min, x_max, y_min, y_max = [0.0, 5.0, -5.0, 0.0]
    bbox = [x_min, x_max, y_min, y_max]
    raster = get_sample_raster()

    array = raster_to_array(raster, bbox=bbox)
    gdal.Unlink(raster)

    assert array.shape == (5, 5, 1)

def test_raster_to_array_pixel_offsets():
    """ Test: raster to array with pixel offsets. """
    pixel_offsets = [2, 2, 6, 6]
    raster = get_sample_raster()

    array = raster_to_array(raster, pixel_offsets=pixel_offsets)

    gdal.Unlink(raster)

    assert array.shape == (6, 6, 1)

def test_raster_to_array_invalid_bbox():
    """ Test: raster to array with invalid bbox. """
    raster = get_sample_raster()

    with pytest.raises(ValueError):
        raster_to_array(raster, bbox=[-20, -10, -20, -10])

    gdal.Unlink(raster)

def test_raster_to_array_invalid_pixel_offsets():
    """ Test: raster to array with invalid pixel offsets. """
    raster = get_sample_raster()

    with pytest.raises(ValueError):
        raster_to_array(raster, pixel_offsets=[-20, -10, -20, -10])

    gdal.Unlink(raster)


def test_raster_to_array_nodata_auto():
    """ Test: raster to array with nodata auto. """
    raster = get_sample_raster_with_nodata()

    array = raster_to_array(raster, masked="auto")

    assert np.ma.isMaskedArray(array)
    assert array.fill_value == 0


def test_raster_to_array_nodata_true():
    """ Test: raster to array with nodata auto. """
    raster = get_sample_raster_with_nodata()

    array = raster_to_array(raster, masked=True)

    assert np.ma.isMaskedArray(array)
    assert array.fill_value == 0


def test_raster_to_array_nodata_false():
    """ Test: raster to array with nodata auto. """
    raster = get_sample_raster_with_nodata()

    array = raster_to_array(raster, masked=False)

    assert not np.ma.isMaskedArray(array)


def test_raster_to_array_multibands():
    """ Test: Convert raster to array. """
    raster = create_sample_raster(bands=3)

    array = raster_to_array(raster)

    assert array.shape == (10, 10, 3)
    gdal.Unlink(raster)


def test_raster_to_array_multibands_extract_single():
    """ Test: Convert raster to array. """
    raster = create_sample_raster(bands=3)

    array = raster_to_array(raster, bands=[1])
    array_all = raster_to_array(raster)

    assert array.shape == (10, 10, 1)
    assert array_all.shape == (10, 10, 3)

    assert np.all(array == array_all[:, :, 0:1])

    gdal.Unlink(raster)

def test_raster_to_array_multibands_extract_two():
    """ Test: Convert raster to array. """
    raster = create_sample_raster(bands=3)

    array_all = raster_to_array(raster)
    array = raster_to_array(raster, bands=[2, 3])

    assert array.shape == (10, 10, 2)
    assert array_all.shape == (10, 10, 3)

    assert np.all(array == array_all[:, :, 1:3])

    gdal.Unlink(raster)


def test_raster_to_array_multibands_extract_two_2():
    """ Test: Convert raster to array. """
    raster = create_sample_raster(bands=3)

    array_all = raster_to_array(raster)
    array = raster_to_array(raster, bands=[1, 3])

    assert array.shape == (10, 10, 2)
    assert array_all.shape == (10, 10, 3)

    assert np.all(array == array_all[:, :, [0, 2]])

    gdal.Unlink(raster)


# ARRAY_TO_RASTER
def test_array_to_raster_simple_case_3D():
    """ Test: Convert array to raster. """
    array = np.random.rand(100, 50, 2).astype(np.float32)

    reference = create_sample_raster(
        height=array.shape[0],
        width=array.shape[1],
        bands=array.shape[2],
        datatype=gdal.GDT_Float32,
    )

    result_mempath = array_to_raster(array, reference=reference)

    metadata = raster_to_metadata(result_mempath)
    shape = metadata["shape"]

    back_to_array = raster_to_array(result_mempath)

    assert shape == array.shape
    assert metadata["dtype"] == array.dtype
    assert np.allclose(array, back_to_array)

    gdal.Unlink(reference)
    gdal.Unlink(result_mempath)

def test_array_to_raster_simple_case_2D():
    """ Test: Convert array to raster. 2D. """
    array = (np.random.rand(50, 100) * 100).astype(np.uint8)

    reference = create_sample_raster(
        height=array.shape[0],
        width=array.shape[1],
        datatype=gdal.GDT_Byte,
    )

    result_mempath = array_to_raster(array, reference=reference)

    metadata = raster_to_metadata(result_mempath)
    shape = metadata["shape"]

    back_to_array = raster_to_array(result_mempath)

    assert shape[0:2] == array.shape
    assert metadata["dtype"] == array.dtype
    assert np.allclose(array[:, :, np.newaxis], back_to_array)
    assert array.dtype == back_to_array.dtype

    gdal.Unlink(reference)
    gdal.Unlink(result_mempath)


def test_array_to_raster_mismatched_shape():
    """ Test: Convert array to raster. Mismatched shape. """
    array = np.random.rand(150, 99, 1).astype(np.float32)

    reference = create_sample_raster(
        height=array.shape[0],
        width=array.shape[1],
        datatype=gdal.GDT_Byte,
    )

    result_mempath = array_to_raster(array, reference=reference, allow_mismatches=True)

    metadata = raster_to_metadata(result_mempath)
    shape = metadata["shape"]

    back_to_array = raster_to_array(result_mempath)

    assert shape == array.shape
    assert shape == back_to_array.shape
    assert metadata["dtype"] == array.dtype
    assert np.allclose(array, back_to_array)
    assert array.dtype == back_to_array.dtype

    gdal.Unlink(result_mempath)


def test_array_to_raster_pixel_offsets():
    """ Test: Convert array to raster. PIXEL_OFFSETS. """

    reference = create_sample_raster(
        height=150,
        width=100,
        bands=3,
        datatype=gdal.GDT_Float32,
    )
    array = raster_to_array(reference)
    array_sub = array[0:25, 0:50, :]

    x_offset, y_offset, x_size, y_size = [0, 0, 50, 25]
    offset = [x_offset, y_offset, x_size, y_size]

    arr_offset = raster_to_array(reference, pixel_offsets=offset)

    result_mempath = array_to_raster(arr_offset, reference=reference, pixel_offsets=offset)

    metadata = raster_to_metadata(result_mempath)
    shape = metadata["shape"]

    back_to_array = raster_to_array(result_mempath)

    assert shape == array_sub.shape
    assert shape == back_to_array.shape
    assert metadata["dtype"] == array_sub.dtype
    assert np.allclose(array_sub, back_to_array)
    assert np.allclose(array_sub, arr_offset)

    gdal.Unlink(result_mempath)
    gdal.Unlink(reference)


def test_array_to_raster_bbox():
    """ Test: Convert array to raster. BBOX. """

    reference = create_sample_raster(
        height=150,
        width=100,
        bands=3,
        datatype=gdal.GDT_Float32,
    )

    x_min, x_max, y_min, y_max = [0, 50, 0, 25]
    bbox = [x_min, x_max, y_min, y_max]

    array = raster_to_array(reference, bbox=bbox)

    result_mempath = array_to_raster(array, reference=reference, bbox=bbox)

    metadata = raster_to_metadata(result_mempath)
    shape = metadata["shape"]

    back_to_array = raster_to_array(result_mempath)

    assert shape == array.shape
    assert shape == back_to_array.shape
    assert metadata["dtype"] == array.dtype
    assert metadata["dtype"] == back_to_array.dtype
    assert np.allclose(array, back_to_array)

    gdal.Unlink(result_mempath)
    gdal.Unlink(reference)


def test_array_to_raster_nodata():
    """ Test: Convert array to raster. NODATA. """

    reference = create_sample_raster(
        height=150,
        width=100,
        bands=3,
        datatype=gdal.GDT_Float32,
        nodata=0,
    )

    array = raster_to_array(reference)

    result_mempath = array_to_raster(array, reference=reference)

    metadata = raster_to_metadata(result_mempath)
    shape = metadata["shape"]

    back_to_array = raster_to_array(result_mempath)

    array_data = np.ma.getdata(array)
    nodata_test = np.ma.masked_where(array_data > 100, array_data)
    nodata_test.fill_value = -9999.9

    nodata_raster = array_to_raster(nodata_test, reference=reference)
    nodata_raster_nodata = raster_to_metadata(nodata_raster)["nodata_value"]

    assert nodata_raster_nodata == nodata_test.fill_value

    assert shape == array.shape
    assert shape == back_to_array.shape
    assert metadata["dtype"] == array.dtype
    assert metadata["dtype"] == back_to_array.dtype
    assert np.allclose(array, back_to_array)

    gdal.Unlink(result_mempath)
    gdal.Unlink(reference)
    gdal.Unlink(nodata_raster)

def test_raster_set_datatype_single():
    """Test: Change the datatype of a single raster."""
    reference = create_sample_raster(
        height=150, width=100, bands=3, datatype=gdal.GDT_Float32
    )

    new_dtype = "UInt16"
    output_path = raster_set_datatype(reference, new_dtype, overwrite=True)

    metadata = raster_to_metadata(output_path)

    assert metadata["dtype"] == gdal_enums.translate_gdal_dtype_to_str(gdal.GDT_UInt16)

    gdal.Unlink(output_path)
    gdal.Unlink(reference)


def test_raster_set_datatype_list():
    """Test: Change the datatype of multiple rasters."""
    reference1 = create_sample_raster(
        height=150, width=100, bands=3, datatype=gdal.GDT_Float32
    )
    reference2 = create_sample_raster(
        height=200, width=150, bands=1, datatype=gdal.GDT_Float32
    )

    new_dtype = "UInt16"
    raster_list = [reference1, reference2]
    output_paths = raster_set_datatype(raster_list, new_dtype, overwrite=True)

    for path in output_paths:
        metadata = raster_to_metadata(path)
        assert metadata["dtype"] == gdal_enums.translate_gdal_dtype_to_str(gdal.GDT_UInt16)

    gdal.Unlink(reference1)
    gdal.Unlink(reference2)

def test_stack_rasters_single_band():
    """ Test: Stack rasters with a single band. """
    raster1 = create_sample_raster()
    raster2 = create_sample_raster()

    output_path = stack_rasters([raster1, raster2], out_path=None)

    output_raster = gdal.Open(output_path)
    assert output_raster.RasterCount == 2

    gdal.Unlink(output_path)
    gdal.Unlink(raster1)
    gdal.Unlink(raster2)

def test_stack_rasters_multiple_bands():
    """ Test: Stack rasters with multiple bands."""
    raster1 = create_sample_raster(bands=3)
    raster2 = create_sample_raster(bands=4)

    output_path = stack_rasters([raster1, raster2], out_path=None)

    output_raster = gdal.Open(output_path)
    assert output_raster.RasterCount == 7

    gdal.Unlink(output_path)
    gdal.Unlink(raster1)
    gdal.Unlink(raster2)

def test_stack_rasters_dtype_conversion():
    """ Test: Stack rasters with different data types. """
    raster1 = create_sample_raster()
    raster2 = create_sample_raster()

    output_path = stack_rasters([raster1, raster2], out_path=None, dtype="Int16")

    output_raster = gdal.Open(output_path)
    assert output_raster.GetRasterBand(1).DataType == gdal.GDT_Int16

    gdal.Unlink(output_path)
    gdal.Unlink(raster1)
    gdal.Unlink(raster2)


def test_stack_rasters_nodata_value():
    """ Test: Stack rasters with a NoDataValue. """
    raster1 = create_sample_raster(nodata=-9999)
    raster2 = create_sample_raster(nodata=-9999)

    output_path = stack_rasters([raster1, raster2], out_path=None)

    output_raster = gdal.Open(output_path)
    assert output_raster.GetRasterBand(1).GetNoDataValue() == -9999

    gdal.Unlink(output_path)
    gdal.Unlink(raster1)
    gdal.Unlink(raster2)

def test_stack_rasters_nodata_value_mismatch():
    """ Test: Stack rasters with different NoDataValues. """
    raster1 = create_sample_raster(nodata=-9999)
    raster2 = create_sample_raster(nodata=-8888)

    with pytest.warns(UserWarning):
        output_path = stack_rasters([raster1, raster2], out_path=None)

    output_raster = gdal.Open(output_path)

    assert output_raster.GetRasterBand(1).GetNoDataValue() is None

    gdal.Unlink(output_path)
    gdal.Unlink(raster1)
    gdal.Unlink(raster2)

def test_stack_rasters_projection_mismatch():
    """ Test: Stack rasters with different projections. """
    raster1 = create_sample_raster(epsg_code=4326)
    raster2 = create_sample_raster(epsg_code=3857)

    with pytest.raises(ValueError):
        stack_rasters([raster1, raster2], out_path=None)

    gdal.Unlink(raster1)
    gdal.Unlink(raster2)

def test_stack_rasters_vrt_single_band():
    """ Test: Stack rasters with a single band using VRT. """
    raster1 = create_sample_raster(width=10, height=10, bands=1, nodata=0)
    raster2 = create_sample_raster(width=10, height=10, bands=1, nodata=0)

    output_path = stack_rasters_vrt([raster1, raster2], "/vsimem/stacked_vrt_single_band.vrt")
    output_dataset = gdal.Open(output_path)

    assert output_dataset is not None
    assert output_dataset.RasterCount == 2

    gdal.Unlink(raster1)
    gdal.Unlink(raster2)
    gdal.Unlink(output_path)

def test_stack_rasters_vrt_multi_band():
    """ Test: Stack rasters with multiple bands using VRT. """
    raster1 = create_sample_raster(width=10, height=10, bands=3, nodata=0)
    raster2 = create_sample_raster(width=10, height=10, bands=3, nodata=0)

    output_path = stack_rasters_vrt([raster1, raster2], "/vsimem/stacked_vrt_multi_band.vrt")
    output_dataset = gdal.Open(output_path)

    assert output_dataset is not None
    assert output_dataset.RasterCount == 6

    gdal.Unlink(raster1)
    gdal.Unlink(raster2)
    gdal.Unlink(output_path)

def test_stack_rasters_vrt_mixed_bands():
    """ Test: Stack rasters with mixed bands using VRT. """
    raster1 = create_sample_raster(width=10, height=10, bands=2, nodata=0)
    raster2 = create_sample_raster(width=10, height=10, bands=3, nodata=0)

    output_path = stack_rasters_vrt([raster1, raster2], "/vsimem/stacked_vrt_mixed_bands.vrt")
    output_dataset = gdal.Open(output_path)

    assert output_dataset is not None
    assert output_dataset.RasterCount == 5

    gdal.Unlink(raster1)
    gdal.Unlink(raster2)
    gdal.Unlink(output_path)

def test_stack_rasters_vrt_separate_false():
    """ Test: Stack rasters with separate=False using VRT. """
    raster1 = create_sample_raster(width=10, height=10, bands=2, nodata=0)
    raster2 = create_sample_raster(width=10, height=10, bands=3, nodata=0)

    with pytest.raises(ValueError):
        stack_rasters_vrt([raster1, raster2], "/vsimem/stacked_vrt_separate_false.vrt", separate=False)

    gdal.Unlink(raster1)
    gdal.Unlink(raster2)

def test_rasters_intersect_true():
    """ Test: Rasters intersect. True Case """
    raster1 = create_sample_raster(height=10, width=10, pixel_height=1, pixel_width=1)
    raster2 = create_sample_raster(height=10, width=10, pixel_height=1, pixel_width=1, x_min=5, y_max=15)

    assert rasters_intersect(raster1, raster2) is True

    gdal.Unlink(raster1)
    gdal.Unlink(raster2)

def test_rasters_intersect_false():
    """ Test: Rasters intersect. False Case """
    raster1 = create_sample_raster(height=10, width=10, pixel_height=1, pixel_width=1)
    raster2 = create_sample_raster(height=10, width=10, pixel_height=1, pixel_width=1, x_min=15, y_max=25)

    assert rasters_intersect(raster1, raster2) is False

    gdal.Unlink(raster1)
    gdal.Unlink(raster2)

def test_rasters_intersect_same():
    """ Test: Rasters intersect. Same Case """
    raster1 = create_sample_raster(width=10, height=10, x_min=0, y_max=10)
    raster2 = create_sample_raster(width=10, height=10, x_min=0, y_max=10)

    assert rasters_intersect(raster1, raster2) is True

    gdal.Unlink(raster1)
    gdal.Unlink(raster2)

def test_rasters_intersect_different_pixelsizes():
    """ Test: Rasters intersect. Different pixel sizes """
    raster1 = create_sample_raster(width=10, height=10, pixel_width=1, pixel_height=1, x_min=0, y_max=10)
    raster2 = create_sample_raster(width=20, height=20, pixel_width=0.5, pixel_height=0.5, x_min=5, y_max=15)

    assert rasters_intersect(raster1, raster2) is True

    gdal.Unlink(raster1)
    gdal.Unlink(raster2)

def rasters_intersect_non_intersecting_difpixel():
    """ Test: Rasters intersect. Different pixel sizes, non-intersecting """
    raster1 = create_sample_raster(width=10, height=10, pixel_width=1, pixel_height=1, x_min=0, y_max=10)
    raster2 = create_sample_raster(width=40, height=40, pixel_width=0.5, pixel_height=0.5, x_min=20, y_max=30)

    assert rasters_intersect(raster1, raster2) is False

    gdal.Unlink(raster1)
    gdal.Unlink(raster2)

def test_rasters_intersection_true():
    """ Test: Rasters intersection. True Case """ 
    raster1 = create_sample_raster(width=10, height=10, x_min=0, y_max=10)
    raster2 = create_sample_raster(width=10, height=10, x_min=5, y_max=15)
    intersection1 = rasters_intersection(raster1, raster2)

    assert intersection1 is not None

    gdal.Unlink(raster1)
    gdal.Unlink(raster2)

def test_rasters_intersection_false():
    """ Test: Rasters intersection. False Case """
    raster1 = create_sample_raster(width=10, height=10, x_min=0, y_max=10)
    raster2 = create_sample_raster(width=10, height=10, x_min=20, y_max=30)

    with pytest.raises(ValueError):
        rasters_intersection(raster1, raster2)

    gdal.Unlink(raster1)
    gdal.Unlink(raster2)

def test_get_overlap_fraction():
    """ Test: Get overlap fraction. Complete overlap. """
    raster1 = create_sample_raster(width=10, height=10, x_min=0, y_max=10)
    raster2 = create_sample_raster(width=10, height=10, x_min=0, y_max=10)
    overlap1 = get_overlap_fraction(raster1, raster2)

    assert overlap1 == 1.0

    gdal.Unlink(raster1)
    gdal.Unlink(raster2)

def test_get_overlap_fraction_partial():
    """ Test: Get overlap fraction. Partial overlap."""
    raster1 = create_sample_raster(width=10, height=10, x_min=0, y_max=10)
    raster2 = create_sample_raster(width=10, height=10, x_min=5, y_max=15)
    overlap = get_overlap_fraction(raster1, raster2)

    assert 0 < overlap < 1

    gdal.Unlink(raster1)
    gdal.Unlink(raster2)

def test_get_overlap_fraction_no_overlap():
    """ Test: Get overlap fraction. No overlap."""
    raster1 = create_sample_raster(width=10, height=10, x_min=0, y_max=10)
    raster2 = create_sample_raster(width=10, height=10, x_min=20, y_max=30)
    overlap = get_overlap_fraction(raster1, raster2)

    assert overlap == 0

    gdal.Unlink(raster1)
    gdal.Unlink(raster2)

def test_create_raster_from_array_default():
    """ Test: Create raster from array with default arguments. """
    arr = np.random.rand(10, 10).astype(np.float32)
    output_name = create_raster_from_array(arr)

    # Check created raster properties
    ds = gdal.Open(output_name)
    assert ds.RasterXSize == 10
    assert ds.RasterYSize == 10
    assert ds.RasterCount == 1
    assert ds.GetProjection() == gdal_utils.parse_projection("EPSG:3857", return_wkt=True)
    assert ds.GetGeoTransform() == (0.0, 10.0, 0.0, 0.0, 0.0, -10.0)

    gdal.Unlink(output_name)

def test_create_raster_from_array_special():
    """ Test: Create raster from array with special arguments. """
    arr = np.random.rand(10, 12, 3).astype(np.float32) # (height, width, bands)
    output_name = create_raster_from_array(
        arr,
        pixel_size=0.5,
        x_min=10,
        projection="EPSG:4326",
    )

    # Check created raster properties
    ds = gdal.Open(output_name)
    assert ds.RasterXSize == 12
    assert ds.RasterYSize == 10
    assert ds.RasterCount == 3
    assert ds.GetProjection() == gdal_utils.parse_projection("EPSG:4326", return_wkt=True)

    gt = ds.GetGeoTransform()

    assert gt[0] == 10.0 # x_min
    assert gt[1] == 0.5 # width_res
    assert gt[5] == -0.5 # height_res

    gdal.Unlink(output_name)

def test_create_raster_from_array_invalid_input():
    """ Test: Create raster from array with an invalid input array. """
    invalid_arr = np.random.rand(10, 10, 3, 3)

    with pytest.raises(AssertionError, match="Array must be 2 or 3 dimensional"):
        create_raster_from_array(invalid_arr)

def test_create_grid_with_coordinates():
    """ Test: Create grid with coordinates. """
    arr = np.random.rand(10, 10, 1)
    raster_file = create_raster_from_array(arr, pixel_size=[2.0, 2.0], x_min=10.0, y_max=20.0)
    raster = gdal.Open(raster_file)

    # Create grid with coordinates
    grid = create_grid_with_coordinates(raster)

    # Check grid properties
    assert grid.shape == (10, 10, 2)

    meta = raster_to_metadata(raster)

    assert (meta["x_max"] + meta["x_min"]) / 2 == 20.0
    assert (meta["y_max"] + meta["y_min"]) / 2 == 10.0
