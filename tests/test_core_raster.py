""" Tests for core_raster.py """


# Standard library
import sys; sys.path.append("../")
import os
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
    array_to_raster,
    raster_set_datatype,
)
from buteo.utils import gdal_enums

# Setup tests
FOLDER = "geometry_and_rasters/"
s2_b04 = os.path.abspath(os.path.join(FOLDER, "s2_b04.jp2"))
s2_b04_subset = os.path.abspath(os.path.join(FOLDER, "s2_b04_beirut_misaligned.tif"))
s2_b04_faraway = os.path.abspath(os.path.join(FOLDER, "s2_b04_baalbeck.tif"))
s2_rgb = os.path.abspath(os.path.join(FOLDER, "s2_tci.jp2"))

vector_file = os.path.abspath(os.path.join(FOLDER, "beirut_city_utm36.gpkg"))


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
        epsg_code=4326,
        datatype=gdal.GDT_Byte,
        nodata=None,
    ):
    """ Create a sample raster file for testing purposes. """
    raster_path = f"/vsimem/mem_raster_{uuid4().int}.tif"
    mem_drv = gdal.GetDriverByName("GTiff")
    raster = mem_drv.Create(raster_path, width, height, bands, datatype)
    raster.SetGeoTransform((0, pixel_width, 0, height * pixel_height, 0, -pixel_height))

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

# def test_raster_set_datatype_single():
#     """Test: Change the datatype of a single raster."""
#     reference = create_sample_raster(
#         height=150, width=100, bands=3, datatype=gdal.GDT_Float32
#     )

#     new_dtype = "UInt16"
#     output_path = raster_set_datatype(reference, new_dtype, overwrite=True)

#     metadata = raster_to_metadata(output_path)

#     assert metadata["dtype"] == gdal_enums.translate_gdal_dtype_to_str(gdal.GDT_UInt16)

#     gdal.Unlink(output_path)
#     gdal.Unlink(reference)


# def test_raster_set_datatype_list():
#     """Test: Change the datatype of multiple rasters."""
#     reference1 = create_sample_raster(
#         height=150, width=100, bands=3, datatype=gdal.GDT_Float32
#     )
#     reference2 = create_sample_raster(
#         height=200, width=150, bands=1, datatype=gdal.GDT_Float32
#     )

#     new_dtype = "UInt16"
#     raster_list = [reference1, reference2]
#     output_paths = raster_set_datatype(raster_list, new_dtype, overwrite=True)

#     for path in output_paths:
#         metadata = raster_to_metadata(path)
#         assert metadata["dtype"] == gdal_enums.translate_gdal_dtype_to_str(gdal.GDT_UInt16)
#         gdal.Unlink(path)

#     gdal.Unlink(reference1)
#     gdal.Unlink(reference2)
