""" Tests for raster/convolution.py """
# pylint: disable=missing-function-docstring

# Standard library
import sys; sys.path.append("../")
import os

# External
from osgeo import gdal
import numpy as np
import pytest

# Internal
from utils_tests import create_sample_raster, create_sample_vector
from buteo.raster import core_raster_io


def test_raster_to_array_shape_dtype():
    raster_path = create_sample_raster(width=10, height=10, bands=1)
    array = core_raster_io.raster_to_array(raster_path)

    assert array.shape == (10, 10, 1)
    assert array.dtype == np.uint8

def test_raster_to_array_multiple_bands():
    raster_path = create_sample_raster(width=10, height=10, bands=3)
    array = core_raster_io.raster_to_array(raster_path, bands='all')

    assert array.shape == (10, 10, 3)

def test_raster_to_array_invalid_pixel_offsets():
    raster_path = create_sample_raster(width=10, height=10, bands=1)
    with pytest.raises(ValueError):
        core_raster_io.raster_to_array(raster_path, pixel_offsets=(-1, 0, 10, 10))

def test_raster_to_array_invalid_extent():
    raster_path = create_sample_raster(width=10, height=10, bands=1)
    with pytest.raises(ValueError):
        core_raster_io.raster_to_array(raster_path, bbox=[-1, 0, 10, 11])

def test_raster_to_array_filled():
    raster_path = create_sample_raster(width=10, height=10, bands=1, nodata=255)
    array_filled = core_raster_io.raster_to_array(raster_path, filled=True)
    array_not_filled = core_raster_io.raster_to_array(raster_path, filled=False)

    assert np.ma.isMaskedArray(array_not_filled)
    assert not np.ma.isMaskedArray(array_filled)
    assert array_filled.dtype == np.uint8
    assert array_not_filled.dtype == np.uint8

def test_raster_to_array_custom_fill_value():
    raster_path = create_sample_raster(width=10, height=10, bands=1, nodata=255)
    array = core_raster_io.raster_to_array(raster_path, filled=True, fill_value=128)

    assert not np.ma.isMaskedArray(array)
    assert array.dtype == np.uint8

def test_raster_to_array_cast_dtype():
    raster_path = create_sample_raster(width=10, height=10, bands=1)
    array = core_raster_io.raster_to_array(raster_path, cast=np.int16)

    assert array.shape == (10, 10, 1)
    assert array.dtype == np.int16

def test_raster_to_array_channel_last():
    raster_path = create_sample_raster(width=10, height=10, bands=3)
    array_channel_last = core_raster_io.raster_to_array(raster_path, bands='all', channel_last=True)
    array_channel_first = core_raster_io.raster_to_array(raster_path, bands='all', channel_last=False)

    assert array_channel_last.shape == (10, 10, 3)
    assert array_channel_first.shape == (3, 10, 10)

def test_raster_to_array_multiple_rasters():
    raster_path1 = create_sample_raster(width=10, height=10, bands=2)
    raster_path2 = create_sample_raster(width=10, height=10, bands=3)
    array = core_raster_io.raster_to_array([raster_path1, raster_path2], bands='all')

    assert array.shape == (10, 10, 5)

def test_array_to_raster_2D():
    reference_path = create_sample_raster(10, 10)
    array = np.random.rand(10, 10)

    result_path = core_raster_io.array_to_raster(array, reference=reference_path)

    result_raster = gdal.Open(result_path)
    assert result_raster is not None

    result_array = result_raster.ReadAsArray()
    assert np.array_equal(result_array, array)

def test_array_to_raster_3D_channel_last():
    reference_path = create_sample_raster(10, 10)
    array = np.random.rand(10, 10, 3)

    result_path = core_raster_io.array_to_raster(array, reference=reference_path, channel_last=True)

    result_raster = gdal.Open(result_path)
    assert result_raster is not None

    result_array = np.dstack([result_raster.GetRasterBand(i + 1).ReadAsArray() for i in range(3)])
    assert np.array_equal(result_array, array)

def test_array_to_raster_3D_channel_first():
    reference_path = create_sample_raster(10, 10)
    array = np.random.rand(3, 10, 10)

    result_path = core_raster_io.array_to_raster(array, reference=reference_path, channel_last=False)

    result_raster = gdal.Open(result_path)
    assert result_raster is not None

    result_array = np.dstack([result_raster.GetRasterBand(i + 1).ReadAsArray() for i in range(3)])
    expected_array = np.transpose(array, (1, 2, 0))
    assert np.array_equal(result_array, expected_array)

def test_array_to_raster_nodata_arr():
    reference_path = create_sample_raster(10, 10)
    array = np.random.rand(10, 10)
    array[array < 0.5] = np.nan

    result_path = core_raster_io.array_to_raster(array, reference=reference_path, set_nodata="arr")
    result_raster = gdal.Open(result_path)
    assert result_raster is not None

    result_array = result_raster.ReadAsArray()
    assert np.allclose(result_array, array, equal_nan=True)

def test_array_to_raster_nodata_ref():
    reference_path = create_sample_raster(10, 10, nodata=-9999)
    array = np.random.rand(10, 10)

    result_path = core_raster_io.array_to_raster(array, reference=reference_path, set_nodata="ref")
    result_raster = gdal.Open(result_path)
    assert result_raster is not None

    result_nodata_value = result_raster.GetRasterBand(1).GetNoDataValue()
    reference_raster = gdal.Open(reference_path)
    reference_nodata_value = reference_raster.GetRasterBand(1).GetNoDataValue()
    assert result_nodata_value == reference_nodata_value

def test_array_to_raster_bounding_box():
    reference_path = create_sample_raster(10, 10)
    array = np.random.rand(5, 5)

    bbox = [5, 10, 5, 10] # [xmin, ymin, xmax, ymax]

    result_path = core_raster_io.array_to_raster(array, reference=reference_path, bbox=bbox)
    result_raster = gdal.Open(result_path)
    assert result_raster is not None

    result_array = result_raster.ReadAsArray()
    assert np.array_equal(result_array, array)

def test_raster_to_array_and_array_to_raster_identity():
    reference_path = create_sample_raster(10, 10)

    original_array = core_raster_io.raster_to_array(reference_path)
    result_path = core_raster_io.array_to_raster(original_array, reference=reference_path)
    result_array = core_raster_io.raster_to_array(result_path)

    assert np.array_equal(original_array, result_array)

def test_raster_to_array_and_array_to_raster_modified():
    reference_path = create_sample_raster(10, 10)

    original_array = core_raster_io.raster_to_array(reference_path)
    modified_array = original_array * 2
    result_path = core_raster_io.array_to_raster(modified_array, reference=reference_path)
    result_array = core_raster_io.raster_to_array(result_path)

    assert np.array_equal(modified_array, result_array)

def test_raster_to_array_and_array_to_raster_custom_nodata():
    reference_path = create_sample_raster(10, 10)

    original_array = core_raster_io.raster_to_array(reference_path)
    modified_array = np.where(original_array < 0.5, -9999, original_array)
    result_path = core_raster_io.array_to_raster(modified_array, reference=reference_path, set_nodata=-9999)
    result_array = core_raster_io.raster_to_array(result_path)

    assert np.array_equal(modified_array, result_array)

def test_raster_to_array_and_array_to_raster_bbox():
    reference_path = create_sample_raster(10, 10)

    original_array = core_raster_io.raster_to_array(reference_path)
    sub_array = original_array[2:7, 2:7]

    bbox = [5, 10, 5, 10] # [xmin, ymin, xmax, ymax]

    result_path = core_raster_io.array_to_raster(sub_array, reference=reference_path, bbox=bbox)
    result_array = core_raster_io.raster_to_array(result_path)

    assert np.array_equal(sub_array, result_array)

def test_raster_to_array_and_array_to_raster_pixel_offsets_identity():
    reference_path = create_sample_raster(10, 10)

    original_array = core_raster_io.raster_to_array(reference_path, pixel_offsets=[2, 2, 6, 6])
    result_path = core_raster_io.array_to_raster(original_array, reference=reference_path, pixel_offsets=[2, 2, 6, 6])
    result_array = core_raster_io.raster_to_array(result_path)

    assert np.array_equal(original_array, result_array)

def test_raster_to_array_and_array_to_raster_pixel_offsets_modified():
    reference_path = create_sample_raster(10, 10)

    original_array = core_raster_io.raster_to_array(reference_path, pixel_offsets=[2, 2, 6, 6])
    modified_array = original_array * 2
    result_path = core_raster_io.array_to_raster(modified_array, reference=reference_path, pixel_offsets=[2, 2, 6, 6])
    result_array = core_raster_io.raster_to_array(result_path)

    assert np.array_equal(modified_array, result_array)

def test_raster_to_array_and_array_to_raster_pixel_offsets_custom_nodata():
    reference_path = create_sample_raster(10, 10)

    original_array = core_raster_io.raster_to_array(reference_path, pixel_offsets=[2, 2, 6, 6])
    modified_array = np.where(original_array < 0.5, -9999, original_array)
    result_path = core_raster_io.array_to_raster(modified_array, reference=reference_path, set_nodata=-9999, pixel_offsets=[2, 2, 6, 6])
    result_array = core_raster_io.raster_to_array(result_path)

    assert np.array_equal(modified_array, result_array)

def test_raster_to_array_and_array_to_raster_pixel_offsets_overlapping():
    reference_path = create_sample_raster(10, 10)

    original_array = core_raster_io.raster_to_array(reference_path, pixel_offsets=[4, 4, 6, 6])
    result_path = core_raster_io.array_to_raster(original_array, reference=reference_path, pixel_offsets=[4, 4, 6, 6])
    result_array = core_raster_io.raster_to_array(result_path)

    assert np.array_equal(original_array, result_array)

def test_save_raster_to_disk():
    in_path = create_sample_raster(10, 10)
    out_path = "./test_raster.tif"

    core_raster_io.save_dataset_to_disk(in_path, out_path)
    assert os.path.exists(out_path)
    os.remove(out_path)

def test_save_vector_to_disk():
    in_path = create_sample_vector()
    out_path = "./test_vector.gkpg"

    core_raster_io.save_dataset_to_disk(in_path, out_path)
    assert os.path.exists(out_path)
    os.remove(out_path)

def test_save_dataset_to_disk_with_prefix():
    in_path = create_sample_raster(10, 10)
    out_path = "./test_raster.tif"
    prefix = "test_prefix_"

    output_path = core_raster_io.save_dataset_to_disk(in_path, out_path, prefix=prefix)

    assert os.path.exists(output_path)
    assert os.path.basename(output_path).startswith(prefix)
    os.remove(output_path)

def test_save_dataset_to_disk_with_suffix():
    in_path = create_sample_raster(10, 10)
    out_path = "./test_raster.tif"
    suffix = "_test_suffix"

    output_path = core_raster_io.save_dataset_to_disk(in_path, out_path, suffix=suffix)
    assert os.path.exists(output_path)
    assert output_path.endswith(suffix + ".tif")
    os.remove(output_path)

def test_save_dataset_to_disk_no_overwrite():
    in_path = create_sample_raster(10, 10)
    out_path = "./test_raster.tif"

    core_raster_io.save_dataset_to_disk(in_path, out_path)
    core_raster_io.save_dataset_to_disk(in_path, out_path, overwrite=False)
    assert os.path.exists(out_path)
    os.remove(out_path)

@pytest.fixture(scope="function")
def sample_raster():
    raster_path = create_sample_raster(width=100, height=100, bands=3)
    yield raster_path
    gdal.Unlink(raster_path)

def test_single_chunk(sample_raster):
    chunk_gen = core_raster_io.raster_to_array_chunks(sample_raster, chunks=1)
    chunk, offsets = next(chunk_gen)

    assert offsets == (0, 0, 100, 100)
    assert chunk.shape == (100, 100, 3)

    with pytest.raises(StopIteration):
        next(chunk_gen)

def test_multiple_chunks(sample_raster):
    chunk_gen = core_raster_io.raster_to_array_chunks(sample_raster, chunks=4)

    expected_offsets = [
        (0, 0, 50, 50),
        (0, 50, 50, 50),
        (50, 0, 50, 50),
        (50, 50, 50, 50),
    ]

    for chunk, offsets in chunk_gen:
        assert offsets in expected_offsets
        assert chunk.shape == (50, 50, 3)

def test_overlap(sample_raster):
    chunk_gen = core_raster_io.raster_to_array_chunks(sample_raster, chunks=4, overlap=10)

    expected_offsets = [
        (0, 0, 55, 55),
        (45, 0, 55, 55),
        (0, 45, 55, 55),
        (45, 45, 55, 55),
    ]

    for i, (chunk, offsets) in enumerate(chunk_gen):
        assert offsets in expected_offsets
        assert chunk.shape == (55, 55, 3)

def test_filled_and_fill_value(sample_raster):
    chunk_gen = core_raster_io.raster_to_array_chunks(
        sample_raster,
        chunks=1,
        filled=True,
        fill_value=0,
    )

    chunk, offsets = next(chunk_gen)

    assert offsets == (0, 0, 100, 100)
    assert chunk.shape == (100, 100, 3)
    assert not np.isnan(chunk).any()

def test_numpy_to_raster(sample_raster):
    array = np.random.rand(100, 100, 3)
    array[:, :, 0] = 1
    array[:, :, 1] = 2
    array[:, :, 2] = 3

    raster_result = core_raster_io.array_to_raster(array, reference=sample_raster)

    array_all = core_raster_io.raster_to_array(raster_result, bands=-1)
    assert np.array_equal(array_all, array)

    array_first = core_raster_io.raster_to_array(raster_result, bands=[1])
    assert np.array_equal(array_first, array[:, :, 0:1])

    array_last = core_raster_io.raster_to_array(raster_result, bands=[3])
    assert np.array_equal(array_last, array[:, :, 2:3])

    array_flip = core_raster_io.raster_to_array(raster_result, bands=[3, 2, 1])
    assert np.array_equal(array_flip, array[:, :, ::-1])


def test_create_raster_from_array_default():
    """ Test: Create raster from array with default arguments. """
    arr = np.random.rand(10, 10).astype(np.float32)
    output_name = core_raster_io.raster_create_from_array(arr)

    # Check created raster properties
    ds = gdal.Open(output_name)
    assert ds.RasterXSize == 10
    assert ds.RasterYSize == 10
    assert ds.RasterCount == 1
    assert ds.GetGeoTransform() == (0.0, 1.0, 0.0, 0.0, 0.0, -1.0)

    gdal.Unlink(output_name)

def test_create_raster_from_array_special():
    """ Test: Create raster from array with special arguments. """
    arr = np.random.rand(10, 12, 3).astype(np.float32) # (height, width, bands)
    output_name = core_raster_io.raster_create_from_array(
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

    gt = ds.GetGeoTransform()

    assert gt[0] == 10.0 # x_min
    assert gt[1] == 0.5 # width_res
    assert gt[5] == -0.5 # height_res

    gdal.Unlink(output_name)

def test_create_raster_from_array_invalid_input():
    """ Test: Create raster from array with an invalid input array. """
    invalid_arr = np.random.rand(10, 10, 3, 3)

    with pytest.raises(AssertionError, match="Array must be 2 or 3 dimensional"):
        core_raster_io.raster_create_from_array(invalid_arr)


def test_bands_raster_to_array():
    """ Test: Raster to array with bands specified. """
    raster_1 = create_sample_raster(width=10, height=10, bands=3)
    raster_2 = create_sample_raster(width=10, height=10, bands=1)

    arr = core_raster_io.raster_to_array([raster_1, raster_2])

    assert arr.shape == (10, 10, 4)

    arr_1 = core_raster_io.raster_to_array(raster_1)
    arr_2 = core_raster_io.raster_to_array(raster_2)

    assert np.array_equal(arr[:, :, 0:3], arr_1)
    assert np.array_equal(arr[:, :, 3:4], arr_2) # <-- all 0.0


def test_raster_to_array_chunks():
    # Create a sample raster
    raster_path = create_sample_raster(width=12, height=12, bands=1)

    # Test case 1: Perfectly divisible image with border_strategy = 1
    chunk_size = (3, 3)
    border_strategy = 1
    channel_last = True
    expected_offsets = [
        (0, 0, 3, 3), (3, 0, 3, 3), (6, 0, 3, 3), (9, 0, 3, 3),
        (0, 3, 3, 3), (3, 3, 3, 3), (6, 3, 3, 3), (9, 3, 3, 3),
        (0, 6, 3, 3), (3, 6, 3, 3), (6, 6, 3, 3), (9, 6, 3, 3),
        (0, 9, 3, 3), (3, 9, 3, 3), (6, 9, 3, 3), (9, 9, 3, 3),
    ]

    rtac = core_raster_io.raster_to_array_chunks(raster_path, chunk_size=chunk_size, border_strategy=border_strategy, channel_last=channel_last)
    used = []
    for chunk, offset in rtac:
        assert offset in expected_offsets and offset not in used, f"Expected {offset} to be in {expected_offsets} and not in {used}"
        assert chunk.shape == (chunk_size[1], chunk_size[0], 1), f"Expected chunk shape {(chunk_size[1], chunk_size[0], 1)}, but got {chunk.shape}"
        used.append(offset)

    # Test case 2: Image with extra pixels on the border, border_strategy = 1
    chunk_size = (4, 4)
    border_strategy = 1
    channel_last = True
    expected_offsets = [
        (0, 0, 4, 4), (4, 0, 4, 4), (8, 0, 4, 4),
        (0, 4, 4, 4), (4, 4, 4, 4), (8, 4, 4, 4),
        (0, 8, 4, 4), (4, 8, 4, 4), (8, 8, 4, 4),
    ]

    rtac = core_raster_io.raster_to_array_chunks(raster_path, chunk_size=chunk_size, border_strategy=border_strategy, channel_last=channel_last)
    used = []
    for chunk, offset in rtac:
        assert offset in expected_offsets and offset not in used, f"Expected {offset} to be in {expected_offsets} and not in {used}"
        assert chunk.shape == (chunk_size[1], chunk_size[0], 1), f"Expected chunk shape {(chunk_size[1], chunk_size[0], 1)}, but got {chunk.shape}"
        used.append(offset)

    # Test case 3: Image with extra pixels on the border, border_strategy = 2
    chunk_size = (4, 4)
    border_strategy = 2
    channel_last = True
    expected_offsets = [
        (0, 0, 4, 4), (4, 0, 4, 4), (8, 0, 4, 4),
        (0, 4, 4, 4), (4, 4, 4, 4), (8, 4, 4, 4),
        (0, 8, 4, 4), (4, 8, 4, 4), (8, 8, 4, 4),
    ]

    rtac = core_raster_io.raster_to_array_chunks(raster_path, chunk_size=chunk_size, border_strategy=border_strategy, channel_last=channel_last)
    used = []
    for chunk, offset in rtac:
        assert offset in expected_offsets and offset not in used, f"Expected {offset} to be in {expected_offsets} and not in {used}"
        assert chunk.shape == (chunk_size[1], chunk_size[0], 1), f"Expected chunk shape {(chunk_size[1], chunk_size[0], 1)}, but got {chunk.shape}"
        used.append(offset)

    # Test case 4: Test with overlap
    chunk_size = (4, 4)
    overlap = 2
    border_strategy = 1
    channel_last = True
    expected_offsets = [
        (2, 2, 4, 4), (6, 2, 4, 4),
        (2, 6, 4, 4), (6, 6, 4, 4),
        (4, 0, 4, 4), (8, 0, 4, 4),
        (0, 4, 4, 4), (0, 8, 4, 4),
    ]

    rtac = core_raster_io.raster_to_array_chunks(raster_path, chunk_size=chunk_size, overlap=overlap, border_strategy=border_strategy, channel_last=channel_last)
    used = []
    for chunk, offset in rtac:
        assert offset in expected_offsets and offset not in used, f"Expected {offset} to be in {expected_offsets} and not in {used}"
        assert chunk.shape == (chunk_size[1], chunk_size[0], 1), f"Expected chunk shape {(chunk_size[1], chunk_size[0], 1)}, but got {chunk.shape}"
        used.append(offset)

    # Test case 5: Test with overlap
    chunk_size = (4, 4)
    overlap = 2
    border_strategy = 2
    channel_last = True
    expected_offsets = [
        (2, 2, 4, 4), (6, 2, 4, 4), (8, 2, 4, 4),
        (2, 6, 4, 4), (6, 6, 4, 4), (8, 6, 4, 4),
        (2, 8, 4, 4), (6, 8, 4, 4), (8, 8, 4, 4),
        (0, 0, 4, 4), (4, 0, 4, 4), (8, 0, 4, 4),
        (0, 4, 4, 4), (0, 8, 4, 4),
    ]

    rtac = core_raster_io.raster_to_array_chunks(raster_path, chunk_size=chunk_size, overlap=overlap, border_strategy=border_strategy, channel_last=channel_last)
    used = []
    for chunk, offset in rtac:
        assert offset in expected_offsets and offset not in used, f"Expected {offset} to be in {expected_offsets} and not in {used}"
        assert chunk.shape == (chunk_size[1], chunk_size[0], 1), f"Expected chunk shape {(chunk_size[1], chunk_size[0], 1)}, but got {chunk.shape}"
        used.append(offset)

    # Test case 6: Test with channel_last=False
    chunk_size = (4, 4)
    overlap = 2
    border_strategy = 1
    channel_last = False
    expected_offsets = [
        (2, 2, 4, 4), (6, 2, 4, 4),
        (2, 6, 4, 4), (6, 6, 4, 4),
        (4, 0, 4, 4), (8, 0, 4, 4),
        (0, 4, 4, 4), (0, 8, 4, 4),
    ]
    rtac = core_raster_io.raster_to_array_chunks(raster_path, chunk_size=chunk_size, overlap=overlap, border_strategy=border_strategy, channel_last=channel_last)
    used = []
    for chunk, offset in rtac:
        assert offset in expected_offsets and offset not in used, f"Expected {offset} to be in {expected_offsets} and not in {used}"
        assert chunk.shape == (1, chunk_size[1], chunk_size[0]), f"Expected chunk shape {(1, chunk_size[1], chunk_size[0])}, but got {chunk.shape}"
        used.append(offset)

    # Cleanup
    gdal.Unlink(raster_path)
