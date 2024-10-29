""" Tests for core_raster.py """


# Standard library
import sys; sys.path.append("../")

# External
import numpy as np
import pytest

# Internal
from buteo.array.patches import (
    _get_kernel_weights,
    _merge_weighted_median,
    _merge_weighted_average,
    _unique_values,
    _merge_weighted_mad,
    _merge_weighted_mode,
    _borders_are_necessary,
    _array_to_patches_single,
    _patches_to_array_single,
    array_to_patches,
    predict_array,
    predict_array_pixel,
)


def test_get_kernel_weights_default_values():
    """ Test that the default values for edge_distance and tile_size are correct. """
    edge_distance = 5
    tile_size = 64

    kernel = _get_kernel_weights(edge_distance=edge_distance, tile_size=tile_size)

    assert kernel.shape == (64, 64)
    assert kernel.dtype == np.float32
    assert np.all(kernel[edge_distance:-edge_distance, edge_distance:-edge_distance] > 0)
    assert np.all(kernel[:edge_distance, :] > 0)
    assert np.all(kernel[-edge_distance:, :] > 0)
    assert np.all(kernel[:, :edge_distance] > 0)
    assert np.all(kernel[:, -edge_distance:] > 0)

def test_get_kernel_weights_custom_values():
    """ Test that the custom values for edge_distance and tile_size are correct. """
    edge_distance = 3
    tile_size = 32

    kernel = _get_kernel_weights(tile_size=tile_size, edge_distance=edge_distance)

    assert kernel.shape == (tile_size, tile_size)
    assert kernel.dtype == np.float32
    assert np.all(kernel[edge_distance:-edge_distance, edge_distance:-edge_distance] > 0)
    assert np.all(kernel[:edge_distance, :] > 0)
    assert np.all(kernel[-edge_distance:, :] > 0)
    assert np.all(kernel[:, :edge_distance] > 0)
    assert np.all(kernel[:, -edge_distance:] > 0)

dummy_arr = np.array([
    [
        [[1, 2], [4, 5]],
        [[1, 2], [4, 5]],
    ],
    [
        [[3, 4], [6, 7]],
        [[3, 4], [6, 7]],
    ]
], dtype="float32")

dummy_arr_weight = np.ones_like(dummy_arr)

def test_merge_weighted_median_basic():
    """ Test the basic functionality of the merge_weighted_median function. """
    merged_arr = _merge_weighted_median(dummy_arr, dummy_arr_weight, np.zeros((dummy_arr.shape[0], dummy_arr.shape[1], 2)))

    expected_arr = np.array([
        [[2, 3], [5, 6]],
        [[2, 3], [5, 6]]
    ], dtype="float32")

    assert np.allclose(merged_arr, expected_arr)

def test_merge_weighted_median_different_weights():
    """ Test the merge_weighted_median function with different weights. """
    arr_weight = np.array([
        [
            [[1, 1], [1, 1]],
            [[1, 1], [1, 1]],
        ],
        [
            [[2, 2], [2, 2]],
            [[2, 2], [2, 2]],
        ]
    ], dtype="float32")

    merged_arr = _merge_weighted_median(dummy_arr, arr_weight, np.zeros((dummy_arr.shape[0], dummy_arr.shape[1], 2)))

    expected_arr = np.array([
        [[2.333333, 3.3333333], [5.3333333, 6.333333]],
        [[2.333333, 3.3333333], [5.3333333, 6.333333]]
    ], dtype="float32")

    assert np.allclose(merged_arr, expected_arr)

def test_merge_weighted_average_basic():
    """ Test the basic functionality of the merge_weighted_average function. """
    merged_arr = _merge_weighted_average(dummy_arr, dummy_arr_weight, np.zeros((dummy_arr.shape[0], dummy_arr.shape[1], 2)))

    expected_arr = np.array([
        [[2, 3], [5, 6]],
        [[2, 3], [5, 6]]
    ], dtype="float32")

    assert np.allclose(merged_arr, expected_arr)

def test_merge_weighted_average_different_weights():
    """ Test the merge_weighted_average function with different weights. """
    arr_weight = np.array([
        [
            [[1, 1], [1, 1]],
            [[1, 1], [1, 1]],
        ],
        [
            [[2, 2], [2, 2]],
            [[2, 2], [2, 2]],
        ]
    ], dtype="float32")

    merged_arr = _merge_weighted_average(dummy_arr, arr_weight, np.zeros((dummy_arr.shape[0], dummy_arr.shape[1], 2)))

    expected_arr = np.array([
        [[2.333333, 3.3333333], [5.3333333, 6.333333]],
        [[2.333333, 3.3333333], [5.3333333, 6.333333]]
    ], dtype="float32")

    assert np.allclose(merged_arr, expected_arr)

def test_merge_weighted_average_with_nan():
    """ Test the merge_weighted_average function with NaN values. """
    arr_with_nan = dummy_arr.copy()
    arr_with_nan[0, 0, 0, 0] = np.nan
    arr_with_nan[1, 0, 1, 1] = np.nan

    merged_arr = _merge_weighted_average(arr_with_nan, dummy_arr_weight, np.zeros((dummy_arr.shape[0], dummy_arr.shape[1], 2)))

    expected_arr = np.array([
        [[3, 3], [5, 5]],
        [[2, 3], [5, 6]]
    ], dtype="float32")

    assert np.allclose(merged_arr, expected_arr)

def test_merge_weighted_mad_basic():
    """ Test the basic functionality of the merge_weighted_mad function. """
    merged_arr = _merge_weighted_mad(dummy_arr, dummy_arr_weight, np.zeros((dummy_arr.shape[0], dummy_arr.shape[1], 2)))

    expected_arr = np.array([
        [[2, 3], [5, 6]],
        [[2, 3], [5, 6]]
    ], dtype="float32")

    assert np.allclose(merged_arr, expected_arr)

def test_unique_values_basic():
    """ Test the basic functionality of the unique_values function. """
    arr = np.array([1, 2, 3, 2, 1, 4, 5, 3, 6], dtype="float32")
    unique_arr = _unique_values(arr)

    expected_arr = np.array([1, 2, 3, 4, 5, 6], dtype="float32")

    assert np.allclose(unique_arr, expected_arr)

def test_merge_weighted_mode_basic():
    """ Test the basic functionality of the merge_weighted_mode function. """
    arr = np.array([
        [
            [[1, 2], [2, 3]],
            [[1, 2], [2, 3]],
        ],
        [
            [[2, 3], [3, 4]],
            [[2, 3], [3, 4]],
        ]
    ], dtype="uint8")

    arr_weight = np.ones_like(arr)

    merged_arr = _merge_weighted_mode(arr, arr_weight, np.zeros((dummy_arr.shape[0], dummy_arr.shape[1], 2)))

    expected_arr = np.array([
        [[1, 2], [2, 3]],
        [[1, 2], [2, 3]]
    ], dtype="uint8")

    assert np.allclose(merged_arr, expected_arr)

def test_borders_are_necessary_basic():
    """ Test that borders are correctly. (basic) """
    arr = np.zeros((64, 64))
    tile_size = 32
    offset = [0, 0]
    expected_output = (False, False)
    result = _borders_are_necessary(arr, tile_size, offset)
    assert result == expected_output

def test_borders_are_necessary_border_needed():
    """ Test that borders are correctly. (border needed) """
    arr = np.zeros((100, 100))
    tile_size = 32
    offset = [0, 0]
    expected_output = (True, True)
    result = _borders_are_necessary(arr, tile_size, offset)
    assert result == expected_output

def test_borders_are_necessary_3d_array():
    """ Test that borders are correctly. (3d array) """
    arr = np.zeros((64, 64, 3))
    tile_size = 32
    offset = [0, 0]
    expected_output = (False, False)
    result = _borders_are_necessary(arr, tile_size, offset)
    assert result == expected_output

def test_borders_are_necessary_with_offset():
    """ Test that borders are correctly. (with offset) """
    arr = np.zeros((64, 64))
    tile_size = 32
    offset = [32, 32]
    expected_output = (False, False)
    result = _borders_are_necessary(arr, tile_size, offset)
    assert result == expected_output

def test_borders_are_necessary_offset_and_border_needed():
    """ Test that borders are correctly. (offset and border needed) """
    arr = np.zeros((64, 64))
    tile_size = 32
    offset = [16, 16]
    expected_output = (True, True)
    result = _borders_are_necessary(arr, tile_size, offset)
    assert result == expected_output

def test_array_to_patches_single_basic():
    """ Test that the array is correctly converted to patches. (basic) """
    arr = np.random.rand(64, 64, 3)
    tile_size = 32
    patches = _array_to_patches_single(arr, tile_size)

    assert patches.shape == (4, 32, 32, 3)

def test_array_to_patches_single_with_offset():
    """ Test that the array is correctly converted to patches. (with offset) """
    arr = np.random.rand(128, 128, 3)
    tile_size = 32
    offset = [32, 32]
    patches = _array_to_patches_single(arr, tile_size, offset)

    assert patches.shape == (9, 32, 32, 3)

def test_array_to_patches_single_no_divisible_dimensions():
    """ Test that the array is correctly converted to patches. (no divisible dimensions) """
    arr = np.random.rand(70, 70, 3)
    tile_size = 32
    patches = _array_to_patches_single(arr, tile_size)

    assert patches.shape == (4, 32, 32, 3)


def test_patches_to_array_single_basic():
    """ Test that the patches are correctly converted to an array. (basic) """
    arr = np.random.rand(64, 64, 3)
    tile_size = 32
    patches = _array_to_patches_single(arr, tile_size)
    restored_arr = _patches_to_array_single(patches, arr.shape, tile_size)

    assert np.allclose(arr, restored_arr)

def test_patches_to_array_single_with_offset():
    """ Test that the patches are correctly converted to an array. (with offset) """
    arr = np.random.rand(64, 64, 3)
    tile_size = 32
    offset = [16, 16]
    patches = _array_to_patches_single(arr, tile_size, offset)
    restored_arr = _patches_to_array_single(patches, arr.shape, tile_size, offset)

    assert np.allclose(arr[16:48, 16:48], restored_arr[16:48, 16:48])

def test_patches_to_array_single_no_divisible_dimensions():
    """ Test that the patches are correctly converted to an array. (no divisible dimensions) """
    arr = np.random.rand(70, 70, 3)
    tile_size = 32
    patches = _array_to_patches_single(arr, tile_size)
    restored_arr = _patches_to_array_single(patches, arr.shape, tile_size)

    assert np.allclose(arr[:64, :64], restored_arr[:64, :64])

def test_patches_to_array_single_different_height_and_width():
    """ Test that the patches are correctly converted to an array. (different height and width) """
    arr = np.random.rand(80, 100, 3)
    tile_size = 32
    offset = [0, 0]
    patches = _array_to_patches_single(arr, tile_size, offset)
    restored_arr = _patches_to_array_single(patches, arr.shape, tile_size, offset)

    assert np.allclose(arr[:64, :96], restored_arr[:64, :96])

def test_patches_to_array_single_different_offset_height_and_width():
    """ Test that the patches are correctly converted to an array. (different offset height and width) """
    arr = np.random.rand(80, 100, 3)
    tile_size = 32
    offset = [20, 10]
    patches = _array_to_patches_single(arr, tile_size, offset)
    restored_arr = _patches_to_array_single(patches, arr.shape, tile_size, offset)

    assert np.allclose(arr[20:52, 10:74], restored_arr[20:52, 10:74])

def test_array_to_patches_basic():
    """ Test that the array is correctly divided into patches (basic). """
    arr = np.random.rand(64, 64, 3)
    tile_size = 32
    offsets = 1
    patches = array_to_patches(arr, tile_size, n_offsets=offsets)

    assert patches.shape == (5, tile_size, tile_size, arr.shape[2])

def test_array_to_patches_multiple_offsets():
    """ Test that the array is correctly divided into patches (multiple offsets). """
    arr = np.random.rand(96, 96, 3)
    tile_size = 32
    offsets = 3
    patches = array_to_patches(arr, tile_size, n_offsets=offsets)

    assert patches.shape == (21, tile_size, tile_size, arr.shape[2])


# Dummy callback function for testing purposes
def dummy_callback(arr: np.ndarray) -> np.ndarray:
    return arr * 2

def dummy_callback_mult(arr: np.ndarray) -> np.ndarray:
    return np.concatenate([arr, arr], axis=3)

def test_predict_array_basic_1d():
    """ Test the basic functionality of the predict_array function. """
    arr = np.random.rand(64, 64, 1)
    tile_size = 32
    offsets = 1

    predicted_array = predict_array(arr, dummy_callback, tile_size=tile_size, n_offsets=offsets)

    assert predicted_array.shape[:2] == arr.shape[:2]
    assert np.allclose(predicted_array, arr * 2)

def test_predict_array_basic_3d():
    """ Test the basic functionality of the predict_array function. """
    arr = np.random.rand(64, 64, 3)
    tile_size = 32
    offsets = 1

    predicted_array = predict_array(arr, dummy_callback, tile_size=tile_size, n_offsets=offsets)

    assert predicted_array.shape == arr.shape
    assert np.allclose(predicted_array, arr * 2)

def test_predict_array_basic_1d_to_3d():
    """ Test the basic functionality of the predict_array function. """
    arr = np.random.rand(64, 64, 3)
    tile_size = 32
    offsets = 1

    predicted_array = predict_array(arr, dummy_callback_mult, tile_size=tile_size, n_offsets=offsets)

    assert predicted_array.shape[:2] == arr.shape[:2]
    assert predicted_array.shape[2] == arr.shape[2] * 2
    assert np.allclose(predicted_array[:, :, :3], arr)
    assert np.allclose(predicted_array[:, :, 3:], arr)

def test_predict_array_invalid_merge_method():
    """ Test the predict_array function with an invalid merge_method. """
    arr = np.random.rand(64, 64, 1)
    tile_size = 32
    offsets = 1

    with pytest.raises(AssertionError):
        predict_array(arr, dummy_callback, tile_size=tile_size, n_offsets=offsets, merge_method="invalid_method")

def test_predict_array_2D_array():
    """ Test the predict_array function with a 2D array. """
    arr = np.random.rand(64, 64)
    tile_size = 32
    offsets = 1

    with pytest.raises(AssertionError):
        predict_array(arr, dummy_callback, tile_size=tile_size, n_offsets=offsets)

def test_predict_array_large_tile_size():
    """ Test the predict_array function with a tile_size larger than the array dimensions. """
    arr = np.random.rand(64, 64, 1)
    tile_size = 128
    offsets = 1

    with pytest.raises(AssertionError):
        predict_array(arr, dummy_callback, tile_size=tile_size, n_offsets=offsets)

def test_predict_array_merge_method_median():
    """ Test the predict_array function with merge_method set to 'median'. """
    arr = np.random.rand(64, 64, 2)
    tile_size = 32
    offsets = 1

    predicted_array = predict_array(arr, dummy_callback, tile_size=tile_size, n_offsets=offsets, merge_method="median")

    assert predicted_array.shape == arr.shape
    assert np.allclose(predicted_array, arr * 2)

def test_predict_array_merge_method_mean():
    """ Test the predict_array function with merge_method set to 'mean'. """
    arr = np.random.rand(64, 64, 3)
    tile_size = 32
    offsets = 1

    predicted_array = predict_array(arr, dummy_callback, tile_size=tile_size, n_offsets=offsets, merge_method="mean")

    assert predicted_array.shape == arr.shape
    assert np.allclose(predicted_array, arr * 2)

def test_predict_array_merge_method_mode():
    """ Test the predict_array function with merge_method set to 'mode'. """
    arr = np.rint(np.random.rand(64, 64, 1) * 100).astype(np.int32)
    tile_size = 32
    offsets = 1

    predicted_array = predict_array(arr, dummy_callback, tile_size=tile_size, n_offsets=offsets, merge_method="mode")

    assert predicted_array.shape == arr.shape

def dummy_callback_pixel(arr: np.ndarray) -> np.ndarray:
    """ A simple dummy callback function that returns the input array squared. """
    return arr * 3

def test_predict_array_pixel():
    """ Test the predict_array_pixel function to ensure it returns an array of the same shape as input. """
    arr = np.rint(np.random.rand(64, 64, 3) * 100)

    arr_pred = predict_array_pixel(arr, dummy_callback_pixel)

    assert arr_pred.shape == arr.shape, f"Expected shape {arr.shape}, but got {arr_pred.shape}"
    assert np.allclose(arr_pred, arr * 3)
