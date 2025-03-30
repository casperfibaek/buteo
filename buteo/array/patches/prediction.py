""" ### Functions for predicting on patches. ### """

# Standard library
from typing import Union, List, Tuple, Optional, Callable

# External
import numpy as np

# Internal
from buteo.array.utils_array import channel_first_to_last, channel_last_to_first
from buteo.array.patches.util import (
    _get_offsets,
    _borders_are_necessary_list,
    _patches_to_weights,
)
from buteo.array.patches.extraction import (
    _array_to_patches_single,
    _patches_to_array_single,
)
from buteo.array.patches.merging import (
    _merge_weighted_mad,
    _merge_weighted_median,
    _merge_weighted_average,
    _merge_weighted_mode,
    _merge_weighted_minmax,
    _merge_weighted_olympic,
)


def predict_array(
    arr: np.ndarray,
    callback: Callable[[np.ndarray], np.ndarray], *,
    tile_size: int = 64,
    n_offsets: int = 1,
    border_check: bool = True,
    merge_method: str = "median",
    edge_weighted: bool = True,
    edge_distance: int = 3,
    batch_size: int = 1,
    channel_last: bool = True,
    background_value: Optional[Union[int, float]] = None,
) -> np.ndarray:
    """Generate patches from an array. Also outputs the offsets and the shapes of the offsets. Only
    suppors the prediction of single values in the rasters/arrays.

    Parameters
    ----------
    arr : np.ndarray
        A numpy array to be divided into patches.

    callback : Callable[[np.ndarray], np.ndarray]
        The callback function to be used for prediction. The callback function
        must take a numpy array as input and return a numpy array as output.

    tile_size : int, optional
        The size of each tile/patch, e.g., 64 for 64x64 tiles. Default: 64

    n_offsets : int, optional
        The desired number of offsets to be calculated. Default: 1

    border_check : bool, optional
        Whether or not to include border patches. Default: True

    merge_method : str, optional
        The method to use for merging the patches. Valid methods
        are ['mad', 'median', 'mean', 'mode', "min", "max", "olympic1", "olympic2"]. Default: "median"

    edge_weighted : bool, optional
        Whether or not to weight the edges patches of patches less than the central parts. Default: True

    edge_distance : int, optional
        The distance from the edge to be weighted less. Usually good to
        adjust this to your maximum convolution kernel size. Default: 3

    batch_size : int, optional
        The batch size to use for prediction. Default: 1

    channel_last : bool, optional
        Whether or not the channel dimension is the last dimension. Default: True

    background_value : Optional[Union[int, float]], optional
        The value to use for the background. If not provided, defaults to np.nan.

    Returns
    -------
    np.ndarray
        The predicted array.
    """
    assert merge_method in ["mad", "median", "mean", "mode", "min", "max", "olympic1", "olympic2"], "Invalid merge method"
    assert len(arr.shape) == 3, "Array must be 3D"

    if channel_last:
        assert arr.shape[0] >= tile_size, "Array must be larger or equal to tile_size"
        assert arr.shape[1] >= tile_size, "Array must be larger or equal to tile_size"
    else:
        assert arr.shape[1] >= tile_size, "Array must be larger or equal to tile_size"
        assert arr.shape[2] >= tile_size, "Array must be larger or equal to tile_size"

    if not channel_last:
        arr = channel_first_to_last(arr)

    # Get the list of offsets for both x and y dimensions
    offsets = _get_offsets(tile_size, n_offsets)

    if border_check:
        borders_y, borders_x = _borders_are_necessary_list(arr, tile_size, offsets)

        # TODO: Investigate how to handle smarter. Currently we might get duplicates.
        if borders_y or borders_x:
            offsets.append((0, arr.shape[1] - tile_size))
            offsets.append((arr.shape[0] - tile_size, 0))
            offsets.append((arr.shape[0] - tile_size, arr.shape[1] - tile_size))

    # Test output dimensions of prediction
    test_patch = arr[np.newaxis, :tile_size, :tile_size, :]
    test_prediction = callback(test_patch)
    test_shape = test_prediction.shape

    # Initialize an empty list to store the generated patches
    predictions = np.zeros((len(offsets), arr.shape[0], arr.shape[1], test_shape[-1]), dtype=arr.dtype)
    predictions_weights = np.zeros((len(offsets), arr.shape[0], arr.shape[1], 1), dtype="float32")

    # Iterate through the offsets and generate patches for each offset
    for idx_i, offset in enumerate(offsets):
        patches = _array_to_patches_single(arr, tile_size, offset)

        offset_predictions = np.empty((patches.shape[0], patches.shape[1], patches.shape[2], test_shape[-1]), dtype=arr.dtype)
        offset_weights = np.empty((patches.shape[0], patches.shape[1], patches.shape[2], 1), dtype="float32")

        for idx_j in range((patches.shape[0] // batch_size) + (1 if patches.shape[0] % batch_size != 0 else 0)):
            idx_start = idx_j * batch_size
            idx_end = (idx_j + 1) * batch_size

            if not channel_last:
                prediction = channel_first_to_last(
                    callback(
                        channel_last_to_first(patches[idx_start:idx_end, ...])
                    )
                )
            else:
                prediction = callback(patches[idx_start:idx_end, ...])

            if edge_weighted:
                weights = _patches_to_weights(patches[idx_start:idx_end, ...], edge_distance)
            else:
                weights = np.ones((patches.shape[1], patches.shape[2], 1), dtype="float32")
                weights = np.repeat(weights[np.newaxis, ...], min(batch_size, patches.shape[0] - idx_start), axis=0)

            offset_predictions[idx_start:idx_end, ...] = prediction
            offset_weights[idx_start:idx_end, ...] = weights

        predictions[idx_i, :, :, :] = _patches_to_array_single(offset_predictions, (arr.shape[0], arr.shape[1], test_shape[-1]), tile_size, offset)
        predictions_weights[idx_i, :, :, :] = _patches_to_array_single(offset_weights, (arr.shape[0], arr.shape[1], 1), tile_size, offset, 0.0)

    ret_arr = np.empty((predictions.shape[1], predictions.shape[2], predictions.shape[3]), dtype=predictions.dtype)

    # check if patches.dtype is integer types
    if background_value is not None:
        ret_arr[:] = background_value
    elif predictions.dtype.kind in "ui":
        ret_arr[:] = np.iinfo(predictions.dtype).min
    else:
        ret_arr[:] = np.nan

    # Merge the predictions
    if merge_method == "mad":
        ret_arr = _merge_weighted_mad(predictions, predictions_weights, ret_arr)
    elif merge_method == "median":
        ret_arr = _merge_weighted_median(predictions, predictions_weights, ret_arr)
    elif merge_method in ["mean", "average", "avg"]:
        ret_arr = _merge_weighted_average(predictions, predictions_weights, ret_arr)
    elif merge_method == "mode":
        ret_arr = _merge_weighted_mode(predictions, predictions_weights, ret_arr)
    elif merge_method == "max":
        ret_arr = _merge_weighted_minmax(predictions, predictions_weights, ret_arr, "max")
    elif merge_method == "min":
        ret_arr = _merge_weighted_minmax(predictions, predictions_weights, ret_arr, "min")
    elif merge_method == "olympic1":
        ret_arr = _merge_weighted_olympic(predictions, predictions_weights, ret_arr, 1)
    elif merge_method == "olympic2":
        ret_arr = _merge_weighted_olympic(predictions, predictions_weights, ret_arr, 2)

    if not channel_last:
        ret_arr = channel_last_to_first(ret_arr)

    return ret_arr


def predict_array_pixel(
    arr: np.ndarray,
    callback: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """Predicts an array pixel by pixel.

    Args:
        arr (np.ndarray): A numpy array to be divided into patches.
        callback (function): The callback function to be used for prediction. The callback function
            must take a numpy array as input and return a numpy array as output.
            
    Returns:
        np.ndarray: The predicted array.
    """
    assert len(arr.shape) == 3, "Array must be 3D"

    reshaped = arr.reshape((arr.shape[0] * arr.shape[1], arr.shape[2]))
    predicted = callback(reshaped)
    predicted = predicted.reshape((arr.shape[0], arr.shape[1], predicted.shape[-1]))

    return predicted
