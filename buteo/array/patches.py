"""Create patches from rasters, used for machine learnign applications."""

import sys; sys.path.append("../../")
from typing import Union, List, Tuple, Optional, Callable

import numpy as np
from numba import prange, jit
from buteo.array.utils_array import channel_first_to_last, channel_last_to_first


def _get_kernel_weights(
    tile_size: int = 64,
    edge_distance: int = 5,
    epsilon: float = 1e-7,
) -> np.ndarray:
    """Weight a kernel according to how close to an edge a given pixel is.

    Parameters
    ----------
    tile_size : int, optional
        The size of the square kernel. Default: 64

    edge_distance : int, optional
        The distance from the edge to consider for weighting. Default: 5

    epsilon : float, optional
        A small value to prevent division by zero. Default: 1e-7

    Returns
    -------
    np.ndarray
        A 2D NumPy array of shape (tile_size, tile_size) with the kernel weights.
    """
    assert tile_size > 0, "Tile size must be greater than zero."
    assert edge_distance < tile_size // 2, "Edge distance must be less than half the tile size."
    assert edge_distance >= 0, "Edge distance must be greater than or equal to zero."

    arr = np.zeros((tile_size, tile_size), dtype="float32")
    max_dist = edge_distance * 2

    # Iterate through the kernel array
    for idx_y in range(0, arr.shape[0]):
        for idx_x in range(0, arr.shape[1]):

            # Calculate vertical distance to the closest edge
            val_y_top = max(edge_distance - idx_y, 0.0)
            val_y_bot = max((1 + edge_distance) - (tile_size - idx_y), 0.0)
            val_y = val_y_top + val_y_bot

            # Calculate horizontal distance to the closest edge
            val_x_lef = max(edge_distance - idx_x, 0.0)
            val_x_rig = max((1 + edge_distance) - (tile_size - idx_x), 0.0)
            val_x = val_x_lef + val_x_rig

            # Calculate the weight based on the distance to the closest edge
            val = (max_dist - abs(val_y + val_x)) / max_dist

            # Set a minimum weight to avoid division by zero
            if val <= 0.0:
                val = epsilon

            # Assign the calculated weight to the kernel array
            arr[idx_y, idx_x] = val

    return arr


@jit(nopython=True, parallel=True, nogil=True, cache=True)
def _merge_weighted_median(
    arr: np.ndarray,
    arr_weight: np.ndarray,
) -> np.ndarray:
    """Calculate the weighted median of a multi-dimensional array along the first axis.
    This is the order (number_of_overlaps, tile_size, tile_size, number_of_bands)

    Parameters
    ----------
    arr : np.ndarray
        The input array.

    arr_weight : np.ndarray
        The weight array with the same shape as the input array.

    Returns
    -------
    np.ndarray
        A 3D NumPy array of shape (arr.shape[1], arr.shape[2], arr.shape[3]) with the weighted medians.
    """
    ret_arr = np.empty((arr.shape[1], arr.shape[2], arr.shape[3]), dtype="float32")
    ret_arr[:] = np.nan

    # Iterate through the input array
    for idx_y in prange(arr.shape[1]):
        for idx_x in range(arr.shape[2]):
            for idx_band in range(arr.shape[3]):

               # Flatten the input and weight arrays
                values = arr[:, idx_y, idx_x, idx_band].flatten()
                weights = arr_weight[:, idx_y, idx_x, 0].flatten()

                nan_mask = np.where(~np.isnan(values))[0]

                if len(nan_mask) == 0:
                    continue

                values = values[nan_mask]
                weights = weights[nan_mask]

                # Sort the values and weights based on the values
                sort_mask = np.argsort(values)
                sorted_data = values[sort_mask]
                sorted_weights = weights[sort_mask]

                # Calculate the cumulative sum of the sorted weights
                cumsum = np.cumsum(sorted_weights)

                # Normalize the cumulative sum to the range [0, 1]
                intersect = (cumsum - 0.5 * sorted_weights) / cumsum[-1]

                # Interpolate the weighted median and store it in the result array
                ret_arr[idx_y, idx_x, idx_band] = np.interp(0.5, intersect, sorted_data)

    return ret_arr


@jit(nopython=True, parallel=True, nogil=True, cache=True)
def _merge_weighted_average(
    arr: np.ndarray,
    arr_weight: np.ndarray,
) -> np.ndarray:
    """Calculate the weighted average of a multi-dimensional array along the last axis.

    Parameters
    ----------
    arr : np.ndarray
        The input array.

    arr_weight : np.ndarray
        The weight array with the same shape as the input array.

    Returns
    -------
    np.ndarray
        A 3D NumPy array of shape (arr.shape[0], arr.shape[1], 1) with the weighted averages.
    """
    ret_arr = np.empty((arr.shape[1], arr.shape[2], arr.shape[3]), dtype="float32")
    ret_arr[:] = np.nan

    # Iterate through the input array
    for idx_y in prange(arr.shape[1]):
        for idx_x in range(arr.shape[2]):
            for idx_band in range(arr.shape[3]):

                # Flatten the input and weight arrays
                values = arr[:, idx_y, idx_x, idx_band].flatten()
                weights = arr_weight[:, idx_y, idx_x, 0].flatten()

                nan_mask = np.where(~np.isnan(values))[0]

                if len(nan_mask) == 0:
                    continue

                values = values[nan_mask]
                weights = weights[nan_mask]

                # Calculate the weighted sum and total weight
                weighted_sum = np.nansum(values * weights)
                total_weight = np.nansum(weights)

                # Calculate the weighted average and store it in the result array
                ret_arr[idx_y, idx_x, idx_band] = weighted_sum / total_weight

    return ret_arr

@jit(nopython=True, parallel=True, nogil=True, cache=True)
def _merge_weighted_minmax(
    arr: np.ndarray,
    arr_weight: np.ndarray,
    method="max",
) -> np.ndarray:
    """Calculate the weighted min or max of a multi-dimensional array along the last axis.

    Parameters
    ----------
    arr : np.ndarray
        The input array.

    arr_weight : np.ndarray
        The weight array with the same shape as the input array.

    method : str, optional
        The method to use. Either "min" or "max". Default: "max"

    Returns
    -------
    np.ndarray
        A 3D NumPy array of shape (arr.shape[0], arr.shape[1], 1) with the weighted min or max.
    """
    ret_arr = np.empty((arr.shape[1], arr.shape[2], arr.shape[3]), dtype="float32")
    ret_arr[:] = np.nan

    minmax = 0
    if method == "min":
        minmax = 1

    # Iterate through the input array
    for idx_y in prange(arr.shape[1]):
        for idx_x in range(arr.shape[2]):
            for idx_band in range(arr.shape[3]):

                # Flatten the input and weight arrays
                values = arr[:, idx_y, idx_x, idx_band].flatten()
                weights = arr_weight[:, idx_y, idx_x, 0].flatten()

                nan_mask = np.where(~np.isnan(values))[0]

                if len(nan_mask) == 0:
                    continue

                values = values[nan_mask]
                weights = weights[nan_mask]

                weighted = values * weights

                if minmax == 0: # max
                    index = np.nanargmax(weighted)
                else:
                    index = np.nanargmin(weighted)

                value = values[index]

                # Calculate the weighted average and store it in the result array
                ret_arr[idx_y, idx_x, idx_band] = value

    return ret_arr


@jit(nopython=True, parallel=True, nogil=True, cache=True)
def _merge_weighted_olympic(
    arr: np.ndarray,
    arr_weight: np.ndarray,
    level: int = 1,
) -> np.ndarray:
    """Calculate the olympic value of a multi-dimensional array along the last axis.
    Using olympic sort, the highest and lowest values are removed from the calculation.
    If level is 1, then the highest and loweest values are removed. If the level is 2,
    then the 2 highest and lowest values are removed, and so on.

    Parameters
    ----------
    arr : np.ndarray
        The input array.

    arr_weight : np.ndarray
        The weight array with the same shape as the input array.

    level : int, optional
        The level of olympic sort to use. Default: 1

    Returns
    -------
    np.ndarray
        A 3D NumPy array of shape (arr.shape[0], arr.shape[1], 1) with the olympic value.
    """
    ret_arr = np.empty((arr.shape[1], arr.shape[2], arr.shape[3]), dtype="float32")
    ret_arr[:] = np.nan

    required = int((level * 2) + 1)

    # Iterate through the input array
    for idx_y in prange(arr.shape[1]):
        for idx_x in range(arr.shape[2]):
            for idx_band in range(arr.shape[3]):

                # Flatten the input and weight arrays
                values = arr[:, idx_y, idx_x, idx_band].flatten()
                weights = arr_weight[:, idx_y, idx_x, 0].flatten()

                nan_mask = np.where(~np.isnan(values))[0]

                if len(nan_mask) == 0:
                    continue

                values = values[nan_mask]
                weights = weights[nan_mask]

                if len(values) < required: # Take the average of all
                    value = np.mean(values)
                elif len(values) == required: # Take the middle value
                    value = np.sort(values)[level]
                else:
                    sort_olympic = np.argsort(values)[level:-level]
                    sort_weights = weights[sort_olympic] / np.sum(weights[sort_olympic])
                    sort_values = values[sort_olympic]

                    value = np.sum(sort_values * sort_weights)

                # Calculate the weighted average and store it in the result array
                ret_arr[idx_y, idx_x, idx_band] = value

    return ret_arr


@jit(nopython=True, parallel=True, nogil=True)
def _merge_weighted_mad(
    arr: np.ndarray,
    arr_weight: np.ndarray,
    mad_dist: float = 2.0,
) -> np.ndarray:
    """Merge an array of predictions using the MAD-merge methodology.

    Parameters
    ----------
    arr : np.ndarray
        The input array.

    arr_weight : np.ndarray
        The weight array with the same shape as the input array.

    mad_dist : float, optional
        The MAD distance. Default: 2.0

    Returns
    -------
    np.ndarray
        A 3D NumPy array of shape (arr.shape[0], arr.shape[1], 1) with the MAD-merged values.
    """
    ret_arr = np.empty((arr.shape[1], arr.shape[2], arr.shape[3]), dtype="float32")
    ret_arr[:] = np.nan

    # Iterate through the input array
    for idx_y in prange(arr.shape[1]):
        for idx_x in range(arr.shape[2]):
            for idx_band in range(arr.shape[3]):

                # Flatten the input and weight arrays
                values = arr[:, idx_y, idx_x, idx_band].flatten()
                weights = arr_weight[:, idx_y, idx_x, 0].flatten()

                nan_mask = np.where(~np.isnan(values))[0]

                if len(nan_mask) == 0:
                    continue

                values = values[nan_mask]
                weights = weights[nan_mask]

                # Sort the values and weights based on the values
                sort_mask = np.argsort(values)
                sorted_data = values[sort_mask]
                sorted_weights = weights[sort_mask]

                # Calculate the cumulative sum of the sorted weights and normalize to the range [0, 1]
                cumsum = np.cumsum(sorted_weights)
                intersect = (cumsum - 0.5 * sorted_weights) / cumsum[-1]

                # Interpolate the median
                median = np.interp(0.5, intersect, sorted_data)

                # Calculate the median absolute deviation (MAD)
                mad = np.median(np.abs(median - values))

                # If MAD is zero, store the median in the result array and continue
                if mad == 0.0:
                    ret_arr[idx_y, idx_x, 0] = median
                    continue

                # Calculate the new weights based on the MAD
                new_weights = np.zeros_like(sorted_weights)
                for idx_z in range(sorted_data.shape[0]):
                    new_weights[idx_z] = 1.0 - (np.minimum(np.abs(sorted_data[idx_z] - median) / (mad * mad_dist), 1))

                if np.sum(new_weights) == 0.0:
                    ret_arr[idx_y, idx_x, 0] = median
                    continue

                # Calculate the cumulative sum of the new weights and normalize to the range [0, 1]
                cumsum = np.cumsum(new_weights)
                intersect = (cumsum - 0.5 * new_weights) / cumsum[-1]

                # Interpolate the MAD-merged value and store it in the result array
                ret_arr[idx_y, idx_x, idx_band] = np.interp(0.5, intersect, sorted_data)

    return ret_arr


@jit(nopython=True, nogil=True)
def _unique_values(arr: np.ndarray) -> np.ndarray:
    """Find the unique values in a 1D NumPy array.

    Parameters
    ----------
    arr : np.ndarray
        The input array.

    Returns
    -------
    np.ndarray
        A 1D NumPy array with the unique values.
    """
    unique = np.empty(arr.size, dtype=arr.dtype)
    unique_count = 0
    for i in range(arr.shape[0]):
        if arr[i] not in unique[:unique_count]:
            unique[unique_count] = arr[i]
            unique_count += 1

    return unique[:unique_count]


@jit(nopython=True, parallel=True, nogil=True)
def _merge_weighted_mode(
    arr: np.ndarray,
    arr_weight: np.ndarray,
) -> np.ndarray:
    """Calculate the weighted mode of a multi-dimensional array along the last axis.

    Parameters
    ----------
    arr : np.ndarray
        The input array.

    arr_weight : np.ndarray
        The weight array with the same shape as the input array.

    Returns
    -------
    np.ndarray
        A 3D NumPy array of shape (arr.shape[0], arr.shape[1], 1) with the weighted modes.
    """
    ret_arr = np.empty((arr.shape[1], arr.shape[2], arr.shape[3]), dtype=arr.dtype)
    ret_arr[:] = np.nan

    # Iterate through the input array
    for idx_y in prange(arr.shape[1]):
        for idx_x in range(arr.shape[2]):
            for idx_band in range(arr.shape[3]):

                # Flatten the input and weight arrays
                values = arr[:, idx_y, idx_x, idx_band].flatten()
                weights = arr_weight[:, idx_y, idx_x, 0].flatten()

                nan_mask = np.where(~np.isnan(values))[0]

                if len(nan_mask) == 0:
                    continue

                values = values[nan_mask]
                weights = weights[nan_mask]

                # Get unique values and their weighted counts
                unique_vals = _unique_values(values)
                weighted_counts = np.zeros(unique_vals.shape[0])

                # Calculate the weighted sum for each unique value
                for i in range(unique_vals.shape[0]):
                    idxs = np.where(values == unique_vals[i])
                    weighted_counts[i] = np.sum(weights[idxs])

                # Get the index of the maximum weighted sum
                mode_idx = np.argmax(weighted_counts)

                # Store the weighted mode in the result array
                ret_arr[idx_y, idx_x, idx_band] = unique_vals[mode_idx]

    return ret_arr


def _get_offsets(
    tile_size: int,
    n_offsets: int,
):
    """Generate a list of offset pairs for a given tile size and number of offsets in y and x dimensions.

    Parameters
    ----------
    tile_size : int
        The size of each tile.

    n_offsets : int
        The desired number of offsets to be calculated in the y and x dimensions.

    Returns
    -------
    List[Tuple[int, int]]
        A list of tuples containing offset pairs for y and x dimensions.
        order is (y, x)
    """
    offset_props = np.arange(0, 1, 1 / (n_offsets + 1))[1:].tolist()
    offsets = [(0, 0)]

    assert tile_size > n_offsets, f"Too many offsets ({n_offsets}) requested for tile_size {tile_size}"

    for val in offset_props:
        offset = int(round((val * tile_size), 2))
        offsets.append((offset, offset))

    return offsets


def _borders_are_necessary(
    arr: np.ndarray,
    tile_size: int,
    offset: List[int],
) -> Tuple[bool, bool]:
    """Checks if borders are necessary for the given array.
    Width and height are returned as a tuple.
    order is (y, x).

    Parameters
    ----------
    arr : np.ndarray
        The array to be checked.

    tile_size : int
        The size of each tile.

    offset : List[int]
        The offset to be used.

    Returns
    -------
    Tuple[bool, bool]
        A tuple containing of borders are needed in (height, width) dims.
    """
    if arr.ndim == 2:
        height, width = arr.shape
    else:
        height, width, _ = arr.shape

    if (height - offset[0]) % tile_size == 0:
        height_border = False
    else:
        height_border = True

    if (width - offset[1]) % tile_size == 0:
        width_border = False
    else:
        width_border = True

    return height_border, width_border


def _borders_are_necessary_list(
    arr: np.ndarray,
    tile_size: int,
    offsets: List[List[int]],
) -> Tuple[bool, bool]:
    """Checks if borders are necessary for the given array.
    Width and height are returned as a tuple.

    Parameters
    ----------
    arr : np.ndarray
        The array to be checked.

    tile_size : int
        The size of each tile.

    offsets : List[List[int]]
        The offsets to be used.

    Returns
    -------
    Tuple[bool, bool]
        A tuple containing of borders are needed in (height, width) dims.
    """
    height_border = True
    width_border = True

    for offset in offsets:
        offset_height_border, offset_width_border = _borders_are_necessary(
            arr, tile_size, offset
        )
        if not offset_height_border:
            height_border = False

        if not offset_width_border:
            width_border = False

        if not height_border and not width_border:
            break

    return height_border, width_border


def _array_to_patches_single(
    arr: np.ndarray,
    tile_size: int,
    offset: Optional[Union[List[int], Tuple[int, int]]] = None,
) -> np.ndarray:
    """Generate patches from an array. Offsets in (y, x) order.

    Parameters
    ----------
    arr : np.ndarray
        The array to be divided into patches.

    tile_size : int
        The size of each tile/patch, e.g., 64 for 64x64 tiles.

    offset : Optional[Union[List[int], Tuple[int, int]]], optional
        The y and x offset values for the input array. If not provided, defaults to [0, 0].

    Returns
    -------
    np.ndarray
        A numpy array containing the patches.
    """
    assert arr.ndim in [2, 3], "Array must be 2D or 3D"
    assert tile_size > 0, "Tile size must be greater than 0"
    assert offset is None or len(offset) == 2, "Offset must be a list or tuple of length 2"

    # Set default offset to [0, 0] if not provided
    if offset is None:
        offset = [0, 0]

    # Calculate the number of patches in the y and x dimensions
    patches_y = (arr.shape[0] - offset[0]) // tile_size
    patches_x = (arr.shape[1] - offset[1]) // tile_size

    # Calculate cut dimensions for the y and x dimensions
    cut_y = -((arr.shape[0] - offset[0]) % tile_size)
    cut_x = -((arr.shape[1] - offset[1]) % tile_size)

    # Set cut dimensions to None if they are 0
    cut_y = None if cut_y == 0 else cut_y
    cut_x = None if cut_x == 0 else cut_x

    # Reshape the array to separate the patches
    reshaped = arr[offset[0]:cut_y, offset[1]:cut_x].reshape(
        patches_y,
        tile_size,
        patches_x,
        tile_size,
        arr.shape[2],
    )

    # Swap axes to rearrange patches in the correct order
    swaped = reshaped.swapaxes(1, 2)

    # Combine the patches into a single array
    blocks = swaped.reshape(-1, tile_size, tile_size, arr.shape[2])

    return blocks


def _patches_to_array_single(
    patches: np.ndarray,
    shape: Union[List, Tuple],
    tile_size: int,
    offset: Optional[Union[List, Tuple]] = None,
    background_value: Optional[Union[int, float]] = None,
) -> np.ndarray:
    """Reconstitute an array from patches.

    Given an array of patches, this function stitches them back together
    to form the original array of the specified shape.

    Parameters
    ----------
    patches : np.ndarray
        A numpy array containing the patches to be stitched together.

    shape : Union[List, Tuple]
        The desired shape of the output array.

    tile_size : int
        The size of each tile/patch, e.g., 64 for 64x64 tiles.

    offset : Optional[Union[List, Tuple]], optional
        The y and x offset values for the target array. If not provided, defaults to [0, 0].

    background_value : Optional[Union[int, float]], optional
        The value to use for the background. If not provided, defaults to np.nan.

    Returns
    -------
    np.ndarray
        A numpy array with the original shape, formed by stitching together the provided patches.
    """
    assert len(shape) in [2, 3], "Shape must be a tuple or list of length 2 or 3"
    assert len(patches.shape) == 4, "Patches must be a 4D array"
    assert patches.shape[1] == tile_size, "Patches must be of size tile_size"
    assert patches.shape[2] == tile_size, "Patches must be of size tile_size"
    assert offset is None or len(offset) == 2, "Offset must be a tuple or list of length 2"

    # Set default offset to [0, 0] if not provided
    if offset is None:
        offset = [0, 0]

    # Create an empty target array of the specified shape
    if background_value is None:
        target = np.full(shape, np.nan, dtype=patches.dtype)
    else:
        target = np.full(shape, background_value, dtype=patches.dtype)

    # Calculate target dimensions
    target_y = ((shape[0] - offset[0]) // tile_size) * tile_size
    target_x = ((shape[1] - offset[1]) // tile_size) * tile_size

    # Calculate cut dimensions
    cut_y = -((shape[0] - offset[0]) % tile_size)
    cut_x = -((shape[1] - offset[1]) % tile_size)

    # Set cut dimensions to None if they are 0
    cut_y = None if cut_y == 0 else cut_y
    cut_x = None if cut_x == 0 else cut_x

    # Calculate the number of tiles in the y and x dimensions
    num_tiles_y = target_y // tile_size
    num_tiles_x = target_x // tile_size

    # Reshape the patches for stitching
    reshape = patches.reshape(
        num_tiles_y,
        num_tiles_x,
        tile_size,
        tile_size,
        patches.shape[3],
        1,
    )

    # Swap axes to rearrange patches in the correct order for stitching
    swap = reshape.swapaxes(1, 2)

    # Combine the patches into a single array
    destination = swap.reshape(
        num_tiles_y * tile_size,
        num_tiles_x * tile_size,
        patches.shape[3],
    )

    # Assign the combined patches to the target array
    target[offset[0]:cut_y, offset[1]:cut_x] = destination

    return target


def _patches_to_weights(
    patches: np.ndarray,
    edge_distance: int,
) -> np.ndarray:
    """Calculate the weights for each patch based on the distance to the edge."""
    assert len(patches.shape) == 4, "Patches must be a 4D array"
    assert patches.shape[1] == patches.shape[2], "Patches must be square"

    # Calculate the distance to the edge for each patch
    weights = _get_kernel_weights(patches.shape[1], edge_distance)

    # Expand the weights to match the number of patches
    weights = np.repeat(weights[np.newaxis, ...], patches.shape[0], axis=0)[..., np.newaxis]

    return weights


def array_to_patches(
    arr: np.ndarray,
    tile_size: int, *,
    n_offsets: int = 0,
    border_check: bool = True,
    channel_last: bool = True,
) -> np.ndarray:
    """Generate patches from an array based on the specified parameters.

    Parameters
    ----------
    arr : np.ndarray
        A numpy array to be divided into patches.

    tile_size : int
        The size of each tile/patch, e.g., 64 for 64x64 tiles.

    n_offsets : int, optional
        The desired number of offsets to be calculated. Default: 0

    border_check : bool, optional
        Whether or not to include border patches. Default: True

    channel_last : bool, optional
        Whether or not the channel dimension is the last dimension. Default: True

    Returns
    -------
    np.ndarray
        The concatenate patches along axis 0. In the order (patches, y, x, channels)
    """
    assert len(arr.shape) == 3, "Array must be 3D"
    if channel_last:
        assert arr.shape[0] >= tile_size, "Array must be larger or equal to tile_size"
        assert arr.shape[1] >= tile_size, "Array must be larger or equal to tile_size"
    else:
        assert arr.shape[1] >= tile_size, "Array must be larger or equal to tile_size"
        assert arr.shape[2] >= tile_size, "Array must be larger or equal to tile_size"

    assert tile_size > 0, "Tile size must be greater than 0"
    assert n_offsets >= 0, "Number of offsets must be greater than or equal to 0"
    assert isinstance(border_check, bool), "Border check must be a boolean"
    assert isinstance(n_offsets, int), "Number of offsets must be an integer"

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

    # Initialize an empty list to store the generated patches
    patches = []

    # Iterate through the offsets and generate patches for each offset
    for offset in offsets:
        patches.append(
            _array_to_patches_single(arr, tile_size, offset),
        )

    patches = np.concatenate(patches, axis=0)

    if not channel_last:
        patches = channel_last_to_first(patches)

    return patches


# TODO: Add support for multiple input values in the rasters/arrays
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
    predictions_weights = np.zeros((len(offsets), arr.shape[0], arr.shape[1], 1), dtype=np.float32)

    # Iterate through the offsets and generate patches for each offset
    for idx_i, offset in enumerate(offsets):
        patches = _array_to_patches_single(arr, tile_size, offset)

        offset_predictions = np.empty((patches.shape[0], patches.shape[1], patches.shape[2], test_shape[-1]), dtype=arr.dtype)
        offset_weights = np.empty((patches.shape[0], patches.shape[1], patches.shape[2], 1), dtype=np.float32)
        
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
                weights = np.ones((tile_size, tile_size, 1), dtype=np.float32)
                weights = np.repeat(weights[np.newaxis, ...], batch_size, axis=0)

            offset_predictions[idx_start:idx_end, ...] = prediction
            offset_weights[idx_start:idx_end, ...] = weights

        predictions[idx_i, :, :, :] = _patches_to_array_single(offset_predictions, (arr.shape[0], arr.shape[1], test_shape[-1]), tile_size, offset)
        predictions_weights[idx_i, :, :, :] = _patches_to_array_single(offset_weights, (arr.shape[0], arr.shape[1], 1), tile_size, offset, 0.0)

    # Merge the predictions
    if merge_method == "mad":
        predictions = _merge_weighted_mad(predictions, predictions_weights)
    elif merge_method == "median":
        predictions = _merge_weighted_median(predictions, predictions_weights)
    elif merge_method in ["mean", "average", "avg"]:
        predictions = _merge_weighted_average(predictions, predictions_weights)
    elif merge_method == "mode":
        predictions = _merge_weighted_mode(predictions, predictions_weights)
    elif merge_method == "max":
        predictions = _merge_weighted_minmax(predictions, predictions_weights, "max")
    elif merge_method == "min":
        predictions = _merge_weighted_minmax(predictions, predictions_weights, "min")
    elif merge_method == "olympic1":
        predictions = _merge_weighted_olympic(predictions, predictions_weights, 1)
    elif merge_method == "olympic2":
        predictions = _merge_weighted_olympic(predictions, predictions_weights, 2)

    if not channel_last:
        predictions = channel_last_to_first(predictions)

    return predictions


def predict_array_pixel(
    arr: np.ndarray,
    callback: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """Predicts an array pixel by pixel.

    Args:
        arr (np.ndarray): A numpy array to be divided into patches.
        callback (function): The callback function to be used for prediction. The callback function
            must take a numpy array as input and return a numpy array as output.
    """
    assert len(arr.shape) == 3, "Array must be 3D"

    reshaped = arr.reshape((arr.shape[0] * arr.shape[1], arr.shape[2]))
    predicted = callback(reshaped)
    predicted = predicted.reshape((arr.shape[0], arr.shape[1], predicted.shape[-1]))

    return predicted
