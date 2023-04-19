"""
This module contains utility functions for augmenting images that are
suited to remote sensing imagery.
"""
# Standard library
from typing import Tuple, List, Union

# External
import numpy as np
from numba import jit, prange

@jit(nopython=True, nogil=True, cache=True)
def is_integer_dtype(dtype: np.dtype) -> bool:
    """
    Check if a dtype is an integer type 
    
    Args:
        dtype (np.dtype): The dtype to check.

    Returns:
        bool: True if the dtype is an integer type.
    """
    return dtype.kind in ('i', 'u')


# These should have .astype(dtype, copy=False) to avoid copying the data
# But this is currently not supported by numba.
@jit(nopython=True, nogil=True, cache=True)
def fit_data_to_dtype(
    data: np.ndarray,
    dtype: np.dtype,
) -> np.ndarray:
    """
    Fit data to a dtype. If the dtype is an integer type, the data will be
        rounded to the nearest integer. If the dtype is a float type, the data
        will be clipped to the min and max values of the dtype.
    
    Args:
        data (np.ndarray): The data to fit to the dtype.
        dtype (np.dtype): The dtype to fit the data to.

    Returns:
        np.ndarray: The data fitted to the dtype.
    """
    max_value = np.iinfo(dtype).max
    min_value = np.iinfo(dtype).min

    if is_integer_dtype(dtype):
        return np.rint(
            np.clip(data, min_value, max_value)
        ).astype(dtype)

    return np.clip(
        data,
        min_value,
        max_value,
    ).astype(dtype)


@jit(nopython=True, nogil=True, cache=True, fastmath=True, parallel=True)
def feather_box_2d(
    array: np.ndarray,
    bbox: Union[List[int], Tuple[int]],
    feather_dist: int = 3,
) -> np.ndarray:
    """
    Feather a box into an array (2D). Box should be the original box
        buffered by feather_dist.

    Args:
        array_whole (np.ndarray): The array containing the box.
        bbox (Union[List[int], Tuple[int]]): The box.
            the bbox should be in the form [x_min, x_max, y_min, y_max].
    
    Keyword Args:
        feather_dist (int=3): The distance to feather the box.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The featherweights for the array and the bbox.
    """
    # Get the bbox
    x_min, x_max, y_min, y_max = bbox

    kernel_size = int((feather_dist * 2) + 1)
    n_offsets = kernel_size ** 2
    max_dist = (np.sqrt(2) / 2) + feather_dist

    feather_offsets = np.zeros((n_offsets, 2), dtype=np.int64)

    within_circle = 0
    for i in prange(-feather_dist, feather_dist + 1):
        for j in prange(-feather_dist, feather_dist + 1):
            dist = np.sqrt(i ** 2 + j ** 2)
            if dist <= max_dist:
                feather_offsets[within_circle][0] = i
                feather_offsets[within_circle][1] = j
                within_circle += 1

    feather_offsets = feather_offsets[:within_circle]
    n_offsets = feather_offsets.shape[0]

    x_min = max(0, x_min - feather_dist)
    x_max = min(x_max + 1 + feather_dist, array.shape[1])
    y_min = max(0, y_min - feather_dist)
    y_max = min(y_max + 1 + feather_dist, array.shape[0])

    feather_weights_array = np.ones_like(array, dtype=np.float32)

    # Feather the box
    for y in prange(y_min, y_max + 1):
        for x in prange(x_min, x_max + 1):
            total_possible = n_offsets
            within_bbox = 0
            for offset in feather_offsets:
                x_offset, y_offset = offset
                x_new = x + x_offset
                y_new = y + y_offset

                if y_new < 0 or y_new >= array.shape[0]:
                    total_possible -= 1
                    continue

                if x_new < 0 or x_new >= array.shape[1]:
                    total_possible -= 1
                    continue

                if x_new - feather_dist >= x_min and x_new + feather_dist <= x_max and y_new - feather_dist >= y_min and y_new + feather_dist <= y_max:
                    within_bbox += 1

            weights_box = within_bbox / total_possible
            feather_weights_array[y, x] = 1 - weights_box

    return feather_weights_array


@jit(nopython=True, nogil=True, cache=True, fastmath=True, inline='always')
def rotate_90(
    arr: np.ndarray,
    channel_last: bool = True
) -> np.ndarray:
    """ Rotate a 3D array 90 degrees clockwise.
    
    Args:
        arr (np.ndarray): The array to rotate.
    
    Keyword Args:
        channel_last (bool=True): Whether the last axis is the channel axis.

    Returns:
        np.ndarray: The rotated array.
    """
    if channel_last:
        return arr[::-1, :, :].transpose(1, 0, 2) # (H, W, C)

    return arr[:, ::-1, :].transpose(0, 2, 1) # (C, H, W)


@jit(nopython=True, nogil=True, cache=True, fastmath=True, inline='always')
def rotate_180(
    arr: np.ndarray,
    channel_last: bool = True
) -> np.ndarray:
    """ Rotate a 3D array 180 degrees clockwise.

    Args:
        arr (np.ndarray): The array to rotate.
    
    Keyword Args:
        channel_last (bool=True): Whether the last axis is the channel axis.

    Returns:
        np.ndarray: The rotated array.
    """
    if channel_last:
        return arr[::-1, ::-1, :]  # (H, W, C)

    return arr[:, ::-1, ::-1] # (C, H, W)


@jit(nopython=True, nogil=True, cache=True, fastmath=True, inline='always')
def rotate_270(
    arr: np.ndarray,
    channel_last: bool = True
) -> np.ndarray:
    """ 
    Rotate a 3D image array 270 degrees clockwise.

    Args:
        arr (np.ndarray): The array to rotate.

    Keyword Args:
        channel_last (bool=True): Whether the last axis is the channel axis.

    Returns:
        np.ndarray: The rotated array.
    """
    if channel_last:
        return arr[:, ::-1, :].transpose(1, 0, 2) # (H, W, C)

    return arr[:, :, ::-1].transpose(0, 2, 1) # (C, H, W)


@jit(nopython=True, nogil=True, cache=True, fastmath=True, inline='always')
def rotate_arr(
    arr: np.ndarray,
    k: int,
    channel_last: bool = True,
) -> np.ndarray:
    """ Rotate an array by 90 degrees intervals clockwise.
    
    Args:
        arr (np.ndarray): The array to rotate.
        k (int): The number of 90 degree intervals to rotate by.
            1 for 90 degrees, 2 for 180 degrees, 3 for 270 degrees. 0 for no rotation.

    Keyword Args:
        channel_last (bool=True): Whether the last axis is the channel axis.

    Returns:
        np.ndarray: The rotated array.
    """
    if k == 0:
        return arr.copy()

    view = arr

    if k == 1:
        view = rotate_90(arr, channel_last=channel_last)
    elif k == 2:
        view = rotate_180(arr, channel_last=channel_last)
    elif k == 3:
        view = rotate_270(arr, channel_last=channel_last)

    return view.copy()


@jit(nopython=True, nogil=True, cache=True, fastmath=True, inline='always')
def mirror_horizontal(
    arr: np.ndarray,
    channel_last: bool = True
) -> np.ndarray:
    """ Mirror a 3D array horizontally.
    
    Args:
        arr (np.ndarray): The array to mirror.
    
    Keyword Args:
        channel_last (bool=True): Whether the last axis is the channel axis.

    Returns:
        np.ndarray: The mirrored array.
    """
    if channel_last:
        return arr[:, ::-1, :] # (H, W, C)

    return arr[:, :, ::-1] # (C, H, W)


@jit(nopython=True, nogil=True, cache=True, fastmath=True, inline='always')
def mirror_vertical(
    arr: np.ndarray,
    channel_last: bool = True
) -> np.ndarray:
    """ Mirror a 3D array vertically.

    Args:
        arr (np.ndarray): The array to mirror.

    Keyword Args:
        channel_last (bool=True): Whether the last axis is the channel axis.
    
    Returns:
        np.ndarray: The mirrored array.
    """
    if channel_last:
        return arr[::-1, :, :] # (H, W, C)

    return arr[:, ::-1, :] # (C, H, W)


@jit(nopython=True, nogil=True, cache=True, fastmath=True, inline='always')
def mirror_horisontal_vertical(
    arr: np.ndarray,
    channel_last: bool = True
) -> np.ndarray:
    """ Mirror a 3D array horizontally and vertically.

    Args:
        arr (np.ndarray): The array to mirror.

    Keyword Args:
        channel_last (bool=True): Whether the last axis is the channel axis.

    Returns:
        np.ndarray: The mirrored array.
    """
    if channel_last:
        return arr[::-1, ::-1, :] # (H, W, C)

    return arr[:, ::-1, ::-1] # (C, H, W)


@jit(nopython=True, nogil=True, cache=True, fastmath=True, inline='always')
def mirror_arr(
    arr: np.ndarray,
    k: int,
    channel_last: bool = True,
) -> np.ndarray:
    """ Mirror an array horizontally and/or vertically.

    Args:
        arr (np.ndarray): The array to mirror.
        k (int): 1 for horizontal, 2 for vertical, 3 for both, 0 for no mirroring.
    
    Keyword Args:
        channel_last (bool=True): Whether the last axis is the channel axis.
    """
    if k == 0:
        return arr.copy()

    view = arr

    if k == 1:
        view = mirror_horizontal(arr, channel_last=channel_last)
    elif k == 2:
        view = mirror_vertical(arr, channel_last=channel_last)
    elif k == 3:
        view = mirror_horisontal_vertical(arr, channel_last=channel_last)

    return view.copy()
