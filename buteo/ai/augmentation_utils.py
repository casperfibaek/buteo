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
def simple_blur_kernel_2d_3x3() -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a 2D blur kernel.

    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1],

    Returns:
        Tuple[np.ndarray, np.ndarray]: The offsets and weights.
    """
    offsets = np.array([
        [ 1, -1], [ 1, 0], [ 1, 1],
        [ 0, -1], [ 0, 0], [ 0, 1],
        [-1, -1], [-1, 0], [-1, 1],
    ])

    weights = np.array([
        0.08422299, 0.12822174, 0.08422299,
        0.12822174, 0.1502211 , 0.12822174,
        0.08422299, 0.12822174, 0.08422299,
    ], dtype=np.float32)

    return offsets, weights


@jit(nopython=True, nogil=True, cache=True, fastmath=True, inline='always')
def simple_unsharp_kernel_2d_3x3(
    strength: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a 2D unsharp kernel.

    baseweights:
        0.09911165, 0.15088834, 0.09911165,
        0.15088834, 0.        , 0.15088834,
        0.09911165, 0.15088834, 0.09911165,

    Returns:
        Tuple[np.ndarray, np.ndarray]: The offsets and weights.
    """
    offsets = np.array([
        [ 1, -1], [ 1, 0], [ 1, 1],
        [ 0, -1], [ 0, 0], [ 0, 1],
        [-1, -1], [-1, 0], [-1, 1],
    ])

    base_weights = np.array([
        0.09911165, 0.15088834, 0.09911165,
        0.15088834, 0.        , 0.15088834,
        0.09911165, 0.15088834, 0.09911165,
    ], dtype=np.float32)

    weights = base_weights * strength
    middle_weight = np.sum(weights) + 1.0
    weights *= -1.0
    weights[4] = middle_weight

    return offsets, weights


def simple_shift_kernel_2d(
    x_offset: float,
    y_offset: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a 2D shift kernel. Useful for either aligning rasters at the sub-pixel
    level or for shifting a raster by a whole pixel while keeping the bbox.
    Can be used to for an augmentation, where channel misalignment is simulated.

    Args:
        x_offset (float): The x offset.
        y_offset (float): The y offset.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The offsets and weights.
    """
    if x_offset == 0.0 and y_offset == 0.0:
        offsets = np.array([[0, 0]], dtype=np.int64)
        weights = np.array([1.0], dtype=np.float32)

        return offsets, weights

    y0 = [int(np.floor(y_offset)), int(np.ceil(y_offset))] if y_offset != 0 else [0, 0]
    x0 = [int(np.floor(x_offset)), int(np.ceil(x_offset))] if x_offset != 0 else [0, 0]

    if x_offset == 0.0 or x_offset % 1 == 0.0:
        offsets = np.zeros((2, 2), dtype=np.int64)
        weights = np.zeros(2, dtype=np.float32)

        offsets[0] = [int(x_offset) if x_offset % 1 == 0.0 else 0, y0[0]]
        offsets[1] = [int(x_offset) if x_offset % 1 == 0.0 else 0, y0[1]]

        weights[0] = y_offset - y0[0]
        weights[1] = 1 - weights[0]

    elif y_offset == 0.0 or y_offset % 1 == 0.0:
        offsets = np.zeros((2, 2), dtype=np.int64)
        weights = np.zeros(2, dtype=np.float32)

        offsets[0] = [x0[0], int(y_offset) if y_offset % 1 == 0.0 else 0]
        offsets[1] = [x0[1], int(y_offset) if y_offset % 1 == 0.0 else 0]

        weights[0] = x_offset - x0[0]
        weights[1] = 1 - weights[0]

    else:
        offsets = np.zeros((4, 2), dtype=np.int64)
        weights = np.zeros(4, dtype=np.float32)

        offsets[0] = [x0[0], y0[0]]
        offsets[1] = [x0[0], y0[1]]
        offsets[2] = [x0[1], y0[0]]
        offsets[3] = [x0[1], y0[1]]

        weights[0] = (1 - (x_offset - offsets[0][0])) * (1 - (y_offset - offsets[0][1]))
        weights[1] = (1 - (x_offset - offsets[1][0])) * (1 + (y_offset - offsets[1][1]))
        weights[2] = (1 + (x_offset - offsets[2][0])) * (1 - (y_offset - offsets[2][1]))
        weights[3] = (1 + (x_offset - offsets[3][0])) * (1 + (y_offset - offsets[3][1]))

    return offsets, weights


# Sharpen still offsets everything, blur is better.
@jit(nopython=True, nogil=True, cache=True, fastmath=True, parallel=True)
def convolution_simple(
    array: np.ndarray,
    offsets: np.ndarray,
    weights: np.ndarray,
    intensity: float = 1.0,
):
    """
    Convolve a kernel with an array using a simple method.

    Args:
        array (np.ndarray): The array to convolve.
        offsets (np.ndarray): The offsets of the kernel.
        weights (np.ndarray): The weights of the kernel.
    
    Keyword Args:
        intensity (float=1.0): The intensity of the convolution. If
            1.0, the convolution is applied as is. If 0.5, the
            convolution is applied at half intensity.

    Returns:
        np.ndarray: The convolved array.
    """
    result = np.empty_like(array, dtype=np.float32)

    if intensity <= 0.0:
        return array.astype(np.float32)

    for col in prange(array.shape[0]):
        for row in prange(array.shape[1]):
            result_value = 0.0
            for i in range(offsets.shape[0]):
                new_col = col + offsets[i, 0]
                new_row = row + offsets[i, 1]

                if new_col < 0:
                    new_col = 0
                elif new_col >= array.shape[0]:
                    new_col = array.shape[0] - 1

                if new_row < 0:
                    new_row = 0
                elif new_row >= array.shape[1]:
                    new_row = array.shape[1] - 1

                result_value += array[new_col, new_row] * weights[i]

            result[col, row] = result_value

    if intensity < 1.0:
        result *= intensity
        array *= (1.0 - intensity)
        result += array

    return result


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
