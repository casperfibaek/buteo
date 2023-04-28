"""
### Perform convolutions on arrays.  ###
"""

# Standard Library
from typing import List, Tuple, Optional, Union

# External
import numpy as np
from numba import jit, prange

# Internal
from buteo.utils import core_utils
from buteo.array.convolution_funcs import _hood_to_value



_METHOD_ENUMS = {
    "sum": 1,
    "mode": 2,
    "max": 3,
    "dilate": 3,
    "min": 4,
    "erode": 4,
    "contrast": 5,
    "median": 6,
    "std": 7,
    "mad": 8,
    "z_score": 9,
    "z_score_mad": 10,
    "sigma_lee": 11,
    "quantile": 12,
    "occurrances": 13,
    "feather": 14,
    "roughness": 15,
    "roughness_tri": 16,
    "roughness_tpi": 17,
}


def pad_array(
    arr: np.ndarray,
    pad_size: int = 1,
    method: str = "same",
    constant_value: Union[float, int] = 0.0,
) -> np.ndarray:
    """
    Create a padded view of an array using SAME padding.

    Parameters
    ----------
    arr : numpy.ndarray
        The input array to be padded.

    pad_size : int, optional
        The number of padding elements to add to each side of the array.
        Default: 1.

    method : str, optional
        The padding method to use. Default: "same". Other options are
        "edge" and "constant".

    constant_value : int, optional
        The constant value to use when padding with "constant". Default: 0.

    Returns
    -------
    numpy.ndarray
        A padded view of the input array.

    Notes
    -----
    This function creates a padded view of an array using SAME padding, which
    adds padding elements to each side of the array so that the output shape
    is the same as the input shape. The amount of padding is determined by the
    `pad_size` parameter. The padding method can be one of three options: "same"
    (the default), "edge", or "constant". If "constant" padding is used, the
    `constant_value` parameter specifies the value to use.
    """
    core_utils.type_check(arr, [np.ndarray], "arr")
    core_utils.type_check(pad_size, [int], "pad_size")
    core_utils.type_check(method, [str], "method")

    assert pad_size >= 0, "pad_size must be a non-negative integer"
    assert method in ["same", "SAME", "edge", "EDGE"], "method must be one of ['same', 'SAME', 'constant', 'CONSTANT']"

    if method in ["same", "SAME"]:
        padded_view = np.pad(
            arr,
            pad_width=((pad_size, pad_size), (pad_size, pad_size), (0, 0)),
            mode='edge',
        )
    elif method in ["constant", "CONSTANT"]:
        padded_view = np.pad(
            arr,
            pad_width=((pad_size, pad_size), (pad_size, pad_size), (0, 0)),
            mode='constant',
            constant_values=constant_value,
        )

    return padded_view


@jit(nopython=True, parallel=True, nogil=False, fastmath=True, cache=True)
def _convolve_array(
    arr: np.ndarray,
    offsets: List[Tuple[int, int, int]],
    weights: List[float],
    method: int = 1,
    nodata: bool = False,
    nodata_value: float = -9999.9,
    normalise_edges: bool = True,
    value: Union[int, float, None] = None,
) -> np.ndarray:
    """
    Internal function. Convolve an array using a set of offsets and weights.

    Parameters
    ----------
    arr : numpy.ndarray
        The input array to be convolved.

    offsets : list of tuples
        The list of pixel offsets to use in the convolution. Each tuple should be in the
        format (row_offset, col_offset, depth_offset), where row_offset and col_offset
        are the row and column offsets from the center pixel, and depth_offset is the
        depth offset if the input array has more than two dimensions.

    weights : list of floats
        The list of weights to use in the convolution. The length of the weights list should
        be the same as the length of the offsets list.

    method : int, optional
        The convolution method to use. Default: 1.

    nodata : bool, optional
        If True, treat the nodata value as a valid value. Default: False.

    nodata_value : float, optional
        The nodata value to use when computing the result. Default: -9999.9.

    normalise_edges : bool, optional
        If True, normalise the edge pixels based on the number of valid pixels in the kernel.
        Default: True.

    value : int or float or None, optional
        The value to use for pixels where the kernel extends outside the input array.
        If None, use the edge value. Default: None.

    Returns
    -------
    numpy.ndarray
        The convolved array.

    Notes
    -----
    This function convolves an array using a set of offsets and weights. The function supports
    different convolution methods, including nearest, linear, and cubic. The function can also
    handle nodata values and cases where the kernel extends outside the input array.
    """
    result = np.zeros((arr.shape[0], arr.shape[1], arr.shape[2]), dtype="float32")
    hood_size = len(offsets)

    for idx_y in prange(0, arr.shape[0]):
        for idx_x in range(0, arr.shape[1]):
            for idx_z in range(0, arr.shape[2]):

                center_idx = 0
                hood_normalise = False
                hood_values = np.zeros(hood_size, dtype="float32")
                hood_weights = np.zeros(hood_size, dtype="float32")
                hood_count = 0

                if nodata and arr[idx_y, idx_x, idx_z] == nodata_value:
                    result[idx_y, idx_x, idx_z] = nodata_value
                    continue

                for idx_h in range(0, hood_size):
                    hood_x = idx_x + offsets[idx_h][0]
                    hood_y = idx_y + offsets[idx_h][1]
                    hood_z = idx_z + offsets[idx_h][2]

                    if hood_x < 0 or hood_x >= arr.shape[1]:
                        if normalise_edges:
                            hood_normalise = True
                        continue

                    if hood_y < 0 or hood_y >= arr.shape[0]:
                        if normalise_edges:
                            hood_normalise = True
                        continue

                    if hood_z < 0 or hood_z >= arr.shape[2]:
                        if normalise_edges:
                            hood_normalise = True
                        continue

                    if nodata and arr[hood_y, hood_x, hood_z] == nodata_value:
                        if normalise_edges:
                            hood_normalise = True
                        continue

                    hood_values[hood_count] = arr[hood_y, hood_x, hood_z]
                    hood_weights[hood_count] = weights[idx_h]
                    hood_count += 1

                    if offsets[idx_h][0] == 0 and offsets[idx_h][1] == 0 and offsets[idx_h][2] == 0:
                        center_idx = hood_count - 1

                if hood_count == 0:
                    result[idx_y, idx_x, idx_z] = nodata_value
                    continue

                hood_values = hood_values[:hood_count]
                hood_weights = hood_weights[:hood_count]

                if hood_normalise:
                    hood_weights /= np.sum(hood_weights)

                result[idx_y, idx_x, idx_z] = _hood_to_value(method, hood_values, hood_weights, nodata_value, center_idx, value)

    return result


@jit(nopython=True, nogil=True, cache=True, fastmath=True, parallel=True)
def convolve_array_simple_hwc(
    array: np.ndarray,
    offsets: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """
    Convolve a kernel with an array using a simple method.
    Array must be 3D and in channel last format. (height, width, channels)

    Parameters
    ----------
    array : numpy.ndarray
        The array to convolve.

    offsets : numpy.ndarray
        The offsets of the kernel.

    weights : numpy.ndarray
        The weights of the kernel.

    Returns
    -------
    numpy.ndarray
        The convolved array.

    Notes
    -----
    This function convolves a kernel with an array using a simple method. The function supports
    applying the convolution at a reduced intensity to achieve a blended effect.
    """
    result = np.zeros(array.shape, dtype=np.float32)

    for col in prange(array.shape[0]):
        for row in prange(array.shape[1]):
            for channel in prange(array.shape[2]):

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

                    result_value += array[new_col, new_row, channel] * weights[i]

                result[col, row, channel] = result_value

    return result


@jit(nopython=True, nogil=True, cache=True, fastmath=True, parallel=True)
def convolve_array_simple_chw(
    array: np.ndarray,
    offsets: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """
    Convolve a kernel with an array using a simple method.
    Array must be 3D and in channel first format. (channels, height, width)

    Parameters
    ----------
    array : numpy.ndarray
        The array to convolve.

    offsets : numpy.ndarray
        The offsets of the kernel.

    weights : numpy.ndarray
        The weights of the kernel.

    Returns
    -------
    numpy.ndarray
        The convolved array.

    Notes
    -----
    This function convolves a kernel with an array using a simple method. The function supports
    applying the convolution at a reduced intensity to achieve a blended effect.
    """
    result = np.zeros(array.shape, dtype=np.float32)

    for channel in prange(array.shape[0]):
        for col in prange(array.shape[1]):
            for row in prange(array.shape[2]):

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

                    result_value += array[channel, new_col, new_row] * weights[i]

                result[channel, col, row] = result_value

    return result


@jit(nopython=True, nogil=True, cache=True, fastmath=True, parallel=True)
def convolve_array_simple_2D(
    array: np.ndarray,
    offsets: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """
    Convolve a kernel with an array using a simple method.
    Array must be 2D (height, width).

    Parameters
    ----------
    array : numpy.ndarray
        The array to convolve.

    offsets : numpy.ndarray
        The offsets of the kernel.

    weights : numpy.ndarray
        The weights of the kernel.

    Returns
    -------
    numpy.ndarray
        The convolved array.

    Notes
    -----
    This function convolves a kernel with an array using a simple method. The function supports
    applying the convolution at a reduced intensity to achieve a blended effect.
    """
    result = np.zeros(array.shape, dtype=np.float32)

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

    return result


def convolve_array(
    arr: np.ndarray,
    offsets: np.ndarray,
    weights: np.ndarray,
    method: int = 1,
    nodata: bool = False,
    nodata_value: float = -9999.9,
    normalise_edges: bool = True,
    value: Union[int, float, None] = None,
) -> np.ndarray:
    """
    Convolve an image with a neighborhood function.

    Parameters
    ----------
    arr : numpy.ndarray
        The input array to convolve.

    offsets : list of tuples
        The list of pixel offsets to use in the convolution. Each tuple should be in the
        format (row_offset, col_offset, depth_offset), where row_offset and col_offset
        are the row and column offsets from the center pixel, and depth_offset is the
        depth offset if the input array has more than two dimensions.

    weights : list of floats
        The list of weights to use in the convolution. The length of the weights list should
        be the same as the length of the offsets list.

    method : int, optional
        The method to use for the convolution. The available options are:
            1: hood_sum
            2: hood_mode
            3: hood_max
            4: hood_min
            5: hood_contrast
            6: hood_quantile
            7: hood_standard_deviation
            8: hood_median_absolute_deviation
            9: hood_z_score
            10: hood_z_score_mad
            11: hood_sigma_lee
        Default: 1.

    nodata : bool, optional
        If True, treat the nodata value as a valid value. Default: False.

    nodata_value : float, optional
        The nodata value to use when computing the result. Default: -9999.9.

    normalise_edges : bool, optional
        If True, normalise the edge pixels based on the number of valid pixels in the kernel.
        Only relevant for border pixels. Use False if you are interested in the sum; otherwise,
        you likely want to use True. Default: True.

    value : int or float or None, optional
        If not None, the value to use for the convolution depending on the method specified.
        Default: None.

    Returns
    -------
    numpy.ndarray
        The convolved array.

    Notes
    -----
    This function convolves an array with a neighborhood function using a set of pixel
    offsets and weights. The function supports various convolution methods, including
    sum, mode, maximum, minimum, contrast, quantile, standard deviation, median absolute
    deviation, z-score, and sigma-lee. The function can also handle nodata values and cases
    where the kernel extends outside the input array.
    """
    core_utils.type_check(arr, [np.ndarray], "arr")
    core_utils.type_check(offsets, [np.ndarray], "offsets")
    core_utils.type_check(weights, [np.ndarray], "weights")
    core_utils.type_check(method, [int], "method")
    core_utils.type_check(nodata, [bool], "nodata")
    core_utils.type_check(nodata_value, [float], "nodata_value")
    core_utils.type_check(normalise_edges, [bool], "normalise_edges")
    core_utils.type_check(value, [int, float, type(None)], "value")

    assert len(offsets) == len(weights), "offsets and weights must be the same length"
    assert method in range(1, len(_METHOD_ENUMS)), "method must be between 1 and 11"
    assert arr.ndim in [2, 3], "arr must be 2 or 3 dimensional"

    if value is None:
        value = 0.5

    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]

    return _convolve_array(
        arr,
        offsets,
        weights,
        method=method,
        nodata=nodata,
        nodata_value=nodata_value,
        normalise_edges=normalise_edges,
        value=value,
    )
