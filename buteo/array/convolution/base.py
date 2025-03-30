"""### Base convolution operations for arrays. ###"""

# Standard library
from typing import Union, Optional

# External
import numpy as np
from numba import jit, prange

# Internal
from buteo.utils.utils_base import _type_check
from buteo.array.convolution.funcs import _hood_to_value


def pad_array(
    arr: np.ndarray,
    pad_size: int = 1,
    method: str = "same",
    constant_value: Union[float, int] = 0.0,
) -> np.ndarray:
    """Create a padded view of an array using SAME padding.

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
    _type_check(arr, [np.ndarray], "arr")
    _type_check(pad_size, [int], "pad_size")
    _type_check(method, [str], "method")

    assert pad_size >= 0, "pad_size must be a non-negative integer"
    method = method.lower()
    assert method in ["same", "edge", "constant"], "method must be one of ['same', 'edge', 'constant']"

    # Initialize padded_view with a default value
    padded_view = None
    
    if method in ["same", "edge"]:
        padded_view = np.pad(
            arr,
            pad_width=((pad_size, pad_size), (pad_size, pad_size), (0, 0)),
            mode='edge',
        )
    else:  # method == "constant"
        padded_view = np.pad(
            arr,
            pad_width=((pad_size, pad_size), (pad_size, pad_size), (0, 0)),
            mode='constant',
            constant_values=constant_value,
        )

    return padded_view


@jit(nopython=True, parallel=True, nogil=False, fastmath=True, cache=True)
def _convolve_array_2D(
    arr: np.ndarray,
    offsets: np.ndarray,
    weights: np.ndarray,
    method: int = 1,
    nodata: bool = False,
    nodata_value: float = -9999.9,
    func_value: Union[int, float] = 0.5,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Internal function for convolving a 2D array.
    Input should be float32.
    """
    result = np.zeros((arr.shape[0], arr.shape[1]), dtype="float32")
    hood_size = len(offsets)
    weights_total = np.sum(weights)

    nodata_value = float(nodata_value)
    mask_2d = mask[:, :, 0] if mask is not None else np.ones((arr.shape[0], arr.shape[1]), dtype=np.uint8)

    for idx_y in prange(arr.shape[0]):
        for idx_x in range(arr.shape[1]):
            if mask_2d[idx_y, idx_x] == 0:
                result[idx_y, idx_x] = arr[idx_y, idx_x]
                continue

            hood_values = np.zeros(hood_size, dtype="float32")
            hood_weights = np.zeros(hood_size, dtype="float32")
            hood_normalise = False
            hood_center = 0
            hood_count = 0

            if nodata and arr[idx_y, idx_x] == nodata_value:
                result[idx_y, idx_x] = nodata_value
                continue

            for idx_h in range(0, hood_size):
                hood_y = idx_y + offsets[idx_h][0]
                hood_x = idx_x + offsets[idx_h][1]

                if hood_y < 0 or hood_y >= arr.shape[0]:
                    hood_normalise = True
                    continue

                if hood_x < 0 or hood_x >= arr.shape[1]:
                    hood_normalise = True
                    continue

                if nodata and arr[hood_y, hood_x] == nodata_value:
                    hood_normalise = True
                    continue

                hood_val = arr[hood_y, hood_x]
                if method == 9 and hood_val == func_value:
                    hood_values[hood_count] = 0.0
                    hood_weights[hood_count] = 0.0
                    hood_normalise = True
                else:
                    hood_values[hood_count] = hood_val
                    hood_weights[hood_count] = weights[idx_h]
                    hood_count += 1

                if offsets[idx_h][0] == 0 and offsets[idx_h][1] == 0:
                    hood_center = hood_count - 1

            if hood_count == 0:
                result[idx_y, idx_x] = nodata_value
                continue

            hood_values = hood_values[:hood_count]
            hood_weights = hood_weights[:hood_count]

            if hood_normalise:
                hood_weights /= np.sum(hood_weights)
                hood_weights *= weights_total

            result[idx_y, idx_x] = _hood_to_value(
                method,
                hood_values,
                hood_weights,
                nodata_value,
                hood_center,
                func_value,
            )

    return result


@jit(nopython=True, nogil=True, cache=True, fastmath=True, parallel=True)
def convolve_array_simple(
    arr: np.ndarray,
    offsets: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Convolve a kernel with an array using a simple method.
    Array must be 2D (height, width).
    The function ignores nodata values and pads the array with 'same'.
    It only supports 'summation' method.

    Parameters
    ----------
    arr : numpy.ndarray
        The array to convolve.

    offsets : numpy.ndarray
        The offsets of the kernel.

    weights : numpy.ndarray
        The weights of the kernel.

    Returns
    -------
    numpy.ndarray
        The convolved array.
    """
    result = np.zeros(arr.shape, dtype=np.float32)

    for col in prange(arr.shape[0]):
        for row in range(arr.shape[1]):

            result_value = 0.0
            for i in range(offsets.shape[0]):
                new_col = col + offsets[i, 0]
                new_row = row + offsets[i, 1]

                if new_col < 0:
                    new_col = 0
                elif new_col >= arr.shape[0]:
                    new_col = arr.shape[0] - 1

                if new_row < 0:
                    new_row = 0
                elif new_row >= arr.shape[1]:
                    new_row = arr.shape[1] - 1

                result_value += arr[new_col, new_row] * weights[i]

            result[col, row] = result_value

    return result


@jit(nopython=True, parallel=True, nogil=False, fastmath=True, cache=True)
def _convolve_array_channels_HWC(
    arr: np.ndarray,
    method: int = 1,
    nodata: bool = False,
    nodata_value: float = -9999.9,
    func_value: Union[int, float] = 0.5,
) -> np.ndarray:
    """Internal function for convoling a 3D array along its channels.
    Input should be float32. Channel-last version.
    """
    result = np.zeros((arr.shape[0], arr.shape[1], 1), dtype="float32")

    hood_size = arr.shape[2]
    center_idx = int(hood_size / 2)

    for idx_y in prange(arr.shape[0]):
        for idx_x in range(arr.shape[1]):
            hood_values = np.zeros(hood_size, dtype="float32")
            hood_weights = np.ones(hood_size, dtype="float32")
            hood_count = 0

            for idx_c in range(hood_size):
                if nodata and arr[idx_y, idx_x, idx_c] == nodata_value:
                    continue

                hood_values[hood_count] = arr[idx_y, idx_x, idx_c]
                hood_count += 1

            if hood_count == 0:
                result[idx_y, idx_x, 0] = nodata_value
                continue

            hood_values = hood_values[:hood_count]
            hood_weights = hood_weights[:hood_count] / hood_weights[:hood_count].sum()

            result[idx_y, idx_x, 0] = _hood_to_value(
                method,
                hood_values,
                hood_weights,
                nodata_value,
                center_idx,
                func_value,
            )

    return result


@jit(nopython=True, parallel=True, nogil=False, fastmath=True, cache=True)
def _convolve_array_channels_CHW(
    arr: np.ndarray,
    method: int = 1,
    nodata: bool = False,
    nodata_value: float = -9999.9,
    func_value: Union[int, float] = 0.5,
) -> np.ndarray:
    """Internal function for convoling a 3D array along its channels.
    Input should be float32. Channel-first version.
    """
    result = np.zeros((1, arr.shape[1], arr.shape[2]), dtype="float32")

    hood_size = arr.shape[0]
    center_idx = int(hood_size / 2)

    for idx_y in prange(arr.shape[1]):
        for idx_x in range(arr.shape[2]):
            hood_count = 0
            hood_values = np.zeros(hood_size, dtype="float32")
            hood_weights = np.ones(hood_size, dtype="float32")

            for idx_c in range(hood_size):
                if nodata and arr[idx_c, idx_y, idx_x] == nodata_value:
                    continue
                hood_values[hood_count] = arr[idx_c, idx_y, idx_x]
                hood_count += 1

            if hood_count == 0:
                result[0, idx_y, idx_x] = nodata_value
                continue

            hood_values = hood_values[:hood_count]
            hood_weights = hood_weights[:hood_count] / hood_weights[:hood_count].sum()

            result[0, idx_y, idx_x] = _hood_to_value(
                method,
                hood_values,
                hood_weights,
                nodata_value,
                center_idx,
                func_value,
            )

    return result


def convolve_array_channels(
    arr: np.ndarray,
    method: int = 1,
    nodata: bool = False,
    nodata_value: float = -9999.9,
    func_value: Union[int, float] = 0.5,
    channel_last: bool = True,
) -> np.ndarray:
    """Convolves a 3D array along its channels.
    Useful for 'collapsing' a 3D array into a 2D array.

    Parameters
    ----------
    arr : np.ndarray
        A 3D array.

    method : int, optional
        The method to use for convolving the array.

        The following methods are valid:
        1. sum
        2. max
        3. min
        4. mean
        5. median
        6. variance
        7. standard deviation
        8. contrast
        9. mode
        10. median absolute deviation (mad)
        11. z-score
        12. z-score (mad)
        13. sigma lee
        14. quantile
        15. occurances
        16. feather
        17. roughness
        18. roughness tri
        19. roughness tpi

        Default: 1.

    nodata : bool, optional
        Whether to use nodata values in the convolution. Default: False.

    nodata_value : float, optional
        The nodata value to use in the convolution. Default: -9999.9.

    func_value : int or float, optional
        The value to use in the convolution. Default: 0.5.

    channel_last : bool, optional
        Whether the channels are the last axis in the array. Default: True.

    Returns
    -------
    np.ndarray
        The convolved array.
    """
    _type_check(arr, [np.ndarray], "arr")
    _type_check(method, [int], "method")
    _type_check(nodata, [bool], "nodata")
    _type_check(nodata_value, [float], "nodata_value")
    _type_check(func_value, [int, float], "value")

    assert arr.ndim == 3, "arr must be a 3D array"
    assert method in range(1, 20), "method must be between 1 and 19"

    if channel_last and arr.shape[2] == 1:
        return arr
    elif not channel_last and arr.shape[0] == 1:
        return arr

    arr = arr.astype(np.float32, copy=False)

    if channel_last:
        return _convolve_array_channels_HWC(
            arr,
            method=method,
            nodata=nodata,
            nodata_value=nodata_value,
            func_value=func_value,
        )
    else:
        return _convolve_array_channels_CHW(
            arr,
            method=method,
            nodata=nodata,
            nodata_value=nodata_value,
            func_value=func_value,
        )


def convolve_array(
    arr: np.ndarray,
    offsets: np.ndarray,
    weights: np.ndarray,
    method: int = 1,
    nodata: bool = False,
    nodata_value: float = -9999.9,
    func_value: Union[int, float] = 0.5,
    channel_last: bool = True,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Convolve an array using a set of offsets and weights.
    Array can be 2D or 3D.

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
        The following methods are valid:
        1. sum
        2. max
        3. min
        4. mean
        5. median
        6. variance
        7. standard deviation
        8. contrast
        9. mode
        10. median absolute deviation (mad)
        11. z-score
        12. z-score (mad)
        13. sigma lee
        14. quantile
        15. occurances
        16. feather
        17. roughness
        18. roughness tri
        19. roughness tpi

    nodata : bool, optional
        If True, treat the nodata value as a valid value. Default: False.

    nodata_value : float, optional
        The nodata value to use when computing the result. Default: -9999.9.

    func_value : int or float, optional
        The value to use for pixels where the kernel extends outside the input array.
        If None, use the edge value. Default: 0.5.

    channel_last : bool, optional
        Whether the channels are the last axis in the array. Default: True.

    mask : np.ndarray, optional
        A mask array with the same shape as arr. If provided, only pixels where
        mask is non-zero will be processed. Default: None.

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
    _type_check(arr, [np.ndarray], "arr")
    _type_check(offsets, [np.ndarray], "offsets")
    _type_check(weights, [np.ndarray], "weights")
    _type_check(method, [int], "method")
    _type_check(nodata, [bool], "nodata")
    _type_check(nodata_value, [float], "nodata_value")
    _type_check(func_value, [int, float, type(None)], "value")
    _type_check(channel_last, [bool], "channel_last")

    assert len(offsets) == len(weights), "offsets and weights must be the same length"
    assert arr.ndim in [2, 3], "arr must be 2 or 3 dimensional"
    assert method in range(1, 20), "method must be between 1 and 19"

    arr = arr.astype(np.float32, copy=False)

    if mask is None:
        mask = np.ones(arr.shape, dtype=np.uint8)

    if arr.ndim == 2:
        return _convolve_array_2D(
            arr,
            offsets,
            weights,
            method=method,
            nodata=nodata,
            nodata_value=nodata_value,
            func_value=func_value,
            mask=mask,
        )

    result = np.zeros((arr.shape), dtype="float32")

    if channel_last:
        for idx_d in range(arr.shape[2]):
            result[:, :, idx_d] = _convolve_array_2D(
                arr[:, :, idx_d],
                offsets,
                weights,
                method=method,
                nodata=nodata,
                nodata_value=nodata_value,
                func_value=func_value,
                mask=mask,
            )
    else:
        for idx_d in range(arr.shape[0]):
            result[idx_d, :, :] = _convolve_array_2D(
                arr[idx_d, :, :],
                offsets,
                weights,
                method=method,
                nodata=nodata,
                nodata_value=nodata_value,
                func_value=func_value,
                mask=mask,
            )

    return result
