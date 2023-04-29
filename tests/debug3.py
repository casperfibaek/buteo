"""
### Perform convolutions on arrays.  ###
"""

# Standard Library
from typing import List, Union, Optional

# External
import numpy as np
from numba import jit, prange

# Internal
from buteo.utils.core_utils import type_check
from buteo.array.convolution_funcs import _hood_to_value


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
    type_check(arr, [np.ndarray], "arr")
    type_check(pad_size, [int], "pad_size")
    type_check(method, [str], "method")

    assert pad_size >= 0, "pad_size must be a non-negative integer"
    method = method.lower()
    assert method in ["same", "edge", "constant"], "method must be one of ['same', 'edge', 'constant']"

    if method in ["same", "edge"]:
        padded_view = np.pad(
            arr,
            pad_width=((pad_size, pad_size), (pad_size, pad_size), (0, 0)),
            mode='edge',
        )
    elif method in "constant":
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
    offsets: List[List[int]],
    weights: List[float],
    method: int = 1,
    nodata: bool = False,
    nodata_value: float = -9999.9,
    value: Optional[Union[int, float]] = None,
) -> np.ndarray:
    """ Internal function for convolving a 2D array. """
    result = np.zeros((arr.shape[0], arr.shape[1]), dtype="float32")
    hood_size = len(offsets)
    weights_total = np.sum(weights)

    for idx_y in prange(arr.shape[0]):
        for idx_x in prange(arr.shape[1]):
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

                hood_values[hood_count] = arr[hood_y, hood_x]
                hood_weights[hood_count] = weights[idx_h]
                hood_count += 1

                if offsets[idx_h][0] == 0 and offsets[idx_h][1] == 0 and offsets[idx_h][2] == 0:
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
                value,
            )

    return result


def _convolve_array_channels(
    arr: np.ndarray,
    method: int = 1,
    nodata: bool = False,
    nodata_value: float = -9999.9,
    value: Optional[Union[int, float]] = None,
) -> np.ndarray:
    """ Internal function for convoling a 3D array along its channels. """
    result = np.zeros((arr.shape[0], arr.shape[1], 1), dtype="float32")

    hood_size = arr.shape[2]
    hood_weights = np.ones(hood_size, dtype="float32")
    center_idx = int(hood_size / 2)

    for idx_y in prange(arr.shape[0]):
        for idx_x in prange(arr.shape[1]):
            hood_count = 0
            hood_values = np.zeros(hood_size, dtype="float32")

            for idx_c in range(hood_size):
                value = arr[idx_y, idx_x, idx_c]
                if nodata and value == nodata_value:
                    continue
                hood_values[hood_count] = value
                hood_count += 1

            if hood_count == 0:
                result[idx_y, idx_x, 0] = nodata_value
                continue

            hood_values = hood_values[:hood_count]
            hood_weights = hood_weights[:hood_count]

            result[idx_y, idx_x, 0] = _hood_to_value(
                method,
                hood_values,
                hood_weights,
                nodata_value,
                center_idx,
                value,
            )

    return result


def convolve_array_channels(
    arr: np.ndarray,
    method: int = 1,
    nodata: bool = False,
    nodata_value: float = -9999.9,
    value: Optional[Union[int, float]] = None,
) -> np.ndarray:
    """
    Convolves a 3D array along its channels.
    Useful for 'collapsing' a 3D array into a 2D array.

    Parameters
    ----------
    arr : np.ndarray
        A 3D array.

    method : int, optional
        The method to use for convolving the array.

        The following methods are valid:
        ```text
            1. sum
            2. mode
            3. max/dilate
            4. min/erode
            5. contrast
            6. median
            7. std
            8. mad
            9. z_score
            10. z_score_mad
            11. sigma_lee
            12. quantile
            13. occurrances
            14. feather
            15. roughness
            16. roughness_tri
            17. roughness_tpi
        ```

        Default: 1.
    
    nodata : bool, optional
        Whether to use nodata values in the convolution. Default: False.
    
    nodata_value : float, optional
        The nodata value to use in the convolution. Default: -9999.9.
    
    value : int or float, optional
        The value to use in the convolution. Default: None.

    Returns
    -------
    np.ndarray
        The convolved array.

    """
    type_check(arr, [np.ndarray], "arr")
    type_check(method, [int], "method")
    type_check(nodata, [bool], "nodata")
    type_check(nodata_value, [float], "nodata_value")
    type_check(value, [int, float, type(None)], "value")

    assert arr.ndim == 3, "arr must be a 3D array"

    if arr.shape[2] == 1:
        return arr

    return _convolve_array_channels(
        arr,
        method=method,
        nodata=nodata,
        nodata_value=nodata_value,
        value=value,
    )


@jit(nopython=True, parallel=True, nogil=False, fastmath=True, cache=True)
def convolve_array(
    arr: np.ndarray,
    offsets: List[List[int]],
    weights: List[float],
    method: int = 1,
    nodata: bool = False,
    nodata_value: float = -9999.9,
    value: Optional[Union[int, float]] = None,
) -> np.ndarray:
    """
    Convolve an array using a set of offsets and weights.

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
        ```text
            1. sum
            2. mode
            3. max/dilate
            4. min/erode
            5. contrast
            6. median
            7. std
            8. mad
            9. z_score
            10. z_score_mad
            11. sigma_lee
            12. quantile
            13. occurrances
            14. feather
            15. roughness
            16. roughness_tri
            17. roughness_tpi
        ```

    nodata : bool, optional
        If True, treat the nodata value as a valid value. Default: False.

    nodata_value : float, optional
        The nodata value to use when computing the result. Default: -9999.9.

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
    type_check(arr, [np.ndarray], "arr")
    type_check(offsets, [np.ndarray], "offsets")
    type_check(weights, [np.ndarray], "weights")
    type_check(method, [int], "method")
    type_check(nodata, [bool], "nodata")
    type_check(nodata_value, [float], "nodata_value")
    type_check(value, [int, float, type(None)], "value")

    assert len(offsets) == len(weights), "offsets and weights must be the same length"
    assert arr.ndim in [2, 3], "arr must be 2 or 3 dimensional"

    if arr.ndim == 2:
        return _convolve_array_2D(
            arr,
            offsets,
            weights,
            method,
            nodata,
            nodata_value,
            value,
        )

    result = np.zeros((arr.shape), dtype="float32")

    for idx_d in range(arr.shape[2]):
        result[:, :, idx_d] = _convolve_array_2D(
            arr[:, :, idx_d],
            offsets,
            weights,
            method,
            nodata,
            nodata_value,
            value,
        )

    return result
