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
from buteo.array import convolution_funcs



@jit(nopython=True, nogil=True, cache=True, fastmath=True, inline='always')
def _simple_blur_kernel_2d_3x3() -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a 2D blur kernel.

    The kernel has the following form:

        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1],

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple of two numpy arrays. The first array contains the (x, y) offsets
        of the kernel values. The second array contains the corresponding weights
        of each kernel value.
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
def _simple_unsharp_kernel_2d_3x3(
    strength: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a 2D unsharp kernel.

    The kernel has the following form:
    ```python
    [
        0.09911165, 0.15088834, 0.09911165,
        0.15088834, 0.        , 0.15088834,
        0.09911165, 0.15088834, 0.09911165,
    ]
    ```

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple of two numpy arrays. The first array contains the (x, y) offsets
        of the kernel values. The second array contains the corresponding weights
        of each kernel value.
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


def _simple_shift_kernel_2d(
    x_offset: float,
    y_offset: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a 2D shift kernel.

    This function returns a kernel that can be used to shift a raster by a fractional
    number of pixels in the x and y directions. The kernel can also be used to simulate
    channel misalignment in image augmentation.

    Parameters
    ----------
    x_offset : float
        The horizontal (x) offset to apply.

    y_offset : float
        The vertical (y) offset to apply.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple of two numpy arrays. The first array contains the (x, y) offsets
        of the kernel values. The second array contains the corresponding weights
        of each kernel value.
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


def _weight_distance(
    arr: np.ndarray,
    method: Optional[str] = None,
    decay: float = 0.2,
    sigma: float = 1.0,
    center: float = 0.0,
    spherical: bool = False,
    radius: float = 3.0,
) -> float:
    """
    Weights the kernel by distance using various methods.

    Parameters
    ----------
    arr : numpy.ndarray
        The input array.

    method : str, optional
        The weighting method to use. Can be one of the following:
        - "none": No weighting (default).
        - "linear": Linear decay.
        - "sqrt": Square root decay.
        - "power": Power decay.
        - "log": Logarithmic decay.
        - "gaussian": Gaussian decay.

    decay : float, optional
        The decay rate for the `linear`, `sqrt`, and `power` methods.

    sigma : float, optional
        The standard deviation for the Gaussian method.

    center : float, optional
        The center of the array.

    spherical : bool, optional
        If True, adjust weights based on the radius.

    radius : float, optional
        The radius for spherical adjustments.

    Returns
    -------
    float
        The computed weight.

    Notes
    -----
    This function applies a weighting scheme to an input array based on the distance
    of each value from a specified center. The weighting method can be one of several
    options, including no weighting (the default), linear decay, square root decay,
    power decay, logarithmic decay, or Gaussian decay. If the `spherical` parameter is
    set to True, the weighting will be adjusted based on the distance from the center
    using a spherical radius.
    """
    if center == 0.0:
        normed = np.linalg.norm(arr)
    else:
        normed = np.linalg.norm(arr - np.array([0, 0, center]))

    if normed == 0.0:
        weights = 1.0

    if method is None or method == "none" or method == "":
        weights = 1.0
    elif method == "linear":
        weights = np.power((1 - decay), normed)
    elif method == "sqrt":
        weights = np.power(np.sqrt((1 - decay)), normed)
    elif method == "power":
        weights = np.power(np.power((1 - decay), 2), normed)
    elif method == "log":
        weights = np.log(normed + 2) # +2 to avoid log(0)
    elif method == "gaussian":
        weights = np.exp(-(np.power(normed, 2)) / (2 * np.power(sigma, 2)))
    else:
        raise ValueError("Unable to parse parameters for weight_distance.")

    if spherical:
        sqrt_2 = np.sqrt(2)
        half_sqrt_2 = np.divide(sqrt_2, 2)

        if normed > radius + half_sqrt_2:
            return 0.0
        elif normed < radius - half_sqrt_2:
            return 1.0

        dist_min = radius - half_sqrt_2
        dist_max = radius + half_sqrt_2

        normed = 1 - np.interp(normed, [dist_min, dist_max], [0, 1])
        return weights * normed

    return weights


def _rotate_kernel(bottom_right: np.ndarray) -> np.ndarray:
    """
    Creates a whole kernel from a quadrant.

    Parameters
    ----------
    bottom_right : numpy.ndarray
        The bottom-right quadrant of the kernel.

    Returns
    -------
    numpy.ndarray
        The complete kernel generated from the given quadrant.

    Notes
    -----
    This function takes a quadrant of a kernel as input and returns the complete kernel
    by mirroring the quadrant across both axes. The input quadrant should be the
    bottom-right quadrant of the kernel, as this quadrant contains the highest frequency
    components of the kernel.
    """
    size = ((bottom_right.shape[0] - 1) * 2) + 1
    depth = bottom_right.shape[2]
    kernel = np.zeros((size, size, depth), dtype="float32")

    top_right = np.flipud(bottom_right)
    lower_left = np.fliplr(bottom_right)
    top_left = np.flipud(lower_left)

    kernel[size // 2:, size // 2:, :] = bottom_right
    kernel[:1 + -size // 2, :1 + -size // 2, :] = top_left
    kernel[1 + size // 2:, :size // 2, :] = lower_left[1:, :-1, :]
    kernel[:size // 2, 1 + size // 2:, :] = top_right[:-1, 1:, :]

    return kernel


def get_kernel(
    size: int,
    depth: int = 1,
    hole: bool = False,
    inverted: bool = False,
    normalise: bool = True,
    multi_dimensional: bool = False,
    multi_dimensional_center: int = 0,
    spherical: bool = False,
    distance_weight: Optional[str] = None,
    distance_decay: float = 0.2,
    distance_sigma: float = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates a kernel for convolutions.

    Parameters
    ----------
    size : int
        The size of the kernel (must be odd).

    depth : int, optional
        The depth of the kernel. Default: 1.

    hole : bool, optional
        If True, create a hole in the center of the kernel. Default: False.

    inverted : bool, optional
        If True, invert the kernel values. Default: False.

    normalise : bool, optional
        If True, normalize the kernel values. Default: True.

    multi_dimensional : bool, optional
        If True, consider the kernel multi-dimensional. Default: False.

    multi_dimensional_center : int, optional
        The center of the multi-dimensional kernel. Default: 0.

    spherical : bool, optional
        If True, consider the kernel spherical. Default: False.

    distance_weight : str or None, optional
        The distance weighting method. Can be one of the following:
        - "none": No weighting.
        - "linear": Linear decay.
        - "sqrt": Square root decay.
        - "power": Power decay.
        - "log": Logarithmic decay.
        - "gaussian": Gaussian decay.
        Default: None.
    distance_decay : float, optional
        The distance decay factor. Default: 0.2.

    distance_sigma : float, optional
        The distance sigma for Gaussian distance weighting. Default: 1.

    Returns
    -------
    tuple
        A tuple containing the kernel, weights, and offsets.

    Notes
    -----
    This function generates a square kernel for use in convolutions. The kernel can
    have multiple dimensions, and can be spherically or distance-weighted. The function
    returns a tuple containing the kernel, weights, and offsets.
    """
    assert size >= 3, "Kernel must have atleast size 3."
    assert size % 2 != 0, "Kernel must be an uneven size."
    assert isinstance(size, int), "Kernel must be an integer."
    assert depth >= 1, "Depth must be a positive integer"
    assert isinstance(depth, int), "Depth must be an integer."

    if distance_weight is False:
        distance_weight = None

    quadrant = np.zeros((1 + size // 2, 1 + size // 2, depth), dtype="float32")

    for idx_x in range(0, quadrant.shape[0]):
        for idx_y in range(0, quadrant.shape[1]):
            for idx_z in range(0, quadrant.shape[2]):

                z_value = idx_z if multi_dimensional else 0

                weighted = _weight_distance(
                    np.array([idx_x, idx_y, z_value], dtype="float32"),
                    method=distance_weight,
                    decay=distance_decay,
                    sigma=distance_sigma,
                    center=multi_dimensional_center,
                    spherical=spherical,
                    radius=size / 2,
                )

                quadrant[idx_x, idx_y, idx_z] = weighted

    if hole:
        for idx_z in range(0, quadrant.shape[2]):
            quadrant[0, 0, idx_z] = 0

    kernel = _rotate_kernel(quadrant)

    if distance_weight == "log":
        kernel = kernel.max() - kernel

    if inverted:
        kernel = 1 - kernel

    if normalise:
        if multi_dimensional:
            summed = kernel.sum()
            if summed != 0.0:
                kernel = kernel / summed
        else:
            summed = kernel.sum(axis=(0, 1))

            for dim in range(0, depth):
                kernel[:, :, dim] = kernel[:, :, dim] / summed[dim]

    weights = []
    offsets = []

    for idx_x in range(0, kernel.shape[0]):
        for idx_y in range(0, kernel.shape[1]):
            for idx_z in range(0, kernel.shape[2]):
                current_weight = kernel[idx_x][idx_y][idx_z]

                if current_weight <= 0.0:
                    continue

                offsets.append(
                    [
                        idx_x - (kernel.shape[0] // 2),
                        idx_y - (kernel.shape[1] // 2),
                        idx_z
                    ]
                )

                weights.append(current_weight)

    return kernel, np.array(weights, dtype="float32"), np.array(offsets, dtype=int)


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


@jit(nopython=True, nogil=False, fastmath=True, cache=True)
def _hood_to_value(
    method: int,
    values: np.ndarray,
    weights: np.ndarray,
    nodata_value: float = -9999.9,
    center_idx: int = 0,
    value: Union[int, float] = 0.5,
):
    """
    Convert an array of values and weights to a single value using a given method.

    Parameters
    ----------
    method : str
        The method to use for combining the values and weights.

    values : numpy.ndarray
        The values to be combined.

    weights : numpy.ndarray
        The weights to be used in the combination.

    nodata_value : float, optional
        The nodata value to use when computing the result. Default: -9999.9.

    center_idx : int, optional
        The index of the center value in the input arrays. Default: 0.

    value : float, optional
        The value to use when the input arrays have no valid values. Default: 0.5.

    Returns
    -------
    float
        The result of the combination, or the `nodata_value` if there are no valid values.

    Notes
    -----
    This function takes an array of values and an array of weights and combines them
    into a single value using a specified method. The function supports several methods,
    including mean, median, mode, maximum, and minimum. The function can also handle nodata
    values and cases where there are no valid values.
    """
    if method == 1:
        return convolution_funcs._hood_sum(values, weights)
    elif method == 2:
        return convolution_funcs._hood_mode(values, weights)
    elif method == 3:
        return convolution_funcs._hood_max(values, weights)
    elif method == 4:
        return convolution_funcs._hood_min(values, weights)
    elif method == 5:
        return convolution_funcs._hood_contrast(values, weights)
    elif method == 6:
        return convolution_funcs._hood_quantile(values, weights, 0.5)
    elif method == 7:
        return convolution_funcs._hood_standard_deviation(values, weights)
    elif method == 8:
        return convolution_funcs._hood_median_absolute_deviation(values, weights)
    elif method == 9:
        return convolution_funcs._hood_z_score(values, weights, center_idx)
    elif method == 10:
        return convolution_funcs._hood_z_score_mad(values, weights, center_idx)
    elif method == 11:
        return convolution_funcs._hood_sigma_lee(values, weights)
    elif method == 12:
        return convolution_funcs._hood_quantile(values, weights, value)
    elif method == 13:
        return convolution_funcs._hood_count_occurances(values, weights, value, normalise=False)
    elif method == 14:
        return convolution_funcs._hood_count_occurances(values, weights, value, normalise=True)
    elif method == 15:
        return convolution_funcs._hood_roughness(values, weights, center_idx)
    elif method == 16:
        return convolution_funcs._hood_roughness_tri(values, weights, center_idx)
    elif method == 17:
        return convolution_funcs._hood_roughness_tpi(values, weights, center_idx)
    else:
        return nodata_value


@jit(nopython=True, parallel=True, nogil=False, fastmath=True, cache=True)
def _convolve_array_collapse(
    arr: np.ndarray,
    offsets: List[Tuple[int, int, int]],
    weights: List[float],
    method: int = 1,
    nodata: bool = False,
    nodata_value: float = -9999.9,
    normalise_edges: bool = True,
    value: Union[int, float] = 0.5,
) -> np.ndarray:
    """
    Internal function. Convolve an array using a set of offsets and weights and collapse the
    result along the last axis.

    Parameters
    ----------
    arr : numpy.ndarray
        The input array to be convolved and collapsed.

    offsets : list of tuples
        The list of pixel offsets to use in the convolution. Each tuple should be in the
        format `(row_offset, col_offset, depth_offset)`, where row_offset and col_offset
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

    value : int or float, optional
        The value to use for pixels where the kernel extends outside the input array.
        Default: 0.5.

    Returns
    -------
    numpy.ndarray
        The convolved and collapsed array.

    Notes
    -----
    This function convolves an array using a set of offsets and weights and collapses the result
    along the last axis. The function supports different convolution methods, including nearest,
    linear, and cubic. The function can also handle nodata values and cases where the kernel
    extends outside the input array.
    """
    result = np.zeros((arr.shape[0], arr.shape[1], 1), dtype="float32")
    hood_size = len(offsets)

    for idx_y in prange(0, arr.shape[0]):
        for idx_x in range(0, arr.shape[1]):

            center_idx = 0

            if nodata and arr[idx_y, idx_x] == nodata_value:
                result[idx_y, idx_x] = nodata_value
                continue

            hood_normalise = False
            hood_values = np.zeros(hood_size, dtype="float32")
            hood_weights = np.zeros(hood_size, dtype="float32")
            hood_count = 0

            for idx_h in range(0, hood_size):
                hood_x = idx_x + offsets[idx_h][0]
                hood_y = idx_y + offsets[idx_h][1]
                hood_z = offsets[idx_h][2]

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

                if offsets[idx_h][0] == 0 and offsets[idx_h][1] == 0:
                    center_idx = hood_count - 1

            if hood_count == 0:
                result[idx_y, idx_x] = nodata_value
                continue

            hood_values = hood_values[:hood_count]
            hood_weights = hood_weights[:hood_count]

            if hood_normalise:
                hood_weights /= np.sum(hood_weights)

            result[idx_y, idx_x, 0] = _hood_to_value(method, hood_values, hood_weights, nodata_value, center_idx, value)

    return result


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
def convolve_array_simple(
    array: np.ndarray,
    offsets: np.ndarray,
    weights: np.ndarray,
    intensity: float = 1.0,
):
    """
    Convolve a kernel with an array using a simple method.

    Parameters
    ----------
    array : numpy.ndarray
        The array to convolve.

    offsets : numpy.ndarray
        The offsets of the kernel.

    weights : numpy.ndarray
        The weights of the kernel.

    intensity : float, optional
        The intensity of the convolution. If 1.0, the convolution is applied as is. If 0.5,
        the convolution is applied at half intensity. Default: 1.0.

    Returns
    -------
    numpy.ndarray
        The convolved array.

    Notes
    -----
    This function convolves a kernel with an array using a simple method. The function supports
    applying the convolution at a reduced intensity to achieve a blended effect.
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



def convolve_array(
    arr: np.ndarray,
    offsets: np.ndarray,
    weights: np.ndarray,
    method: int = 1,
    nodata: bool = False,
    nodata_value: float = -9999.9,
    normalise_edges: bool = True,
    collapse: bool = False,
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

    collapse : bool, optional
        If True, the convolution results in a (height, width, 1) array. Otherwise, the
        convolution results in a (height, width, depth) array that is applied channel-wise.
        Default: False.

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
    core_utils.type_check(collapse, [bool], "collapse")
    core_utils.type_check(value, [int, float, type(None)], "value")

    assert len(offsets) == len(weights), "offsets and weights must be the same length"
    assert method in range(1, len(_METHOD_ENUMS)), "method must be between 1 and 11"
    assert arr.ndim in [2, 3], "arr must be 2 or 3 dimensional"

    if value is None:
        value = 0.5

    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]

    if collapse:
        return _convolve_array_collapse(
            arr,
            offsets,
            weights,
            method=method,
            nodata=nodata,
            nodata_value=nodata_value,
            normalise_edges=normalise_edges,
            value=value,
        )

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
