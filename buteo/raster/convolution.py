"""
### Perform convolutions on arrays.  ###
"""

# External
import numpy as np
from numpy.lib.stride_tricks import as_strided
from numba import jit, prange


def weight_distance(
    arr,
    method=None,
    decay=0.2,
    sigma=1.0,
    center=0.0,
    spherical=False,
    radius=3.0,
):
    """
    Weights the kernel by distance using various methods.

    Args:
        arr (numpy.ndarray): The input array.
        method (str, default=None): The weighting method to use.
            "none": No weighting (default).
            "linear": Linear decay.
            "sqrt": Square root decay.
            "power": Power decay.
            "log": Logarithmic decay.
            "gaussian": Gaussian decay.
        decay (float, default=0.2): The decay rate for the `linear`, `sqrt`, and `power` methods.
        sigma (float, default=1.0): The standard deviation for the Gaussian method.
        center (float, default=0.0): The center of the array.
        spherical (bool, default=False): If True, adjust weights based on the radius.
        radius (float, default=3.0): The radius for spherical adjustments.

    Returns:
        float: The computed weight.
    """

    if center == 0.0:
        normed = np.linalg.norm(arr)
    else:
        normed = np.linalg.norm(arr - np.array([0, 0, center]))

    if normed == 0.0:
        weight = 1.0

    if method is None or method == "none" or method == "":
        weight = 1.0
    elif method == "linear":
        weight = np.power((1 - decay), normed)
    elif method == "sqrt":
        weight = np.power(np.sqrt((1 - decay)), normed)
    elif method == "power":
        weight = np.power(np.power((1 - decay), 2), normed)
    elif method == "log":
        weight = np.log(normed + 2)
    elif method == "gaussian":
        weight = np.exp(-(np.power(normed, 2)) / (2 * np.power(sigma, 2)))
    else:
        raise ValueError("Unable to parse parameters distance_calc.")

    if spherical:
        sqrt_2 = np.sqrt(2)
        half_sqrt_2 = np.divide(sqrt_2, 2)

        if normed > (radius - half_sqrt_2) and normed < (radius + half_sqrt_2):
            adjustment = sqrt_2 / normed
        elif normed > (radius - half_sqrt_2):
            adjustment = 0.0
        elif normed < (radius + half_sqrt_2):
            adjustment = 1.0
        else:
            adjustment = 1.0

        return weight * adjustment

    return weight


def rotate_kernel(bottom_right):
    """
    Creates a whole kernel from a quadrant.

    Args:
        bottom_right (numpy.ndarray): The bottom-right quadrant of the kernel.

    Returns:
        numpy.ndarray: The complete kernel generated from the given quadrant.
    """

    size = ((bottom_right.shape[0] - 1) * 2) + 1
    depth = bottom_right.shape[2]
    kernel = np.zeros((size, size, depth), dtype="float32")

    top_right = np.flipud(bottom_right)
    lower_left = np.fliplr(bottom_right)
    top_left = np.flipud(lower_left)

    kernel = np.block([[top_left, top_right], [lower_left, bottom_right]])

    return kernel


def get_kernel(
    size,
    depth=1,
    hole=False,
    inverted=False,
    normalise=True,
    multi_dimensional=False,
    multi_dimensional_center=0,
    spherical=False,
    distance_weight=None,
    distance_decay=0.2,
    distance_sigma=1,
):
    """
    Generates a square kernel for convolutions.

    Args:
        size (int): Size of the kernel (must be odd).
        depth (int, default=1): Depth of the kernel.
        hole (bool, default=False): Create a hole in the center of the kernel.
        inverted (bool, default=False): Invert the kernel values.
        normalise (bool, default=True): Normalize the kernel values.
        multi_dimensional (bool, default=False): Consider the kernel multi-dimensional.
        multi_dimensional_center (int, default=0): Center of the
            multi-dimensional kernel.
        spherical (bool, default=False): Consider the kernel spherical.
        distance_weight (str or None, default=None): Distance weighting method.
        distance_decay (float, default=0.2): Distance decay factor.
        distance_sigma (float, default=1): Distance sigma for Gaussian distance weighting.

    Returns:
        tuple: A tuple containing the kernel, weights, and offsets.
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

                weighted = weight_distance(
                    np.array([idx_x, idx_y, z_value], dtype="float32"),
                    method=distance_weight,
                    decay=distance_decay,
                    sigma=distance_sigma,
                    center=multi_dimensional_center,
                    spherical=spherical,
                    radius=size // 2,
                )

                quadrant[idx_x, idx_y, idx_z] = weighted
    if hole:
        for idx_z in range(0, quadrant.shape[2]):
            quadrant[0, 0, idx_z] = 0

    kernel = rotate_kernel(quadrant)

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


@jit(nopython=True, parallel=True, nogil=True, fastmath=True, inline="always")
def hood_max(values, weights):
    """ Get the weighted maximum. """
    idx = np.argmax(np.multiply(values, weights))
    return values[idx]


@jit(nopython=True, parallel=True, nogil=True, fastmath=True, inline="always")
def hood_min(values, weights):
    """ Get the weighted minimum. """
    max_val = values.max()
    adjusted_values = np.where(weights == 0.0, max_val, values)
    idx = np.argmin(np.divide(adjusted_values, weights + 1e-7))
    return values[idx]


@jit(nopython=True, parallel=True, nogil=True, fastmath=True, inline="always")
def hood_sum(values, weights):
    """ Get the weighted sum. """
    return np.sum(np.multiply(values, weights))


@jit(nopython=True, parallel=True, nogil=True, fastmath=True, inline="always")
def hood_mode(values, weights):
    """ Get the weighted sum. """
    values_ints = np.rint(values)
    unique = np.unique(values_ints)

    most_occured_value = 0
    most_occured_weight = -9999.9

    for unique_value in unique:
        cum_weight = 0
        for idx in range(values.shape[0]):
            if values_ints[idx] == unique_value:
                cum_weight += weights[idx]

        if cum_weight > most_occured_weight:
            most_occured_weight = cum_weight
            most_occured_value = unique_value

    return most_occured_value


@jit(nopython=True, parallel=True, nogil=True, fastmath=True, inline="always")
def hood_contrast(values, weights):
    """ Get the local contrast. """
    max_val = values.max()
    adjusted_values = np.where(weights == 0.0, max_val, values)
    local_min = np.min(np.divide(adjusted_values, weights + 1e-7))
    local_max = np.max(np.multiply(values, weights))

    return np.abs(local_max - local_min)

@jit(nopython=True, parallel=True, nogil=True, fastmath=True, inline="always")
def hood_quantile(values, weights, quant):
    """ Get the weighted median. """
    sort_mask = np.argsort(values)
    sorted_data = values[sort_mask]
    sorted_weights = weights[sort_mask]
    cumsum = np.cumsum(sorted_weights)
    intersect = (cumsum - 0.5 * sorted_weights) / cumsum[-1]
    return np.interp(quant, intersect, sorted_data)


@jit(nopython=True, parallel=True, nogil=True, fastmath=True, inline="always")
def hood_median_absolute_deviation(values, weights):
    """ Get the median absolute deviation """
    median = hood_quantile(values, weights, 0.5)
    absdeviation = np.abs(np.subtract(values, median))
    return hood_quantile(absdeviation, weights, 0.5)


@jit(nopython=True, parallel=True, nogil=True, fastmath=True, inline="always")
def hood_z_score(values, center_value, weights):
    """ Get the local z score ."""
    std = hood_standard_deviation(values, weights)
    mean = hood_sum(values, weights)

    return (center_value - mean) / std


@jit(nopython=True, parallel=True, nogil=True, fastmath=True, inline="always")
def hood_z_score_mad(values, center_value, weights):
    """ Get the local z score calculated around the MAD. """
    mad_std = hood_median_absolute_deviation(values, weights) * 1.4826
    median = hood_quantile(values, weights, 0.5)

    return (center_value - median) / mad_std


@jit(nopython=True, parallel=True, nogil=True, fastmath=True, inline="always")
def hood_standard_deviation(values, weights):
    "Get the weighted standard deviation. "
    summed = hood_sum(values, weights)
    variance = np.sum(np.multiply(np.power(np.subtract(values, summed), 2), weights))
    return np.sqrt(variance)


@jit(nopython=True, parallel=True, nogil=True, fastmath=True, inline="always")
def k_to_size(size):
    """ Preprocess Sigma Lee limits. """
    return int(np.rint(-0.0000837834 * size ** 2 + 0.045469 * size + 0.805733))


@jit(nopython=True, parallel=True, nogil=True, fastmath=True, inline="always")
def hood_sigma_lee(values, weights):
    """ Sigma lee SAR filter. """
    std = hood_standard_deviation(values, weights)
    selected_values = np.zeros_like(values)
    selected_weights = np.zeros_like(weights)

    sigma_mult = 1
    passed = 0
    attempts = 0
    ks = k_to_size(values.size)

    while passed < ks and attempts < 5:
        for idx, val in np.ndenumerate(values):
            if val >= std * sigma_mult or val <= -std * sigma_mult:
                selected_values[idx] = val
                selected_weights[idx] = weights[idx]
                passed += 1

        sigma_mult += 1
        attempts += 1

    if passed < ks:
        return hood_sum(values, weights)

    sum_of_weights = np.sum(selected_weights)

    if sum_of_weights == 0:
        return 0

    selected_weights = np.divide(selected_weights, sum_of_weights)

    return hood_sum(selected_values, selected_weights)


@jit(nopython=True, nogil=True)
def pad_array_view(arr, pad_size=1):
    """
    Create a padded view of an array using SAME padding.

    Args:
        arr (numpy.ndarray): The input array to be padded.
2
    Keyword Args:
        pad_size (int, default=1): The number of padding elements to add
            to each side of the array. Default is 1.

    Returns:
        numpy.ndarray: A padded view of the input array.
    """
    # Get original array shape and strides
    shape = arr.shape
    strides = arr.strides

    # Compute new shape and strides for the padded view
    new_shape = (shape[0] + 2 * pad_size, shape[1] + 2 * pad_size, shape[2])
    new_strides = (strides[0], strides[1], strides[2])

    # Create the padded view using as_strided
    padded_view = as_strided(arr, shape=new_shape, strides=new_strides)

    # Pad the edges using the SAME padding
    padded_view[:pad_size, :, :] = padded_view[pad_size:pad_size + 1, :, :]
    padded_view[-pad_size:, :, :] = padded_view[-pad_size - 1:-pad_size, :, :]
    padded_view[:, :pad_size, :] = padded_view[:, pad_size:pad_size + 1, :]
    padded_view[:, -pad_size:, :] = padded_view[:, -pad_size - 1:-pad_size, :]

    return padded_view


@jit(nopython=True, nogil=True)
def pad_array(arr, pad_size=1):
    """ Pad an array using SAME """
    # Core
    arr_padded = np.zeros((arr.shape[0] + int(pad_size * 2), arr.shape[1] + int(pad_size * 2), arr.shape[2]), dtype=arr.dtype)
    arr_padded[pad_size:-pad_size, pad_size:-pad_size, :] = arr

    # Corners
    arr_padded[0:pad_size, 0:pad_size, :] = arr[ 0,  0, :]
    arr_padded[-pad_size:, -pad_size:, :] = arr[-1, -1, :]
    arr_padded[0:pad_size, -pad_size:, :] = arr[ 0, -1, :]
    arr_padded[-pad_size:, 0:pad_size, :] = arr[-1,  0, :]

    # Sides
    for idx in range(0, pad_size):
        arr_padded[idx, pad_size:-pad_size, :] = arr[ 0,  :, :]
        arr_padded[-(idx + 1):, pad_size:-pad_size, :] = arr[-1,  :, :]
        arr_padded[pad_size:-pad_size, -(idx + 1), :] = arr[ :, -1, :]
        arr_padded[pad_size:-pad_size, idx, :] = arr[ :,  0, :]

    return arr_padded

METHOD_ENUMS = {
    "sum": 1,
    "mode": 2,
    "max": 3,
    "min": 4,
    "contrast": 5,
    "median": 6,
    "std": 7,
    "mad": 8,
    "z_score": 9,
    "z_score_mad": 10,
    "sigma_lee": 11,
}

@jit(nopython=True, parallel=True, nogil=False, fastmath=True, cache=True)
def convolve_array(
    arr,
    offsets,
    weights,
    method=1,
    nodata=False,
    nodata_value=-9999.9,
):
    """
    Convolve an image with a function.

    Args:
        arr (numpy.ndarray): The input array to convolve.
        offsets (list of tuples): The list of offsets for the neighborhood
            used in the convolution.
        weights (list): The list of weights used in the convolution.

    Keyword Args:
        method (int, default=1): The method to use for the convolution.
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
        nodata (bool, default=False): If True, nodata values are considered
            in the convolution.
        nodata_value (float, default=-9999.9): The value representing nodata.

    Returns:
        numpy.ndarray: The convolved array.
    """
    x_adj = arr.shape[0] - 1
    y_adj = arr.shape[1] - 1
    z_adj = (arr.shape[2] - 1) // 2

    hood_size = len(offsets)

    result = np.zeros((arr.shape[0], arr.shape[1], 1), dtype="float32")

    for idx_x in prange(arr.shape[0]):
        for idx_y in range(arr.shape[1]):

            hood_values = np.zeros(hood_size, dtype="float32")

            hood_weights = np.zeros(hood_size, dtype="float32")
            weight_sum = np.array([0.0], dtype="float32")
            center_value = np.float32(arr[idx_x, idx_y, 0])
            normalise = False

            for idx_n in range(hood_size):
                offset_x = idx_x + offsets[idx_n][0]
                offset_y = idx_y + offsets[idx_n][1]
                offset_z = offsets[idx_n][2]

                outside = False

                if offset_z < -z_adj:
                    offset_z = -z_adj
                    outside = True
                elif offset_z > z_adj:
                    offset_z = z_adj
                    outside = True

                if offset_x < 0:
                    offset_x = 0
                    outside = True
                elif offset_x > x_adj:
                    offset_x = x_adj
                    outside = True

                if offset_y < 0:
                    offset_y = 0
                    outside = True
                elif offset_y > y_adj:
                    offset_y = y_adj
                    outside = True

                value = np.float32(arr[offset_x, offset_y, offset_z])

                if outside is True:
                    normalise = True
                    hood_weights[idx_n] = 0
                elif nodata and value == nodata_value:
                    normalise = True
                    hood_weights[idx_n] = 0
                else:
                    hood_values[idx_n] = value

                    weight = weights[idx_n]

                    hood_weights[idx_n] = weight
                    weight_sum[0] += weight

            if normalise and weight_sum[0] > 0.0:
                hood_weights = np.true_divide(hood_weights, weight_sum[0])

            if method == 1:
                result[idx_x, idx_y, 0] = hood_sum(hood_values, hood_weights)
            elif method == 2:
                result[idx_x, idx_y, 0] = hood_mode(hood_values, hood_weights)
            elif method == 3:
                result[idx_x, idx_y, 0] = hood_max(hood_values, hood_weights)
            elif method == 4:
                result[idx_x, idx_y, 0] = hood_min(hood_values, hood_weights)
            elif method == 5:
                result[idx_x, idx_y, 0] = hood_contrast(hood_values, hood_weights)
            elif method == 6:
                result[idx_x, idx_y, 0] = hood_quantile(hood_values, hood_weights, 0.5)
            elif method == 7:
                result[idx_x, idx_y, 0] = hood_standard_deviation(hood_values, hood_weights)
            elif method == 8:
                result[idx_x, idx_y, 0] = hood_median_absolute_deviation(hood_values, hood_weights)
            elif method == 9:
                result[idx_x, idx_y, 0] = hood_z_score(hood_values, center_value, hood_weights)
            elif method == 10:
                result[idx_x, idx_y, 0] = hood_z_score_mad(hood_values, center_value, hood_weights)
            elif method == 11:
                result[idx_x, idx_y, 0] = hood_sigma_lee(hood_values, hood_weights)
            else:
                result[idx_x, idx_y, 0] = nodata_value

    return result
