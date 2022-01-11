import sys

sys.path.append("../../")


import numpy as np
from numba import jit, prange

from buteo.filters.kernel_generator import create_kernel


@jit(nopython=True, parallel=True, nogil=True, fastmath=True, inline="always")
def hood_summed(values, weights):
    return np.sum(np.multiply(values, weights))


@jit(nopython=True, parallel=True, nogil=True, fastmath=True, inline="always")
def hood_quantile(values, weights, quant):
    sort_mask = np.argsort(values)
    sorted_data = values[sort_mask]
    sorted_weights = weights[sort_mask]
    cumsum = np.cumsum(sorted_weights)
    intersect = (cumsum - 0.5 * sorted_weights) / cumsum[-1]
    return np.interp(quant, intersect, sorted_data)


@jit(nopython=True, parallel=True, nogil=True, fastmath=True, inline="always")
def hood_standard_deviation(values, weights):
    summed = hood_summed(values, weights)
    variance = np.sum(np.multiply(np.power(np.subtract(values, summed), 2), weights))
    return np.sqrt(variance)


@jit(nopython=True, parallel=True, nogil=True, fastmath=True, inline="always")
def hood_median_absolute_deviation(values, weights):
    median = hood_quantile(values, weights, 0.5)
    absdeviation = np.abs(np.subtract(values, median))
    return hood_quantile(absdeviation, weights, 0.5)


@jit(nopython=True, parallel=True, nogil=True, fastmath=True, inline="always")
def hood_z_score(values, weights):
    center_idx = len(values) // 2
    center = values[center_idx]
    std = hood_standard_deviation(values, weights)
    mean = hood_summed(values, weights)

    return (center - mean) / std


@jit(nopython=True, parallel=True, nogil=True, fastmath=True, inline="always")
def hood_z_score_mad(values, weights):
    center_idx = len(values) // 2
    center = values[center_idx]
    mad_std = hood_median_absolute_deviation(values, weights) * 1.4826
    median = hood_quantile(values, weights, 0.5)

    return (center - median) / mad_std


@jit(nopython=True, parallel=True, nogil=True, fastmath=True, inline="always")
def k_to_size(size):
    return int(np.rint(-0.0000837834 * size ** 2 + 0.045469 * size + 0.805733))


@jit(nopython=True, parallel=True, nogil=True, fastmath=True, inline="always")
def hood_sigma_lee_mad(values, weights):
    std = hood_median_absolute_deviation(values, weights) * 1.4826
    selected_values = np.zeros_like(values)
    selected_weights = np.zeros_like(weights)

    sigma2 = std * 2
    sigma_min = -sigma2
    sigma_max = sigma2

    passed = 0
    for idx in range(len(values)):
        if values[idx] >= sigma_min and values[idx] <= sigma_max:
            selected_values[idx] = values[idx]
            selected_weights[idx] = weights[idx]
            passed += 1

    if passed <= k_to_size(values.size):
        steep = np.power(weights, 2)
        steep_sum = np.sum(steep)
        normed_steed = np.divide(steep, steep_sum)
        return hood_quantile(values, normed_steed, 0.5)

    sum_of_weights = np.sum(selected_weights)
    if sum_of_weights == 0:
        return 0

    selected_weights = np.divide(selected_weights, sum_of_weights)

    return hood_quantile(selected_values, selected_weights, 0.5)


@jit(nopython=True, parallel=True, nogil=True, fastmath=True, inline="always")
def hood_sigma_lee(values, weights):
    std = hood_standard_deviation(values, weights)
    selected_values = np.zeros_like(values)
    selected_weights = np.zeros_like(weights)

    sigma_mult = 1
    passed = 0
    attempts = 0
    ks = k_to_size(values.size)

    while passed < ks and attempts < 5:
        for idx in range(len(values)):
            if values[idx] >= std * sigma_mult and values[idx] <= -std * sigma_mult:
                selected_values[idx] = values[idx]
                selected_weights[idx] = weights[idx]
                passed += 1

        sigma_mult += 1
        attempts += 1

    if passed < ks:
        return hood_summed(values, weights)

    sum_of_weights = np.sum(selected_weights)

    if sum_of_weights == 0:
        return 0

    selected_weights = np.divide(selected_weights, sum_of_weights)

    return hood_summed(selected_values, selected_weights)


@jit(nopython=True, parallel=True, nogil=True, fastmath=True)
def convolve_3d(
    arr,
    offsets,
    weights,
    operation="sum",
    border="valid",
    quantile=0.5,
    nodata=False,
    nodata_value=0,
):
    x_adj = arr.shape[0] - 1
    y_adj = arr.shape[1] - 1
    z_adj = (arr.shape[2] - 1) // 2

    hood_size = len(offsets)
    result = np.zeros(arr.shape[:2], dtype="float32")
    border = True if border == "valid" else False

    for x in prange(arr.shape[0]):
        for y in range(arr.shape[1]):

            hood_values = np.zeros(hood_size, dtype="float32")
            hood_weights = np.zeros(hood_size, dtype="float32")
            weight_sum = np.array([0.0], dtype="float32")
            normalise = False

            for n in range(hood_size):
                offset_x = x + offsets[n][0]
                offset_y = y + offsets[n][1]
                offset_z = offsets[n][2]

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

                value = arr[offset_x, offset_y, offset_z]

                if border == True and outside == True:
                    normalise = True
                    hood_weights[n] = 0
                elif nodata and value == nodata_value:
                    normalise = True
                    hood_weights[n] = 0
                else:
                    hood_values[n] = value
                    weight = weights[n]

                    hood_weights[n] = weight
                    weight_sum[0] += weight

            if normalise:
                hood_weights = np.divide(hood_weights, weight_sum[0])

            if operation == "sum":
                result[x][y] = np.sum(np.multiply(hood_values, hood_weights))

            if operation == "mean":
                result[x][y] = (
                    np.sum(np.multiply(hood_values, hood_weights))
                    / np.sum(hood_weights)
                    + 0.0000001
                )

            elif operation == "quantile":
                result[x][y] = hood_quantile(hood_values, hood_weights, quantile)

            elif operation == "median":
                result[x][y] = hood_quantile(hood_values, hood_weights, 0.5)

            elif operation == "median_absolute_deviation":
                result[x][y] == hood_median_absolute_deviation(
                    hood_values, hood_weights
                )
            elif operation == "standard_deviation":
                result[x][y] = hood_standard_deviation(hood_values, hood_weights)

            elif operation == "z_score":
                result[x][y] = hood_z_score(hood_values, hood_weights)

            elif operation == "z_score_mad":
                result[x][y] = hood_z_score_mad(hood_values, hood_weights)

            elif operation == "sigma_lee":
                result[x][y] = hood_sigma_lee(hood_values, hood_weights)

            elif operation == "sigma_lee_mad":
                result[x][y] = hood_sigma_lee_mad(hood_values, hood_weights)

    return result


def filter_array(
    arr,
    shape,
    sigma=1,
    spherical=True,
    edge_weights=True,
    normalised=True,
    nodata=False,
    nodata_value=0,
    quantile=0.5,
    distance_calc="gaussian",
    radius_method="ellipsoid",
    remove_zero_weights=True,
    operation="sum",
    kernel=None,
):
    if len(arr.shape) == 3 and len(shape) != 3:
        shape = (shape[0], shape[1], 1)

    if len(arr.shape) == 2:
        if len(shape) == 3:
            if shape[0] != 1:
                shape[0] = 1
        elif len(shape) == 2:
            arr = arr[np.newaxis, :, :]
        else:
            raise ValueError("Unable to merge shape and array.")

    if kernel is None:
        _kernel, offsets, weights = create_kernel(
            shape,
            sigma=sigma,
            spherical=spherical,
            edge_weights=edge_weights,
            offsets=True,
            normalised=normalised,
            distance_calc=distance_calc,
            radius_method=radius_method,
            remove_zero_weights=remove_zero_weights,
        )
    else:
        _kernel, offsets, weights = kernel

    return convolve_3d(
        arr,
        offsets,
        weights,
        operation=operation,
        nodata=nodata,
        nodata_value=nodata_value,
        quantile=quantile,
    )


# band_last
@jit(nopython=True, parallel=True, nogil=True, fastmath=True)
def interp_array(
    arr,
    min_vals,
    max_vals,
    min_vals_adj,
    max_vals_adj,
):
    out_arr = np.empty_like(arr)
    for img in prange(arr.shape[0]):
        for band in range(arr.shape[3]):
            min_val = min_vals[img, 0, 0, band]
            min_val_adj = min_vals_adj[img, 0, 0, band]

            max_val = max_vals[img, 0, 0, band]
            max_val_adj = max_vals_adj[img, 0, 0, band]

            out_arr[img, :, :, band] = np.interp(
                arr[img, :, :, band], (min_val, max_val), (min_val_adj, max_val_adj)
            )

    return out_arr
