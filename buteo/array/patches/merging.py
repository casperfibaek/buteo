""" ### Merging algorithms for patches. ### """

# Standard library
from typing import Optional

# External
import numpy as np
from numba import jit, prange

# Internal
from buteo.array.patches.util import _unique_values


@jit(nopython=True, parallel=True, nogil=True, cache=True)
def _merge_weighted_median(
    arr: np.ndarray,
    arr_weight: np.ndarray,
    ret_arr: np.ndarray,
) -> np.ndarray:
    """Calculate the weighted median of a multi-dimensional array along the first axis.
    This is the order (number_of_overlaps, tile_size, tile_size, number_of_bands)

    Parameters
    ----------
    arr : np.ndarray
        The input array.

    arr_weight : np.ndarray
        The weight array with the same shape as the input array.
    
    ret_arr : np.ndarray
        The output array with the same shape as the input array.

    Returns
    -------
    np.ndarray
        A 3D NumPy array of shape (arr.shape[1], arr.shape[2], arr.shape[3]) with the weighted medians.
    """
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
    ret_arr: np.ndarray,
) -> np.ndarray:
    """Calculate the weighted average of a multi-dimensional array along the last axis.

    Parameters
    ----------
    arr : np.ndarray
        The input array.

    arr_weight : np.ndarray
        The weight array with the same shape as the input array.
    
    ret_arr : np.ndarray
        The output array with the same shape as the input array.

    Returns
    -------
    np.ndarray
        A 3D NumPy array of shape (arr.shape[0], arr.shape[1], 1) with the weighted averages.
    """
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
    ret_arr: np.ndarray,
    method="max",
) -> np.ndarray:
    """Calculate the weighted min or max of a multi-dimensional array along the last axis.

    Parameters
    ----------
    arr : np.ndarray
        The input array.

    arr_weight : np.ndarray
        The weight array with the same shape as the input array.

    ret_arr : np.ndarray
        The output array with the same shape as the input array.

    method : str, optional
        The method to use. Either "min" or "max". Default: "max"

    Returns
    -------
    np.ndarray
        A 3D NumPy array of shape (arr.shape[0], arr.shape[1], 1) with the weighted min or max.
    """
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
    ret_arr: np.ndarray,
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

    ret_arr : np.ndarray
        The output array with the same shape as the input array.

    level : int, optional
        The level of olympic sort to use. Default: 1

    Returns
    -------
    np.ndarray
        A 3D NumPy array of shape (arr.shape[0], arr.shape[1], 1) with the olympic value.
    """

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
    ret_arr: np.ndarray,
    mad_dist: float = 2.0,
) -> np.ndarray:
    """Merge an array of predictions using the MAD-merge methodology.

    Parameters
    ----------
    arr : np.ndarray
        The input array.

    arr_weight : np.ndarray
        The weight array with the same shape as the input array.

    ret_arr : np.ndarray
        The output array with the same shape as the input array.

    mad_dist : float, optional
        The MAD distance. Default: 2.0

    Returns
    -------
    np.ndarray
        A 3D NumPy array of shape (arr.shape[0], arr.shape[1], 1) with the MAD-merged values.
    """
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
                    ret_arr[idx_y, idx_x, idx_band] = median
                    continue

                # Calculate the new weights based on the MAD
                new_weights = np.zeros_like(sorted_weights)
                for idx_z in range(sorted_data.shape[0]):
                    new_weights[idx_z] = 1.0 - (np.minimum(np.abs(sorted_data[idx_z] - median) / (mad * mad_dist), 1))

                if np.sum(new_weights) == 0.0:
                    ret_arr[idx_y, idx_x, idx_band] = median
                    continue

                # Calculate the cumulative sum of the new weights and normalize to the range [0, 1]
                cumsum = np.cumsum(new_weights)
                intersect = (cumsum - 0.5 * new_weights) / cumsum[-1]

                # Interpolate the MAD-merged value and store it in the result array
                ret_arr[idx_y, idx_x, idx_band] = np.interp(0.5, intersect, sorted_data)

    return ret_arr


@jit(nopython=True, parallel=True, nogil=True, cache=True)
def _merge_weighted_mode(
    arr: np.ndarray,
    arr_weight: np.ndarray,
    ret_arr: np.ndarray,
) -> np.ndarray:
    """Calculate the weighted mode of a multi-dimensional array along the last axis.

    Parameters
    ----------
    arr : np.ndarray
        The input array.

    arr_weight : np.ndarray
        The weight array with the same shape as the input array.

    ret_arr : np.ndarray
        The output array with the same shape as the input array.

    Returns
    -------
    np.ndarray
        A 3D NumPy array of shape (arr.shape[0], arr.shape[1], 1) with the weighted modes.
    """
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
                weighted_counts = np.zeros(unique_vals.shape[0], dtype="float32")

                # Calculate the weighted sum for each unique value
                for i in range(unique_vals.shape[0]):
                    idxs = np.where(values == unique_vals[i])
                    weighted_counts[i] = np.sum(weights[idxs])

                # Get the index of the maximum weighted sum
                mode_idx = np.argmax(weighted_counts)

                # Store the weighted mode in the result array
                ret_arr[idx_y, idx_x, idx_band] = unique_vals[mode_idx]

    return ret_arr
