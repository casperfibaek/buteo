"""
### Perform convolutions on arrays. (Funcs) ###
"""

# Standard Library
from typing import Union

# External
import numpy as np
from numba import jit


@jit(nopython=True, nogil=True, fastmath=True, cache=True)
def _hood_max(
    values: np.ndarray,
    weights: np.ndarray,
) -> Union[float, int]:
    """ Get the weighted maximum. """
    idx = np.argmax(np.multiply(values, weights))
    return values[idx]


@jit(nopython=True, nogil=True, fastmath=True, cache=True)
def _hood_min(
    values: np.ndarray,
    weights: np.ndarray,
) -> Union[float, int]:
    """ Get the weighted minimum. """
    max_val = values.max()
    adjusted_values = np.where(weights == 0.0, max_val, values)
    idx = np.argmin(np.divide(adjusted_values, weights + 1e-7))
    return values[idx]


@jit(nopython=True, nogil=True, fastmath=True, cache=True)
def _hood_sum(
    values: np.ndarray,
    weights: np.ndarray,
) -> Union[float, int]:
    """ Get the weighted sum. """
    return np.sum(np.multiply(values, weights))


@jit(nopython=True, nogil=True, fastmath=True, cache=True)
def _hood_mode(
    values: np.ndarray,
    weights: np.ndarray,
) -> Union[float, int]:
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


@jit(nopython=True, nogil=True, fastmath=True, cache=True)
def _hood_count_occurances(
    values: np.ndarray,
    weights: np.ndarray,
    value: Union[int, float, None],
    normalise: bool=False,
) -> Union[float, int]:
    """ Count how many times a number appears in an array. 
        Can be normalised to the size of the array to do feathering.
    """
    if value is None:
        return 0.0

    occurances = 0.0
    for idx in range(values.shape[0]):
        if values[idx] == value and weights[idx] > 0.0:
            occurances += weights[idx]

    if normalise:
        occurances = occurances / values.size

    return occurances


@jit(nopython=True, nogil=True, fastmath=True, cache=True)
def _hood_contrast(
    values: np.ndarray,
    weights: np.ndarray,
) -> Union[float, int]:
    """ Get the local contrast. """
    max_val = values.max()
    adjusted_values = np.where(weights == 0.0, max_val, values)
    local_min = np.min(np.divide(adjusted_values, weights + 1e-7))
    local_max = np.max(np.multiply(values, weights))

    return np.abs(local_max - local_min)

@jit(nopython=True, nogil=True, fastmath=True, cache=True)
def _hood_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    quantile: float,
) -> Union[float, int]:
    """ Get the weighted median. """
    sort_mask = np.argsort(values)
    sorted_data = values[sort_mask]
    sorted_weights = weights[sort_mask]
    cumsum = np.cumsum(sorted_weights)
    intersect = (cumsum - 0.5 * sorted_weights) / cumsum[-1]
    return np.interp(quantile, intersect, sorted_data)


@jit(nopython=True, nogil=True, fastmath=True, cache=True)
def _hood_median_absolute_deviation(
    values: np.ndarray,
    weights: np.ndarray,
) -> Union[float, int]:
    """ Get the median absolute deviation """
    median = _hood_quantile(values, weights, 0.5)
    absdeviation = np.abs(np.subtract(values, median))
    return _hood_quantile(absdeviation, weights, 0.5)


@jit(nopython=True, nogil=True, fastmath=True, cache=True)
def _hood_z_score(
    values: np.ndarray,
    weights: np.ndarray,
    center_idx: int,
) -> Union[float, int]:
    """ Get the local z score ."""
    std = _hood_standard_deviation(values, weights)
    mean = _hood_sum(values, weights)

    center_value = values[center_idx]

    return (center_value - mean) / std


@jit(nopython=True, nogil=True, fastmath=True, cache=True)
def _hood_z_score_mad(
    values: np.ndarray,
    weights: np.ndarray,
    center_idx: int,
) -> Union[float, int]:
    """ Get the local z score calculated around the MAD. """
    mad_std = _hood_median_absolute_deviation(values, weights) * 1.4826
    median = _hood_quantile(values, weights, 0.5)

    center_value = values[center_idx]

    return (center_value - median) / mad_std


@jit(nopython=True, nogil=True, fastmath=True, cache=True)
def _hood_standard_deviation(
    values: np.ndarray,
    weights: np.ndarray,
) -> Union[float, int]:
    "Get the weighted standard deviation. "
    summed = _hood_sum(values, weights)
    variance = np.sum(np.multiply(np.power(np.subtract(values, summed), 2), weights))
    return np.sqrt(variance)


@jit(nopython=True, nogil=True, fastmath=True, cache=True)
def _k_to_size(size: int) -> int:
    """ Preprocess Sigma Lee limits. """
    return int(np.rint(-0.0000837834 * size ** 2 + 0.045469 * size + 0.805733))


@jit(nopython=True, parallel=True, nogil=True, fastmath=True, cache=True)
def _hood_sigma_lee(
    values: np.ndarray,
    weights: np.ndarray,
) -> Union[float, int]:
    """ Sigma lee SAR filter. """
    std = _hood_standard_deviation(values, weights)
    selected_values = np.zeros_like(values)
    selected_weights = np.zeros_like(weights)

    sigma_mult = 1
    passed = 0
    attempts = 0
    ks = _k_to_size(values.size)

    while passed < ks and attempts < 5:
        for idx, val in np.ndenumerate(values):
            if val >= std * sigma_mult or val <= -std * sigma_mult:
                selected_values[idx] = val
                selected_weights[idx] = weights[idx]
                passed += 1

        sigma_mult += 1
        attempts += 1

    if passed < ks:
        return _hood_sum(values, weights)

    sum_of_weights = np.sum(selected_weights)

    if sum_of_weights == 0:
        return 0

    selected_weights = np.divide(selected_weights, sum_of_weights)

    return _hood_sum(selected_values, selected_weights)


@jit(nopython=True, nogil=True, fastmath=True, cache=True)
def _hood_roughness(
    values: np.ndarray,
    weights: np.ndarray,
    center_idx: int,
) -> Union[float, int]:
    """ 
        Defined as the maximum difference between the center value and the
        surrounding values. Weighted.
    """
    center_value = values[center_idx]

    max_idx = np.argmax(np.abs(np.subtract(values, center_value)) * weights)

    return np.abs(center_value - values[max_idx])


@jit(nopython=True, nogil=True, fastmath=True, cache=True)
def _hood_roughness_tpi(
    values: np.ndarray,
    weights: np.ndarray,
    center_idx: int,
) -> Union[float, int]:
    """ 
        Defined as the difference between the center pixel and the mean of
        the surrounding pixels. Weighted.
    """
    center_value = values[center_idx]
    values_non_center = np.delete(values, center_idx)
    weights_non_center = np.delete(weights, center_idx)
    weights_non_center = np.divide(weights_non_center, np.sum(weights_non_center))

    return np.abs(center_value - _hood_sum(values_non_center, weights_non_center))


@jit(nopython=True, nogil=True, fastmath=True, cache=True)
def _hood_roughness_tri(
    values: np.ndarray,
    weights: np.ndarray,
    center_idx: int,
) -> Union[float, int]:
    """ 
        Defined as the mean difference between the center pixel and the
        surrounding pixels. Weighted.
    """
    center_value = values[center_idx]
    values_non_center = np.delete(values, center_idx)
    weights_non_center = np.delete(weights, center_idx)
    weights_non_center = np.divide(weights_non_center, np.sum(weights_non_center))

    return _hood_sum(np.abs(np.subtract(values_non_center, center_value)), weights_non_center)
