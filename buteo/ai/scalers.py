"""
This module provides a set of functions to normalise data for machine learning.
"""

# Standard library
from typing import Union

# External
import numpy as np


def scaler_minmax(arr: np.array) -> np.array:
    """ 
    Normalise an array using the min-max method.

    Args:
        arr (np.array): The array to normalise.

    Returns:
        np.array: The normalised array. (dtype: float32)
    """
    assert isinstance(arr, np.ndarray), "Input must be a numpy array."

    num = np.subtract(arr, arr.min())
    den = np.subtract(arr.max(), arr.min())

    result = np.zeros_like(arr, dtype="float32")
    np.divide(num, den, out=result, where=den != 0)

    return result


def scaler_standardise(arr: np.array) -> np.array:
    """ 
    Normalise an array using the mean and standard deviation method.

    Args:
        arr (np.array): The array to normalise.

    Returns:
        np.array: The normalised array. (dtype: float32)
    """
    assert isinstance(arr, np.ndarray), "Input must be a numpy array."

    mean = np.mean(arr)
    std = np.std(arr)

    result = np.zeros_like(arr, dtype="float32")

    num = np.subtract(arr, mean)
    np.divide(num, std, out=result, where=std != 0)

    return result


def scaler_standardise_mad(arr: np.array) -> np.array:
    """ Normalise an array using the median absolute deviation (MAD) method.
    
    Args:
        arr (np.array): The array to normalise.

    Returns:
        np.array: The normalised array. (dtype: float32)
    """
    assert isinstance(arr, np.ndarray), "Input must be a numpy array."

    median = np.median(arr)
    absdev = np.abs(np.subtract(arr, median))
    madstd = np.median(absdev) * 1.4826

    result = np.zeros_like(arr, dtype="float32")

    sub = np.subtract(arr, median)
    np.divide(sub, madstd, out=result, where=madstd != 0)

    return result


def scaler_iqr(
    arr: np.array,
    q1=0.25,
    q3=0.75,
) -> np.array:
    """ 
    Normalise an array using the interquartile range (IQR) method.

    Args:
        arr (np.array): The array to normalise.

    Keyword Args:
        q1 (float=0.25): The lower quartile.
        q3 (float=0.75): The upper quartile.

    Returns:
        np.array: The normalised array. (dtype: float32)
    """
    assert isinstance(arr, np.ndarray), "Input must be a numpy array."
    assert q1 < 0.5 and q1 >= 0.0, "q1 must be less than 0.5 and above or equal to 0.0"
    assert q3 > 0.5 and q3 <= 1.0, "q3 must be greater than 0.5 and below or equal to 1.0"
    assert q1 < q3, "q1 must be less than q3"

    q1, median, q3 = np.quantile(arr, [q1, 0.5, q3])

    num = np.subtract(arr, median)
    den = np.subtract(q3, q1)

    result = np.zeros_like(arr, dtype="float32")
    np.divide(num, den, out=result, where=den != 0)

    return result


def scaler_to_range(
    arr: np.ndarray,
    min_val: Union[float, int] = 0.0,
    max_val: Union[float, int] = 1.0,
):
    """ 
    Normalise an array to a given range.

    Args:
        arr (np.array): The array to normalise.

    Keyword Args:
        min_val (float=0.0): The minimum value.
        max_val (float=1.0): The maximum value.

    Returns:
        np.array: The normalised array. (dtype: float32)
    """
    assert isinstance(arr, np.ndarray), "Input must be a numpy array."

    num = np.subtract(arr, arr.min())
    den = np.subtract(arr.max(), arr.min())

    result = np.zeros_like(arr, dtype="float32")
    np.divide(num, den, out=result, where=den != 0)

    return np.multiply(result, max_val - min_val) + min_val


def scaler_truncate(
    arr: np.ndarray,
    trunc_min: Union[float, int],
    trunc_max: Union[float, int],
    target_min: Union[float, int] = 0.0,
    target_max: Union[float, int] = 1.0,
) -> np.ndarray:
    """
    Truncate an array and then normalise it to a given range.

    Args:
        arr (np.array): The array to normalise
        trunc_min (float/int): The minimum value to truncate to.
        trunc_max (float/int): The maximum value to truncate to.

    Keyword Args:
        target_min (float=0.0): The minimum value to normalise to.
        target_max (float=1.0): The maximum value to normalise to.
    
    Returns:
        np.array: The normalised array. (dtype: float32)
    """
    assert isinstance(arr, np.ndarray), "Input must be a numpy array."

    truncated = np.where(arr > trunc_max, trunc_max, arr)
    truncated = np.where(truncated < trunc_min, trunc_min, truncated)

    num = np.subtract(truncated, truncated.min())
    den = np.subtract(truncated.max(), truncated.min())

    result = np.zeros_like(arr, dtype="float32")
    np.divide(num, den, out=result, where=den != 0)

    return np.multiply(result, target_max - target_min) + target_min
