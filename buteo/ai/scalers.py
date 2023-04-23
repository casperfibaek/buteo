"""
This module provides a set of functions to normalise data for machine learning.
"""

# Standard library
from typing import Union, Optional, Tuple

# External
import numpy as np


def scaler_minmax(
    arr: np.array,
    stat_dict: Optional[dict] = None,
) -> Tuple[np.array, dict]:
    """ 
    Normalise an array using the min-max method.

    Args:
        arr (np.array): The array to normalise.

    Keyword Args:
        stat_dict (dict=None): A dictionary containing the min and max values of the array.

    Returns:
        (np.array, dict): The normalised array (always float32) and a dictionary containing
            the min and max values.
    """
    assert isinstance(arr, np.ndarray), "Input must be a numpy array."
    assert isinstance(stat_dict, dict) or stat_dict is None, "stat_dict must be a dictionary or None"

    if stat_dict is not None:
        arr_min = stat_dict["min"]
        arr_max = stat_dict["max"]
    else:
        arr_min = arr.min()
        arr_max = arr.max()

    stat_dict = {
        "min": arr.min(),
        "max": arr.max(),
    }

    num = np.subtract(arr, arr_min)
    den = np.subtract(arr_max, arr_min)

    result = np.zeros_like(arr, dtype="float32")
    np.divide(num, den, out=result, where=den != 0)

    return result, stat_dict


def scaler_standardise(
    arr: np.array,
    stat_dict: Optional[dict] = None,
) -> Tuple[np.array, dict]:
    """ 
    Normalise an array using the mean and standard deviation method.

    Args:
        arr (np.array): The array to normalise.
    
    Keyword Args:
        stat_dict (dict=None): A dictionary containing the mean and
            standard deviation of the array.

    Returns:
        (np.array, dict): The normalised array (always float32) and a dictionary containing statistics.
    """
    assert isinstance(arr, np.ndarray), "Input must be a numpy array."
    assert isinstance(stat_dict, dict) or stat_dict is None, "stat_dict must be a dictionary or None"

    if stat_dict is not None:
        arr_mean = stat_dict["mean"]
        arr_std = stat_dict["std"]
    else:
        arr_mean = np.nanmean(arr)
        arr_std = np.nanstd(arr)

    stat_dict = {
        "mean": arr_mean,
        "std": arr_std,
    }

    result = np.zeros_like(arr, dtype="float32")

    num = np.subtract(arr, arr_mean)
    np.divide(num, arr_std, out=result, where=arr_std != 0)

    return (result, stat_dict)


def scaler_standardise_mad(
    arr: np.array,
    stat_dict: Optional[dict] = None,
) -> Tuple[np.array, dict]:
    """ Normalise an array using the median absolute deviation (MAD) method.
    
    Args:
        arr (np.array): The array to normalise.

    Keyword Args:
        stat_dict (dict=None): A dictionary containing the median, absolute
            and deviations, and the median absolute deviation of the array.

    Returns:
        (np.array, dict): The normalised array (always float32) and a
            dictionary containing statistics.
    """
    assert isinstance(arr, np.ndarray), "Input must be a numpy array."
    assert isinstance(stat_dict, dict) or stat_dict is None, "stat_dict must be a dictionary or None"

    if stat_dict is not None:
        median = stat_dict["median"]
        absdev = stat_dict["absdev"]
        madstd = stat_dict["madstd"]
    else:
        median = np.nanmedian(arr)
        absdev = np.abs(np.subtract(arr, median))
        madstd = np.nanmedian(absdev) * 1.4826

    stat_dict = {
        "median": median,
        "absdev": absdev,
        "madstd": madstd,
    }

    result = np.zeros_like(arr, dtype="float32")

    sub = np.subtract(arr, median)
    np.divide(sub, madstd, out=result, where=madstd != 0)

    return result, stat_dict


def scaler_iqr(
    arr: np.array,
    q1=0.25,
    q3=0.75,
    stat_dict: Optional[dict] = None,
) -> Tuple[np.array, dict]:
    """ 
    Normalise an array using the interquartile range (IQR) method.

    Args:
        arr (np.array): The array to normalise.

    Keyword Args:
        q1 (float=0.25): The lower quartile.
        q3 (float=0.75): The upper quartile.
        stat_dict (dict=None): A dictionary containing the lower, median, and upper quartiles.

    Returns:
        (np.array, dict): The normalised array (always float32) and a dictionary containing statistics.
    """
    assert isinstance(arr, np.ndarray), "Input must be a numpy array."
    assert 0.0 <= q1 < 0.5, "q1 must be less than 0.5 and above or equal to 0.0"
    assert 0.5 < q3 <= 1.0, "q3 must be greater than 0.5 and below or equal to 1.0"
    assert q1 < q3, "q1 must be less than q3"
    assert isinstance(stat_dict, dict) or stat_dict is None, "stat_dict must be a dictionary or None"

    if stat_dict is not None:
        q1 = stat_dict["q1"]
        median = stat_dict["median"]
        q3 = stat_dict["q3"]
    else:
        q1, median, q3 = np.nanquantile(arr, [q1, 0.5, q3])

    stat_dict = {
        "q1": q1,
        "median": median,
        "q3": q3,
    }

    num = np.subtract(arr, median)
    den = np.subtract(q3, q1)

    result = np.zeros_like(arr, dtype="float32")
    np.divide(num, den, out=result, where=den != 0)

    return result, stat_dict


def scaler_to_range(
    arr: np.ndarray,
    min_val: Union[float, int] = 0.0,
    max_val: Union[float, int] = 1.0,
    stat_dict: Optional[dict] = None,
) -> Tuple[np.array, dict]:
    """ 
    Normalise an array to a given range.

    Args:
        arr (np.array): The array to normalise.

    Keyword Args:
        min_val (float=0.0): The minimum value.
        max_val (float=1.0): The maximum value.
        stat_dict (dict=None): A dictionary containing the minimum and maximum values of the array.

    Returns:
        (np.array, dict): The normalised array (always float32) and a dictionary containing statistics.
    """
    assert isinstance(arr, np.ndarray), "Input must be a numpy array."
    assert isinstance(stat_dict, dict) or stat_dict is None, "stat_dict must be a dictionary or None"

    if stat_dict is not None:
        arr_min = stat_dict["min"]
        arr_max = stat_dict["max"]
    else:
        arr_min = np.min(arr)
        arr_max = np.max(arr)

    stat_dict = {
        "min": arr_min,
        "max": arr_max,
    }

    num = np.subtract(arr, arr_min)
    den = np.subtract(arr_max, arr_min)

    result = np.zeros_like(arr, dtype="float32")
    np.divide(num, den, out=result, where=den != 0)

    return np.multiply(result, max_val - min_val) + min_val, stat_dict


def scaler_truncate(
    arr: np.ndarray,
    trunc_min: Union[float, int],
    trunc_max: Union[float, int],
    target_min: Union[float, int] = 0.0,
    target_max: Union[float, int] = 1.0,
    stat_dict: Optional[dict] = None,
) -> Tuple[np.array, dict]:
    """
    Truncate an array and then normalise it to a given range.

    Args:
        arr (np.array): The array to normalise
        trunc_min (float/int): The minimum value to truncate to.
        trunc_max (float/int): The maximum value to truncate to.

    Keyword Args:
        target_min (float=0.0): The minimum value to normalise to.
        target_max (float=1.0): The maximum value to normalise to.
        stat_dict (dict=None): A dictionary containing the minimum and maximum values of the array.
    
    Returns:
        (np.array, dict): The normalised array (always float32) and a dictionary containing statistics.
    """
    assert isinstance(arr, np.ndarray), "Input must be a numpy array."
    assert isinstance(stat_dict, dict) or stat_dict is None, "stat_dict must be a dictionary or None"
    assert trunc_min < trunc_max, "trunc_min must be less than trunc_max"
    assert target_min < target_max, "target_min must be less than target_max"

    if stat_dict is not None:
        trunc_min = stat_dict["trunc_min"]
        trunc_max = stat_dict["trunc_max"]
        target_min = stat_dict["target_min"]
        target_max = stat_dict["target_max"]

    stat_dict = {
        "trunc_min": trunc_min,
        "trunc_max": trunc_max,
        "target_min": target_min,
        "target_max": target_max,
    }

    truncated = np.where(arr > trunc_max, trunc_max, arr)
    truncated = np.where(truncated < trunc_min, trunc_min, truncated)

    if "arr_min" not in stat_dict:
        stat_dict["arr_min"] = np.min(truncated)
    if "arr_max" not in stat_dict:
        stat_dict["arr_max"] = np.max(truncated)

    arr_min = stat_dict["arr_min"]
    arr_max = stat_dict["arr_max"]

    num = np.subtract(truncated, arr_min)
    den = np.subtract(arr_max, arr_min)

    result = np.zeros_like(arr, dtype="float32")
    np.divide(num, den, out=result, where=den != 0)

    return np.multiply(result, target_max - target_min) + target_min, stat_dict
