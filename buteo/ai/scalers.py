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
    Normalize an input numpy array using the min-max scaling method.

    Parameters
    ----------
    arr : np.ndarray
        The input numpy array to be normalized.

    stat_dict : Optional[Dict[str, float]], default: None
        A dictionary containing the minimum and maximum values of the input array.
        If not provided, the function will compute the minimum and maximum values from the input array.

    Returns
    -------
    Tuple[np.ndarray, Dict[str, float]]
        A tuple containing the normalized numpy array (with dtype float32) and a dictionary containing the minimum
        and maximum values used for scaling. The minimum and maximum values are stored in the dictionary using the
        keys "min" and "max", respectively.

    Raises
    ------
    AssertionError
        If the input array is not a numpy array, or if the stat_dict is not a dictionary or None.
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
    Standardize an an input numpy array.

    Parameters
    ----------
    arr : np.ndarray
        The input numpy array to be normalized.

    stat_dict : Optional[Dict[str, float]], default: None
        A dictionary containing the mean and standard deviation values of the input array.
        If not provided, the function will compute the mean and standard deviation from the input array.

    Returns
    -------
    Tuple[np.ndarray, Dict[str, float]]
        A tuple containing the normalized numpy array (with dtype float32) and a dictionary containing the mean
        and standard deviation values used for scaling. The mean and standard deviation values are stored in the
        dictionary using the keys "mean" and "std", respectively.

    Raises
    ------
    AssertionError
        If the input array is not a numpy array, or if the stat_dict is not a dictionary or None.
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

    return result, stat_dict


def scaler_standardise_mad(
    arr: np.array,
    stat_dict: Optional[dict] = None,
) -> Tuple[np.array, dict]:
    """
    Normalize an input numpy array using the median absolute deviation (MAD) scaling method.

    Parameters
    ----------
    arr : np.ndarray
        The input numpy array to be normalized.

    stat_dict : Optional[Dict[str, float]], default: None
        A dictionary containing the median, absolute deviation and median absolute deviation of the input array.
        If not provided, the function will compute these statistics from the input array.

    Returns
    -------
    Tuple[np.ndarray, Dict[str, float]]
        A tuple containing the normalized numpy array (with dtype float32) and a dictionary containing the median,
        absolute deviation, and median absolute deviation used for scaling. These statistics are stored in the
        dictionary using the keys "median", "absdev", and "madstd", respectively.

    Raises
    ------
    AssertionError
        If the input array is not a numpy array, or if the stat_dict is not a dictionary or None.
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
    Normalize an input numpy array using the interquartile range (IQR) method.

    Parameters
    ----------
    arr : np.ndarray
        The input numpy array to be normalized.

    q1 : float, default: 0.25
        The lower quartile to use in the IQR calculation.

    q3 : float, default: 0.75
        The upper quartile to use in the IQR calculation.

    stat_dict : Optional[Dict[str, float]], default: None
        A dictionary containing the lower, median, and upper quartiles of the input array.
        If not provided, the function will compute the quartiles from the input array.

    Returns
    -------
    Tuple[np.ndarray, Dict[str, float]]
        A tuple containing the normalized numpy array (with dtype float32) and a dictionary containing the lower,
        median, and upper quartiles used for scaling. The quartile values are stored in the dictionary using the keys
        "q1", "median", and "q3", respectively.

    Raises
    ------
    AssertionError
        If the input array is not a numpy array, q1 is not between 0.0 and 0.5, q3 is not between 0.5 and 1.0, or if
        q1 is greater than or equal to q3, or if stat_dict is not a dictionary or None.
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
    Normalizes an input numpy array to a given range.

    Parameters
    ----------
    arr : np.ndarray
        The input numpy array to be normalized.

    min_val : Union[float, int], optional
        The minimum value to scale the array to, default: 0.0.

    max_val : Union[float, int], optional
        The maximum value to scale the array to, default: 1.0.

    stat_dict : Optional[Dict[str, float]], optional
        A dictionary containing the minimum and maximum values of the input array.
        If not provided, the function will compute the minimum and maximum values from the input array.
        default: None.

    Returns
    -------
    Tuple[np.array, Dict[str, float]]
        A tuple containing the normalized numpy array (with dtype float32) and a dictionary containing the minimum
        and maximum values used for scaling. The minimum and maximum values are stored in the dictionary using the
        keys "min" and "max", respectively.

    Raises
    ------
    AssertionError
        If the input array is not a numpy array or if the stat_dict is not a dictionary or None.

    Notes
    -----
    This function uses the formula (arr - arr_min) / (arr_max - arr_min) to normalize the input array to a range
    of [min_val, max_val]. The minimum and maximum values of the input array can be passed through stat_dict,
    otherwise, they will be computed from the array.

    Examples
    --------
    ```python
    >>> arr = np.array([1, 2, 3, 4, 5])
    >>> scaler_to_range(arr, 0, 10)
    (array([0., 2., 4., 6., 10.], dtype=float32), {'min': 1, 'max': 5})
    ```
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

    result = np.multiply(result, max_val - min_val) + min_val

    return result, stat_dict


def scaler_truncate(
    arr: np.ndarray,
    trunc_min: Union[float, int] = None,
    trunc_max: Union[float, int] = None,
    target_min: Union[float, int] = 0.0,
    target_max: Union[float, int] = 1.0,
    stat_dict: Optional[dict] = None,
) -> Tuple[np.array, dict]:
    """
    Truncate an input numpy array within the given range, and then normalise it to the target range.

    Parameters
    ----------
    arr : np.ndarray
        The input numpy array to be truncated and normalized.

    trunc_min : float/int, optional
        The minimum value to truncate to. If not provided, truncation will not be applied to the lower end.

    trunc_max : float/int, optional
        The maximum value to truncate to. If not provided, truncation will not be applied to the upper end.

    target_min : float/int, optional
        The minimum value of the target range to normalize to. Default: 0.0.

    target_max : float/int, optional
        The maximum value of the target range to normalize to. Default: 1.0.

    stat_dict : Dict[str, float], optional
        A dictionary containing the minimum and maximum values of the input array. If not provided, the function
        will compute the minimum and maximum values from the truncated array.

    Returns
    -------
    Tuple[np.array, Dict[str, float]]
        A tuple containing the normalized numpy array (with dtype float32) and a dictionary containing the minimum
        and maximum values used for scaling. The minimum and maximum values are stored in the dictionary using the
        keys "arr_min" and "arr_max", respectively.

    Raises
    ------
    AssertionError
        If the input array is not a numpy array, or if the trunc_min is not less than trunc_max, or if the
        target_min is not less than target_max, or if the stat_dict is not a dictionary or None.
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
    else:
        stat_dict = {
            "trunc_min": trunc_min,
            "trunc_max": trunc_max,
            "target_min": target_min,
            "target_max": target_max,
        }

    truncated = arr
    if trunc_min is not None:
        truncated = np.where(arr > trunc_max, trunc_max, arr)

    if trunc_max is not None:
        truncated = np.where(truncated < trunc_min, trunc_min, truncated)

    if "arr_min" not in stat_dict:
        stat_dict["arr_min"] = np.min(truncated)
    if "arr_max" not in stat_dict:
        stat_dict["arr_max"] = np.max(truncated)

    truncated = truncated.astype(np.float32, copy=False)

    arr_min = stat_dict["arr_min"]
    arr_max = stat_dict["arr_max"]

    num = np.subtract(truncated, arr_min)
    den = np.subtract(arr_max, arr_min)

    result = np.zeros_like(arr, dtype="float32")
    np.divide(num, den, out=result, where=den != 0)

    result = np.multiply(result, target_max - target_min) + target_min

    return result, stat_dict
