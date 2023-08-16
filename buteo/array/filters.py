"""
### Perform filter operations on arrays and rasters.  ###
"""

# Standard library
import sys; sys.path.append("../../")
from typing import Union, Optional


# External
import numpy as np

# Internal
from buteo.array.convolution import convolve_array
from buteo.array.convolution_kernels import kernel_base, kernel_get_offsets_and_weights
from buteo.utils.utils_base import _type_check


def filter_operation(
    arr: np.ndarray,
    method: int,
    radius: Union[int, float] = 1,
    spherical: bool = True,
    normalised: bool = True,
    hole: bool = False,
    func_value: Union[int, float] = 0.0,
    distance_weighted: bool = False,
    distance_method: int = 0,
    distance_decay: Union[int, float] = 0.2,
    distance_sigma: Union[int, float] = 2.0,
    kernel: Optional[np.ndarray] = None,
    channel_last: bool = True,
) -> np.ndarray:
    """ Internal function to perform filter operations on arrays and rasters. """
    _type_check(arr, [np.ndarray], "arr")
    _type_check(method, [int], "method")
    _type_check(radius, [int, float], "radius")
    _type_check(spherical, [bool], "spherical")
    _type_check(channel_last, [bool], "channel_last")
    _type_check(normalised, [bool], "normalised")
    _type_check(hole, [bool], "hole")
    _type_check(func_value, [int, float], "func_value")
    _type_check(distance_weighted, [bool], "distance_weighted")
    _type_check(distance_method, [int], "distance_method")
    _type_check(distance_decay, [int, float], "distance_decay")
    _type_check(distance_sigma, [int, float], "distance_sigma")

    if kernel is None:
        kernel = kernel_base(
            radius,
            circular=spherical,
            distance_weighted=distance_weighted,
            normalised=normalised,
            hole=hole,
            method=distance_method,
            decay=distance_decay,
            sigma=distance_sigma,
        )

    offsets, weights = kernel_get_offsets_and_weights(kernel)

    mask = None
    if np.ma.isMaskedArray(arr):
        nodata = True
        nodata_value = arr.fill_value
        arr = np.ma.getdata(arr)
        mask = np.ma.getmask(arr)
    else:
        nodata = False
        nodata_value = 0.0

    arr_convolved = convolve_array(
        arr,
        offsets,
        weights,
        method=method,
        nodata=nodata,
        nodata_value=nodata_value,
        func_value=func_value,
        channel_last=channel_last,
    )

    if nodata:
        arr_convolved = np.ma.array(arr_convolved, mask=mask, fill_value=nodata_value)

    return arr_convolved


_filter_operation = filter_operation


def filter_variance(
    arr: np.ndarray,
    radius: Union[int, float] = 1,
    spherical: bool = True,
    distance_weighted: bool = False,
    distance_method: int = 0,
    distance_decay: Union[int, float] = 0.2,
    distance_sigma: Union[int, float] = 2.0,
    channel_last: bool = True,
):
    """
    Calculate the variance of the array using a weighted moving window.

    Parameters
    ----------
    arr : np.ndarray
        Array to calculate variance on.
    
    radius : Union[int, float], optional
        Radius of the moving window, can be fractional. Default: 1

    spherical : bool, optional
        Whether to use a spherical moving window, default: True.

    distance_weighted : bool, optional
        Whether to weight the moving window by distance, default: False.

    distance_method : int, optional
        Method to use for distance weighting, default: 0.

    distance_decay : Union[int, float], optional
        Decay rate for distance weighting, default: 0.2.

    distance_sigma : Union[int, float], optional
        Sigma for distance weighting, default: 2.0.

    channel_last : bool, optional
        Whether the channels are the last dimension, default: True.

    Returns
    -------
    np.ndarray
        The variance filtered array.
    """
    return _filter_operation(
        arr,
        method=6,
        radius=radius,
        normalised=True,
        spherical=spherical,
        distance_weighted=distance_weighted,
        distance_method=distance_method,
        distance_decay=distance_decay,
        distance_sigma=distance_sigma,
        channel_last=channel_last,
    )


def filter_standard_deviation(
    arr: np.ndarray,
    radius: Union[int, float] = 1,
    spherical: bool = True,
    distance_weighted: bool = False,
    distance_method: int = 0,
    distance_decay: Union[int, float] = 0.2,
    distance_sigma: Union[int, float] = 2.0,
    channel_last: bool = True,
):
    """
    Calculate the standard deviation of the array using a weighted moving window.

    Parameters
    ----------
    arr : np.ndarray
        Array to calculate standard deviation on.
    
    radius : Union[int, float], optional
        Radius of the moving window, can be fractional. Default: 1

    spherical : bool, optional
        Whether to use a spherical moving window, default: True.

    distance_weighted : bool, optional
        Whether to weight the moving window by distance, default: False.

    distance_method : int, optional
        Method to use for distance weighting, default: 0.

    distance_decay : Union[int, float], optional
        Decay rate for distance weighting, default: 0.2.

    distance_sigma : Union[int, float], optional
        Sigma for distance weighting, default: 2.0.

    channel_last : bool, optional
        Whether the channels are the last dimension, default: True.

    Returns
    -------
    np.ndarray
        The standard deviation filtered array.
    """
    return _filter_operation(
        arr,
        method=7,
        radius=radius,
        normalised=True,
        spherical=spherical,
        distance_weighted=distance_weighted,
        distance_method=distance_method,
        distance_decay=distance_decay,
        distance_sigma=distance_sigma,
        channel_last=channel_last,
    )


def filter_blur(
    arr: np.ndarray,
    radius: Union[int, float] = 1,
    spherical: bool = True,
    distance_weighted: bool = False,
    distance_method: int = 0,
    distance_decay: Union[int, float] = 0.2,
    distance_sigma: Union[int, float] = 2.0,
    channel_last: bool = True,
):
    """
    Blur the array using a weighted moving window.

    Parameters
    ----------
    arr : np.ndarray
        Array to blur.
    radius : Union[int, float], optional
        Radius of the moving window, can be fractional. Default: 1
    spherical : bool, optional
        Whether to use a spherical moving window, default: True.
    distance_weighted : bool, optional
        Whether to weight the moving window by distance, default: False.
    distance_method : int, optional
        Method to use for distance weighting, default: 0.
    distance_decay : Union[int, float], optional
        Decay rate for distance weighting, default: 0.2.
    distance_sigma : Union[int, float], optional
        Sigma for distance weighting, default: 2.0.
    channel_last : bool, optional
        Whether the channels are the last dimension, default: True.

    Returns
    -------
    np.ndarray
        The blurred array.
    """
    return _filter_operation(
        arr,
        method=1,
        radius=radius,
        normalised=True,
        spherical=spherical,
        distance_weighted=distance_weighted,
        distance_method=distance_method,
        distance_decay=distance_decay,
        distance_sigma=distance_sigma,
        channel_last=channel_last,
    )


def filter_median(
    arr: np.ndarray,
    radius: Union[int, float] = 1,
    spherical: bool = True,
    distance_weighted: bool = False,
    distance_method: int = 0,
    distance_decay: Union[int, float] = 0.2,
    distance_sigma: Union[int, float] = 2.0,
    channel_last: bool = True,
):
    """
    Calculate the median of the array using a weighted moving window.

    Parameters
    ----------
    arr : np.ndarray
        Array to blur.
    radius : Union[int, float], optional
        Radius of the moving window, can be fractional. Default: 1
    spherical : bool, optional
        Whether to use a spherical moving window, default: True.
    distance_weighted : bool, optional
        Whether to weight the moving window by distance, default: False.
    distance_method : int, optional
        Method to use for distance weighting, default: 0.
    distance_decay : Union[int, float], optional
        Decay rate for distance weighting, default: 0.2.
    distance_sigma : Union[int, float], optional
        Sigma for distance weighting, default: 2.0.
    channel_last : bool, optional
        Whether the channels are the last dimension, default: True.

    Returns
    -------
    np.ndarray
        The median filtered array.
    """
    return _filter_operation(
        arr,
        method=5,
        radius=radius,
        normalised=True,
        spherical=spherical,
        distance_weighted=distance_weighted,
        distance_method=distance_method,
        distance_decay=distance_decay,
        distance_sigma=distance_sigma,
        channel_last=channel_last,
    )


def filter_min(
    arr: np.ndarray,
    radius: Union[int, float] = 1,
    spherical: bool = True,
    distance_weighted: bool = False,
    distance_method: int = 0,
    distance_decay: Union[int, float] = 0.2,
    distance_sigma: Union[int, float] = 2.0,
    channel_last: bool = True,
):
    """
    Takes the min of the array using a weighted moving window.

    Parameters
    ----------
    arr : np.ndarray
        Array to blur.
    radius : Union[int, float], optional
        Radius of the moving window, can be fractional. Default: 1
    spherical : bool, optional
        Whether to use a spherical moving window, default: True.
    distance_weighted : bool, optional
        Whether to weight the moving window by distance, default: False.
    distance_method : int, optional
        Method to use for distance weighting, default: 0.
    distance_decay : Union[int, float], optional
        Decay rate for distance weighting, default: 0.2.
    distance_sigma : Union[int, float], optional
        Sigma for distance weighting, default: 2.0.
    channel_last : bool, optional
        Whether the channels are the last dimension, default: True.

    Returns
    -------
    np.ndarray
        The min filtered array.
    """
    return _filter_operation(
        arr,
        method=3,
        radius=radius,
        normalised=False,
        spherical=spherical,
        distance_weighted=distance_weighted,
        distance_method=distance_method,
        distance_decay=distance_decay,
        distance_sigma=distance_sigma,
        channel_last=channel_last,
    )


def filter_max(
    arr: np.ndarray,
    radius: Union[int, float] = 1,
    spherical: bool = True,
    distance_weighted: bool = False,
    distance_method: int = 0,
    distance_decay: Union[int, float] = 0.2,
    distance_sigma: Union[int, float] = 2.0,
    channel_last: bool = True,
):
    """
    Takes the max of the array using a weighted moving window.

    Parameters
    ----------
    arr : np.ndarray
        Array to blur.
    radius : Union[int, float], optional
        Radius of the moving window, can be fractional. Default: 1
    spherical : bool, optional
        Whether to use a spherical moving window, default: True.
    distance_weighted : bool, optional
        Whether to weight the moving window by distance, default: False.
    distance_method : int, optional
        Method to use for distance weighting, default: 0.
    distance_decay : Union[int, float], optional
        Decay rate for distance weighting, default: 0.2.
    distance_sigma : Union[int, float], optional
        Sigma for distance weighting, default: 2.0.
    channel_last : bool, optional
        Whether the channels are the last dimension, default: True.

    Returns
    -------
    np.ndarray
        The max filtered array.
    """
    return _filter_operation(
        arr,
        method=2,
        radius=radius,
        normalised=False,
        spherical=spherical,
        distance_weighted=distance_weighted,
        distance_method=distance_method,
        distance_decay=distance_decay,
        distance_sigma=distance_sigma,
        channel_last=channel_last,
    )


def filter_sum(
    arr: np.ndarray,
    radius: Union[int, float] = 1,
    spherical: bool = True,
    distance_weighted: bool = False,
    distance_method: int = 0,
    distance_decay: Union[int, float] = 0.2,
    distance_sigma: Union[int, float] = 2.0,
    channel_last: bool = True,
):
    """
    Takes the sum of the array using a weighted moving window.

    Parameters
    ----------
    arr : np.ndarray
        Array to blur.
    radius : Union[int, float], optional
        Radius of the moving window, can be fractional. Default: 1
    spherical : bool, optional
        Whether to use a spherical moving window, default: True.
    distance_weighted : bool, optional
        Whether to weight the moving window by distance, default: False.
    distance_method : int, optional
        Method to use for distance weighting, default: 0.
    distance_decay : Union[int, float], optional
        Decay rate for distance weighting, default: 0.2.
    distance_sigma : Union[int, float], optional
        Sigma for distance weighting, default: 2.0.
    channel_last : bool, optional
        Whether the channels are the last dimension, default: True.

    Returns
    -------
    np.ndarray
        The sum filtered array.
    """
    return _filter_operation(
        arr,
        method=1,
        radius=radius,
        normalised=False,
        spherical=spherical,
        distance_weighted=distance_weighted,
        distance_method=distance_method,
        distance_decay=distance_decay,
        distance_sigma=distance_sigma,
        channel_last=channel_last,
    )


def filter_mode(
    arr: np.ndarray,
    radius: Union[int, float] = 1,
    spherical: bool = True,
    distance_weighted: bool = False,
    distance_method: int = 0,
    distance_decay: Union[int, float] = 0.2,
    distance_sigma: Union[int, float] = 2.0,
    channel_last: bool = True,
):
    """
    Takes the mode of the array using a weighted moving window.

    Parameters
    ----------
    arr : np.ndarray
        Array to blur.
    radius : Union[int, float], optional
        Radius of the moving window, can be fractional. Default: 1
    spherical : bool, optional
        Whether to use a spherical moving window, default: True.
    distance_weighted : bool, optional
        Whether to weight the moving window by distance, default: False.
    distance_method : int, optional
        Method to use for distance weighting, default: 0.
    distance_decay : Union[int, float], optional
        Decay rate for distance weighting, default: 0.2.
    distance_sigma : Union[int, float], optional
        Sigma for distance weighting, default: 2.0.
    channel_last : bool, optional
        Whether the channels are the last dimension, default: True.

    Returns
    -------
    np.ndarray
        The mode filtered array.
    """
    return _filter_operation(
        arr,
        method=9,
        radius=radius,
        normalised=False,
        spherical=spherical,
        distance_weighted=distance_weighted,
        distance_method=distance_method,
        distance_decay=distance_decay,
        distance_sigma=distance_sigma,
        channel_last=channel_last,
    )


def filter_center_difference(
    arr: np.ndarray,
    radius: Union[int, float] = 1,
    spherical: bool = True,
    distance_weighted: bool = False,
    distance_method: int = 0,
    distance_decay: Union[int, float] = 0.2,
    distance_sigma: Union[int, float] = 2.0,
    channel_last: bool = True,
):
    """
    Take the difference from the center to the surrounding values of the array using a weighted moving window.

    Parameters
    ----------
    arr : np.ndarray
        Array to blur.
    radius : Union[int, float], optional
        Radius of the moving window, can be fractional. Default: 1
    spherical : bool, optional
        Whether to use a spherical moving window, default: True.
    distance_weighted : bool, optional
        Whether to weight the moving window by distance, default: False.
    distance_method : int, optional
        Method to use for distance weighting, default: 0.
    distance_decay : Union[int, float], optional
        Decay rate for distance weighting, default: 0.2.
    distance_sigma : Union[int, float], optional
        Sigma for distance weighting, default: 2.0.
    channel_last : bool, optional
        Whether the channels are the last dimension, default: True.

    Returns
    -------
    np.ndarray
        The center difference filtered array.
    """
    convolved = _filter_operation(
        arr,
        method=1,
        radius=radius,
        normalised=True,
        hole=True,
        spherical=spherical,
        distance_weighted=distance_weighted,
        distance_method=distance_method,
        distance_decay=distance_decay,
        distance_sigma=distance_sigma,
        channel_last=channel_last,
    )

    return arr - convolved
