"""
### Perform morphological operations on arrays and rasters.  ###
"""

# Standard library
import sys; sys.path.append("../../")
from typing import Optional, Union

# External
import numpy as np

# Internal
from buteo.array.convolution import convolve_array, get_kernel, _METHOD_ENUMS
from buteo.utils import type_check

def texture_local_variance(
    arr,
    filter_size=5,
    spherical=False,
    nodata=False,
    nodata_value=9999.0,
    distance_weight="linear",
    distance_decay=0.2,
    distance_sigma=1,
):
    """
    Create a variance texture layer.

    ## Args:
    `arr` (_np.ndarray_): The array on which to calculate the filter.\n

    ## Kwargs:
    `filter_size` (_int_): The size of the kernel to use. filter_size x filter_size. (Default: **5**)\n
    `spherical` (_bool_): If True, the filter applied will be weighted by a circle (Default: **False**)\n
    `nodata` (_bool_): Does the array contain nodata and should the values be left? (Default: **False**)\n
    `nodata_value` (_bool_): If nodata is True, what value is nodata. (Default: **9999.9**)\n
    `distance_weight` (_str_): How should the distance from the center be treated: (Default: **"linear"**)\n
        * `"none"`: no distance weighing will be done.\n
        * `"linear"`: np.power((1 - decay), normed).\n
        * `"sqrt"`: np.power(np.sqrt((1 - decay)), normed).\n
        * `"power"`: np.power(np.power((1 - decay), 2), normed).\n
        * `"log"`: np.log(normed + 2).\n
        * `"gaussian"`: np.exp(-(np.power(normed, 2)) / (2 * np.power(sigma, 2))).\n
    `distance_decay` (_float_): Rate of distance decay. (Default: **0.2**)\n
    `distance_sigma` (_float_): The sigma to use for gaussian decay. (Default: **1.0**)\n

    ## Returns:
    (_np.ndarray_): The filtered array.
    """
    type_check(arr, [np.ndarray], "arr")
    type_check(filter_size, [int], "filter_size")
    type_check(spherical, [bool], "spherical")
    type_check(nodata, [bool], "nodata")
    type_check(nodata_value, [int, float], "nodata_value")
    type_check(distance_weight, [str], "distance_weight")
    type_check(distance_decay, [int, float], "distance_decay")
    type_check(distance_sigma, [int, float], "distance_sigma")

    assert filter_size % 2 == 1, "Filter size must be odd."

    _kernel, weights, offsets = get_kernel(
        filter_size,
        distance_weight=distance_weight,
        distance_decay=distance_decay,
        distance_sigma=distance_sigma,
        spherical=spherical,
    )

    std = convolve_array(
        arr,
        offsets,
        weights,
        _METHOD_ENUMS["std"],
        nodata=nodata,
        nodata_value=nodata_value,
    )

    return std


def texture_local_median(
    arr,
    filter_size=5,
    spherical=False,
    nodata=False,
    nodata_value=9999.0,
    distance_weight="linear",
    distance_decay=0.2,
    distance_sigma=1,
):
    """
    Create a median filtered array.

    ## Args:
    `arr` (_np.ndarray_): The array on which to calculate the filter.\n

    ## Kwargs:
    `filter_size` (_int_): The size of the kernel to use. filter_size x filter_size. (Default: **5**)\n
    `spherical` (_bool_): If True, the filter applied will be weighted by a circle (Default: **False**)\n
    `nodata` (_bool_): Does the array contain nodata and should the values be left? (Default: **False**)\n
    `nodata_value` (_bool_): If nodata is True, what value is nodata. (Default: **9999.9**)\n
    `distance_weight` (_str_): How should the distance from the center be treated: (Default: **"linear"**)\n
        * `"none"`: no distance weighing will be done.\n
        * `"linear"`: np.power((1 - decay), normed).\n
        * `"sqrt"`: np.power(np.sqrt((1 - decay)), normed).\n
        * `"power"`: np.power(np.power((1 - decay), 2), normed).\n
        * `"log"`: np.log(normed + 2).\n
        * `"gaussian"`: np.exp(-(np.power(normed, 2)) / (2 * np.power(sigma, 2))).\n
    `distance_decay` (_float_): Rate of distance decay. (Default: **0.2**)\n
    `distance_sigma` (_float_): The sigma to use for gaussian decay. (Default: **1.0**)\n

    ## Returns:
    (_np.ndarray_): The filtered array.
    """
    type_check(arr, [np.ndarray], "arr")
    type_check(filter_size, [int], "filter_size")
    type_check(spherical, [bool], "spherical")
    type_check(nodata, [bool], "nodata")
    type_check(nodata_value, [int, float], "nodata_value")
    type_check(distance_weight, [str], "distance_weight")
    type_check(distance_decay, [int, float], "distance_decay")
    type_check(distance_sigma, [int, float], "distance_sigma")

    assert filter_size % 2 == 1, "Filter size must be odd."

    _kernel, weights, offsets = get_kernel(
        filter_size,
        distance_weight=distance_weight,
        distance_decay=distance_decay,
        distance_sigma=distance_sigma,
        spherical=spherical,
    )

    median = convolve_array(
        arr,
        offsets,
        weights,
        _METHOD_ENUMS["median"],
        nodata=nodata,
        nodata_value=nodata_value,
    )

    return median


def texture_local_blur(
    arr,
    filter_size=5,
    spherical=False,
    nodata=False,
    nodata_value=9999.0,
    distance_weight="linear",
    distance_decay=0.2,
    distance_sigma=1,
):
    """
    Apply a blurring filter. Default is a square linear distance weighted blur.

    ## Args:
    `arr` (_np.ndarray_): The array on which to calculate the filter.\n

    ## Kwargs:
    `filter_size` (_int_): The size of the kernel to use. filter_size x filter_size. (Default: **5**)\n
    `spherical` (_bool_): If True, the filter applied will be weighted by a circle (Default: **False**)\n
    `nodata` (_bool_): Does the array contain nodata and should the values be left? (Default: **False**)\n
    `nodata_value` (_bool_): If nodata is True, what value is nodata. (Default: **9999.9**)\n
    `distance_weight` (_str_): How should the distance from the center be treated: (Default: **"linear"**)\n
        * `"none"`: no distance weighing will be done.\n
        * `"linear"`: np.power((1 - decay), normed).\n
        * `"sqrt"`: np.power(np.sqrt((1 - decay)), normed).\n
        * `"power"`: np.power(np.power((1 - decay), 2), normed).\n
        * `"log"`: np.log(normed + 2).\n
        * `"gaussian"`: np.exp(-(np.power(normed, 2)) / (2 * np.power(sigma, 2))).\n
    `distance_decay` (_float_): Rate of distance decay. (Default: **0.2**)\n
    `distance_sigma` (_float_): The sigma to use for gaussian decay. (Default: **1.0**)\n

    ## Returns:
    (_np.ndarray_): The filtered array.
    """
    type_check(arr, [np.ndarray], "arr")
    type_check(filter_size, [int], "filter_size")
    type_check(spherical, [bool], "spherical")
    type_check(nodata, [bool], "nodata")
    type_check(nodata_value, [int, float], "nodata_value")
    type_check(distance_weight, [str], "distance_weight")
    type_check(distance_decay, [int, float], "distance_decay")
    type_check(distance_sigma, [int, float], "distance_sigma")

    assert filter_size % 2 == 1, "Filter size must be odd."

    _kernel, weights, offsets = get_kernel(
        filter_size,
        distance_weight=distance_weight,
        distance_decay=distance_decay,
        distance_sigma=distance_sigma,
        spherical=spherical,
    )

    return convolve_array(
        arr,
        offsets,
        weights,
        _METHOD_ENUMS["sum"],
        nodata=nodata,
        nodata_value=nodata_value,
    )


def texture_local_mode(
    arr: np.ndarray,
    filter_size: int = 5,
    spherical: bool = True,
    nodata: bool = False,
    nodata_value: float = 9999.0,
    distance_weight: str = "none",
    distance_decay: float = 0.2,
    distance_sigma: float = 1,
) -> np.ndarray:
    """
    Apply a mode filter. Default is a circular filter with no distance decay.

    Args:
        arr (np.ndarray): The array on which to calculate the filter.

    Keyword Args:
        filter_size (int=5): The size of the kernel to use. filter_size x filter_size.
        spherical (bool=True): If True, the filter applied will be weighted by a circle.
        nodata (bool=False): Does the array contain nodata and should the values be left?
        nodata_value (float=9999.0): If nodata is True, what value is nodata.
        distance_weight (str='none'): How should the distance from the center be treated:
            * "none": no distance weighing will be done.
            * "linear": np.power((1 - decay), normed).
            * "sqrt": np.power(np.sqrt((1 - decay)), normed).
            * "power": np.power(np.power((1 - decay), 2), normed).
            * "log": np.log(normed + 2).
            * "gaussian": np.exp(-(np.power(normed, 2)) / (2 * np.power(sigma, 2))).
        distance_decay (float=0.2): Rate of distance decay.
        distance_sigma (float=1.0): The sigma to use for gaussian decay.

    Returns:
        np.ndarray: The filtered array.
    """
    type_check(arr, [np.ndarray], "arr")
    type_check(filter_size, [int], "filter_size")
    type_check(spherical, [bool], "spherical")
    type_check(nodata, [bool], "nodata")
    type_check(nodata_value, [int, float], "nodata_value")
    type_check(distance_weight, [str], "distance_weight")
    type_check(distance_decay, [int, float], "distance_decay")
    type_check(distance_sigma, [int, float], "distance_sigma")

    assert filter_size % 2 == 1, "Filter size must be odd."

    _kernel, weights, offsets = get_kernel(
        filter_size,
        distance_weight=distance_weight,
        distance_decay=distance_decay,
        distance_sigma=distance_sigma,
        spherical=spherical,
    )

    return convolve_array(
        arr,
        offsets,
        weights,
        _METHOD_ENUMS["mode"],
        nodata=nodata,
        nodata_value=nodata_value,
    )


def texture_hole_dif(
    arr: np.ndarray,
    filter_size: int = 5,
    spherical: bool = False,
    nodata: bool = False,
    nodata_value: float = 9999.0,
    distance_weight: str = "linear",
    distance_decay: float = 0.2,
    distance_sigma: float = 1,
) -> np.ndarray:
    """
    Create a 'hole' filter, representing the difference between a pixel and its surrounding neighbourhood.

    Args:
        arr (np.ndarray): The array on which to calculate the filter.

    Keyword Args:
        filter_size (int=5): The size of the kernel to use. filter_size x filter_size.
        spherical (bool=False): If True, the filter applied will be weighted by a circle.
        nodata (bool=False): Does the array contain nodata and should the values be left?
        nodata_value (float=9999.0): If nodata is True, what value is nodata.
        distance_weight (str='linear'): How should the distance from the center be treated:
            * "none": no distance weighing will be done.
            * "linear": np.power((1 - decay), normed).
            * "sqrt": np.power(np.sqrt((1 - decay)), normed).
            * "power": np.power(np.power((1 - decay), 2), normed).
            * "log": np.log(normed + 2).
            * "gaussian": np.exp(-(np.power(normed, 2)) / (2 * np.power(sigma, 2))).
        distance_decay (float=0.2): Rate of distance decay.
        distance_sigma (float=1.0): The sigma to use for gaussian decay.

    Returns:
        np.ndarray: The filtered array.
    """
    type_check(arr, [np.ndarray], "arr")
    type_check(filter_size, [int], "filter_size")
    type_check(spherical, [bool], "spherical")
    type_check(nodata, [bool], "nodata")
    type_check(nodata_value, [int, float], "nodata_value")
    type_check(distance_weight, [str], "distance_weight")
    type_check(distance_decay, [int, float], "distance_decay")
    type_check(distance_sigma, [int, float], "distance_sigma")

    assert filter_size % 2 == 1, "Filter size must be odd."


    _kernel, weights, offsets = get_kernel(
        filter_size,
        distance_weight=distance_weight,
        distance_decay=distance_decay,
        distance_sigma=distance_sigma,
        spherical=spherical,
        hole=True,
    )

    hole = convolve_array(
        arr,
        offsets,
        weights,
        _METHOD_ENUMS["sum"],
        nodata=nodata,
        nodata_value=nodata_value,
    )

    hole_dif = arr / hole

    return hole_dif
