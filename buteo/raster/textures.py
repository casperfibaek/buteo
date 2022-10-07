"""
### Perform morphological operations on arrays and rasters.  ###
"""

# Standard library
import sys; sys.path.append("../../")

# Internal
from buteo.raster.convolution import convolve_array, get_kernel


def texture_local_variance(arr, filter_size=5, spherical=False, nodata=False, nodata_value=9999.0, distance_weight="linear", distance_decay=0.2, distance_sigma=1):
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
    _kernel, weights, offsets = get_kernel(filter_size, distance_weight=distance_weight, distance_decay=distance_decay, distance_sigma=distance_sigma, spherical=spherical)
    std = convolve_array(arr, offsets, weights, "std", nodata=nodata, nodata_value=nodata_value)

    return std


def texture_local_median(img, filter_size=5, spherical=False, nodata=False, nodata_value=9999.0, distance_weight="linear", distance_decay=0.2, distance_sigma=1):
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
    _kernel, weights, offsets = get_kernel(filter_size, distance_weight=distance_weight, distance_decay=distance_decay, distance_sigma=distance_sigma, spherical=spherical)
    median = convolve_array(img, offsets, weights, "median", nodata=nodata, nodata_value=nodata_value)

    return median


def texture_local_blur(img, filter_size=5, spherical=False, nodata=False, nodata_value=9999.0, distance_weight="linear", distance_decay=0.2, distance_sigma=1):
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
    _kernel, weights, offsets = get_kernel(filter_size, distance_weight=distance_weight, distance_decay=distance_decay, distance_sigma=distance_sigma, spherical=spherical)
    return convolve_array(img, offsets, weights, "sum", nodata=nodata, nodata_value=nodata_value)


def texture_hole_dif(img, filter_size=5, spherical=False, nodata=False, nodata_value=9999.0, distance_weight="linear", distance_decay=0.2, distance_sigma=1):
    """
    Create a 'hole' filter, representing the difference between a pixel and its surrounding neighbourhood.

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
    _kernel, weights, offsets = get_kernel(filter_size, distance_weight=distance_weight, distance_decay=distance_decay, distance_sigma=distance_sigma, spherical=spherical, hole=True)
    hole = convolve_array(img, offsets, weights, "sum", nodata=nodata, nodata_value=nodata_value)

    hole_dif = img / hole

    return hole_dif

# def texture_local_mean_match(img1, img2, filter_size=11, spherical=True, nodata=False, nodata_value=9999.0):
#     """ Match the local mean of two images. """
#     _kernel, weights, offsets = get_kernel(filter_size, distance_weight=None, spherical=spherical)

#     return convolve_array(img1, offsets, weights, method="match_mean", additional_array=img2, nodata=nodata, nodata_value=nodata_value)
