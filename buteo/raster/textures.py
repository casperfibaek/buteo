"""
### Perform morphological operations on arrays and rasters.  ###
"""

# Standard library
import sys; sys.path.append("../../")

# Internal
from buteo.raster.convolution import convolve_array, get_kernel


def texture_local_variance(img, filter_size=5, spherical=False, nodata=False, nodata_value=9999.0, distance_weight="linear"):
    """ Create a variance texture layer. """
    _kernel, weights, offsets = get_kernel(filter_size, distance_weight=distance_weight, spherical=spherical)
    std = convolve_array(img, offsets, weights, "std", nodata=nodata, nodata_value=nodata_value)

    return std


def texture_local_median(img, filter_size=5, spherical=False, nodata=False, nodata_value=9999.0, distance_weight="linear"):
    """ Create a variance texture layer. """
    _kernel, weights, offsets = get_kernel(filter_size, distance_weight=distance_weight, spherical=spherical)
    median = convolve_array(img, offsets, weights, "median", nodata=nodata, nodata_value=nodata_value)

    return median


def texture_local_blur(img, filter_size=5, spherical=False, nodata=False, nodata_value=9999.0, distance_weight="linear"):
    """ Create a variance texture layer. """
    _kernel, weights, offsets = get_kernel(filter_size, distance_weight=distance_weight, spherical=spherical, hole=False)
    return convolve_array(img, offsets, weights, "sum", nodata=nodata, nodata_value=nodata_value)


def texture_hole_dif(img, filter_size=5, spherical=False, nodata=False, nodata_value=9999.0, distance_weight="linear"):
    """ Create a variance texture layer. """
    _kernel, weights, offsets = get_kernel(filter_size, distance_weight=distance_weight, spherical=spherical, hole=True)
    hole = convolve_array(img, offsets, weights, "sum", nodata=nodata, nodata_value=nodata_value)

    hole_dif = img / hole

    return hole_dif

def texture_local_mean_match(img1, img2, filter_size=11, spherical=True, nodata=False, nodata_value=9999.0):
    """ Match the local mean of two images. """
    _kernel, weights, offsets = get_kernel(filter_size, distance_weight=None, spherical=spherical)

    return convolve_array(img1, offsets, weights, method="match_mean", additional_array=img2, nodata=nodata, nodata_value=nodata_value)
