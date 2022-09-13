"""
### Perform morphological operations on arrays and rasters.  ###
"""

# Standard library
import sys; sys.path.append("../../")

# Internal
from buteo.raster.convolution import convolve_array, get_kernel


def texture_local_variance(img, filter_size=5, spherical=False):
    """ Create a variance texture layer"""
    _kernel, weights, offsets = get_kernel(filter_size, distance_weight="linear", spherical=spherical)
    std = convolve_array(img, offsets, weights, "std")

    return std


def texture_local_median(img, filter_size=5, spherical=False):
    """ Create a variance texture layer"""
    _kernel, weights, offsets = get_kernel(filter_size, distance_weight="linear", spherical=spherical)
    median = convolve_array(img, offsets, weights, "median")

    return median


def texture_hole_dif(img, filter_size=5, spherical=False):
    """ Create a variance texture layer"""
    _kernel, weights, offsets = get_kernel(filter_size, distance_weight="linear", spherical=spherical, hole=True)
    hole = convolve_array(img, offsets, weights, "sum")

    hole_dif = img / hole

    return hole_dif
