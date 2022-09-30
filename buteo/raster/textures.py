"""
### Perform morphological operations on arrays and rasters.  ###
"""

# Standard library
import sys; sys.path.append("../../")

# Internal
from buteo.raster.convolution import convolve_array, get_kernel


def texture_local_variance(img, filter_size=5, spherical=False, nodata=False, nodata_value=9999.0):
    """ Create a variance texture layer"""
    _kernel, weights, offsets = get_kernel(filter_size, distance_weight="linear", spherical=spherical)
    std = convolve_array(img, offsets, weights, "std", nodata=nodata, nodata_value=nodata_value)

    return std


def texture_local_median(img, filter_size=5, spherical=False, nodata=False, nodata_value=9999.0):
    """ Create a variance texture layer"""
    _kernel, weights, offsets = get_kernel(filter_size, distance_weight="linear", spherical=spherical)
    median = convolve_array(img, offsets, weights, "median", nodata=nodata, nodata_value=nodata_value)

    return median


def texture_local_blur(img, filter_size=5, spherical=False, nodata=False, nodata_value=9999.0):
    """ Create a variance texture layer"""
    _kernel, weights, offsets = get_kernel(filter_size, distance_weight="linear", spherical=spherical, hole=False)
    return convolve_array(img, offsets, weights, "sum", nodata=nodata, nodata_value=nodata_value)


def texture_hole_dif(img, filter_size=5, spherical=False, nodata=False, nodata_value=9999.0):
    """ Create a variance texture layer"""
    _kernel, weights, offsets = get_kernel(filter_size, distance_weight="linear", spherical=spherical, hole=True)
    hole = convolve_array(img, offsets, weights, "sum", nodata=nodata, nodata_value=nodata_value)

    hole_dif = img / hole

    return hole_dif
