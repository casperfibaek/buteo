"""
### Perform morphological operations on arrays and rasters.  ###
"""

# Standard library
import sys; sys.path.append("../../")

# Internal
from buteo.raster.convolution import get_kernel, convolve_array


def morph_erode(arr, filter_size=5, spherical=False, distance_weight=False, distance_decay=0.2, distance_sigma=1.0):
    """ Erode an array by taking the local minimum. """

    _kernel, weights, offsets = get_kernel(filter_size, arr.shape[-1], spherical=spherical, normalise=False, distance_weight=distance_weight, distance_decay=distance_decay, distance_sigma=distance_sigma)
    return convolve_array(arr, offsets, weights, method="min")


def morph_dilate(arr, filter_size=5, spherical=False, distance_weight=False, distance_decay=0.2, distance_sigma=1.0):
    """ Dilate an array by taking the local maximum. """

    _kernel, weights, offsets = get_kernel(filter_size, arr.shape[-1], spherical=spherical, normalise=False, distance_weight=distance_weight, distance_decay=distance_decay, distance_sigma=distance_sigma)
    return convolve_array(arr, offsets, weights, method="max")


def morph_open(arr, filter_size=5, spherical=False, distance_weight=False, distance_decay=0.2, distance_sigma=1.0):
    """ Perform the open mortholigical operation on an array. """

    _kernel, weights, offsets = get_kernel(filter_size, arr.shape[-1], spherical=spherical, normalise=False, distance_weight=distance_weight, distance_decay=distance_decay, distance_sigma=distance_sigma)
    eroded = convolve_array(arr, offsets, weights, method="min")
    return convolve_array(eroded, offsets, weights, method="max")


def morph_close(arr, filter_size=5, spherical=False, distance_weight=False, distance_decay=0.2, distance_sigma=1.0):
    """ Perform the close morphological operation on an array. """

    _kernel, weights, offsets = get_kernel(filter_size, arr.shape[-1], spherical=spherical, normalise=False, distance_weight=distance_weight, distance_decay=distance_decay, distance_sigma=distance_sigma)
    dilate = convolve_array(arr, offsets, weights, method="max")
    return convolve_array(dilate, offsets, weights, method="min")


def morph_tophat(arr, filter_size=5, spherical=False, distance_weight=False, distance_decay=0.2, distance_sigma=1.0):
    """ Perform the top_hat morphological operation on the array. """

    _kernel, weights, offsets = get_kernel(filter_size, arr.shape[-1], spherical=spherical, normalise=False, distance_weight=distance_weight, distance_decay=distance_decay, distance_sigma=distance_sigma)
    eroded = convolve_array(arr, offsets, weights, method="min")
    opened = convolve_array(eroded, offsets, weights, method="max")
    # return arr - opened
    return arr / opened


def morph_bothat(arr, filter_size=5, spherical=False, distance_weight=False, distance_decay=0.2, distance_sigma=1.0):
    """ Perform the bottom_hat morphological operation on the array. """

    _kernel, weights, offsets = get_kernel(filter_size, arr.shape[-1], spherical=spherical, normalise=False, distance_weight=distance_weight, distance_decay=distance_decay, distance_sigma=distance_sigma)
    dilated = convolve_array(arr, offsets, weights, method="max")
    closed = convolve_array(dilated, offsets, weights, method="min")
    # return closed - arr
    return closed / arr


def morph_difference(arr, filter_size=5, spherical=False, distance_weight=False, distance_decay=0.2, distance_sigma=1.0):
    """ Perform the difference morphological operation on the array. """

    _kernel, weights, offsets = get_kernel(filter_size, arr.shape[-1], spherical=spherical, normalise=False, distance_weight=distance_weight, distance_decay=distance_decay, distance_sigma=distance_sigma)
    erode = convolve_array(arr, offsets, weights, method="min")
    dilate = convolve_array(arr, offsets, weights, method="max")

    return dilate - erode
