import sys

sys.path.append("../../")

import numpy as np
from numba import jit, prange
from buteo.earth_observation.s2_utils import get_metadata
from buteo.raster.io import raster_to_array
from buteo.filters.kernel_generator import create_kernel


@jit(nopython=True, parallel=True, nogil=True, fastmath=True)
def weighted_quantile_2d(values, weights, quant):
    values_ravel = np.ravel(values)
    weights_ravel = np.ravel(weights)
    sort_mask = np.argsort(values_ravel)
    sorted_data = values_ravel[sort_mask]
    sorted_weights = weights_ravel[sort_mask]
    cumsum = np.cumsum(sorted_weights)
    intersect = (cumsum - 0.5 * sorted_weights) / cumsum[-1]
    return np.interp(quant, intersect, sorted_data)


def weighted_std(values, weights):
    average = np.average(values, weights=weights)
    variance = np.average((values - average) ** 2, weights=weights)

    return np.sqrt(variance)


@jit(nopython=True, parallel=True, nogil=True, fastmath=True)
def _feather_2d(arr, offsets, values_to_count):
    x_adj = arr.shape[0] - 1
    y_adj = arr.shape[1] - 1

    total_values = len(values_to_count)
    hood_size = len(offsets)
    result = np.empty((arr.shape[0], arr.shape[1], total_values), dtype="float32")

    for x in prange(arr.shape[0]):
        for y in range(arr.shape[1]):
            for z in range(total_values):

                hood_values = np.zeros(hood_size, dtype="float32")

                for n in range(hood_size):
                    offset_x = x + offsets[n][0]
                    offset_y = y + offsets[n][1]

                    if offset_x < 0:
                        offset_x = 0
                    elif offset_x > x_adj:
                        offset_x = x_adj

                    if offset_y < 0:
                        offset_y = 0
                    elif offset_y > y_adj:
                        offset_y = y_adj

                    hood_values[n] = arr[offset_x, offset_y]

                occurrances = 0
                for i in range(hood_size):
                    if hood_values[i] == values_to_count[z]:
                        occurrances += 1

                result[x][y][z] = occurrances / hood_size

    return result


def feather(arr, values_to_count, size=11):
    _kernel, offsets, _weights = create_kernel(
        (size, size),
        spherical=True,
        edge_weights=False,
        offsets=True,
        normalised=False,
        distance_calc=False,
        output_2d=True,
        remove_zero_weights=True,
    )

    return _feather_2d(arr, offsets, values_to_count)


@jit(nopython=True, parallel=True, nogil=True, fastmath=True)
def convolve_2D(
    arr,
    offsets,
    operation="dilate",
):
    x_adj = arr.shape[0] - 1
    y_adj = arr.shape[1] - 1

    hood_size = len(offsets)
    result = np.empty_like(arr)

    for x in prange(arr.shape[0]):
        for y in range(arr.shape[1]):

            hood_values = np.zeros(hood_size, dtype="float32")

            for n in range(hood_size):
                offset_x = x + offsets[n][0]
                offset_y = y + offsets[n][1]

                if offset_x < 0:
                    offset_x = 0
                elif offset_x > x_adj:
                    offset_x = x_adj

                if offset_y < 0:
                    offset_y = 0
                elif offset_y > y_adj:
                    offset_y = y_adj

                hood_values[n] = arr[offset_x, offset_y]

            if operation == "dilate":
                result[x, y] = np.max(hood_values)
            elif operation == "erode":
                result[x, y] = np.min(hood_values)

    return result


def smooth_quality(quality, dilate_size=5, erode_size=15):
    _dilate_kernel, dilate_offsets, _dilate_weights = create_kernel(
        (dilate_size, dilate_size),
        spherical=True,
        edge_weights=False,
        offsets=True,
        normalised=False,
        distance_calc=False,
        output_2d=True,
        remove_zero_weights=True,
    )

    _erode_kernel, erode_offsets, _erode_weights = create_kernel(
        (erode_size, erode_size),
        spherical=True,
        edge_weights=False,
        offsets=True,
        normalised=False,
        distance_calc=False,
        output_2d=True,
        remove_zero_weights=True,
    )

    dilated = convolve_2D(quality, dilate_offsets, operation="dilate")
    eroded = convolve_2D(dilated, erode_offsets, operation="erode")

    return eroded


def smooth_mask(mask, dilate_size=7, erode_size=7):
    _erode_kernel, erode_offsets, _erode_weights = create_kernel(
        (erode_size, erode_size),
        spherical=True,
        edge_weights=False,
        offsets=True,
        normalised=False,
        distance_calc=False,
        output_2d=True,
        remove_zero_weights=True,
    )

    _dilate_kernel, dilate_offsets, _dilate_weights = create_kernel(
        (dilate_size, dilate_size),
        spherical=True,
        edge_weights=False,
        offsets=True,
        normalised=False,
        distance_calc=False,
        output_2d=True,
        remove_zero_weights=True,
    )

    eroded = convolve_2D(mask, erode_offsets, operation="erode")
    dilated = convolve_2D(eroded, dilate_offsets, operation="dilate")

    return dilated


def erode_mask(mask, erode_size=15):
    _erode_kernel, erode_offsets, _erode_weights = create_kernel(
        (erode_size, erode_size),
        spherical=True,
        edge_weights=False,
        offsets=True,
        normalised=False,
        distance_calc=False,
        output_2d=True,
        remove_zero_weights=True,
    )

    eroded = convolve_2D(mask, erode_offsets, operation="erode")

    return eroded


@jit(nopython=True, parallel=True, nogil=True, fastmath=True, inline="always")
def scl_to_quality(scl, b2, b12, cld_prop, sun_adjust):
    quality = np.empty_like(scl, dtype="uint8")
    for x in prange(scl.shape[0]):  # pylint: disable=not-an-iterable
        for y in range(scl.shape[1]):
            pixel = scl[x][y]

            pixel_quality = 0

            if pixel == 0:  # SC_NODATA
                pixel_quality = 0
            elif pixel == 1:  # SC_SATURATED_DEFECTIVE
                pixel_quality = 0
            elif pixel == 2:  # SC_DARK_FEATURE_SHADOW
                pixel_quality = 35
            elif pixel == 3:  # SC_CLOUD_SHADOW
                pixel_quality = 25
            elif pixel == 4:  # SC_VEGETATION
                pixel_quality = 100
            elif pixel == 5:  # SC_NOT_VEGETATED
                pixel_quality = 100
            elif pixel == 6:  # SC_WATER
                pixel_quality = 95
            elif pixel == 7:  # SC_UNCLASSIFIED
                pixel_quality = 90
            elif pixel == 8:  # SC_CLOUD_MEDIUM_PROBA
                pixel_quality = 25
            elif pixel == 9:  # SC_CLOUD_HIGH_PROBA
                pixel_quality = 10
            elif pixel == 10:  # SC_THIN_CIRRUS
                pixel_quality = 75
            elif pixel == 11:  # SC_SNOW_ICE
                pixel_quality = 55
            else:
                pixel_quality = 1

            # Evaluate sun angle
            pixel_quality += sun_adjust

            # Evaluate blue band
            b2_val = b2[x][y]
            b2_adjust = 0
            if b2_val > 700:
                b2_adjust = -0.0175 * b2_val + 7
            elif b2_val < 100:
                b2_adjust = (-0.002 * (b2_val * b2_val)) + (0.4 * b2_val) - 20

            if b2_adjust > 10:
                b2_adjust = 10
            elif b2_adjust < -10:
                b2_adjust = -10

            pixel_quality += b2_adjust

            # Evaluate Clouds
            cldprop_val = cld_prop[x][y]
            cld = 0
            if (
                (pixel == 4)
                | (pixel == 5)
                | (pixel == 6)
                | (pixel == 7)
                | (pixel == 10)
                | (pixel == 11)
            ):
                if b2_val < 1000:
                    cld = cldprop_val + ((0.01 * b2_val) - 10)
                else:
                    cld = cldprop_val

            if cld > 10:
                cld = 10
            elif cld < -10:
                cld = -10

            pixel_quality -= cld

            quality_value = int(pixel_quality) if pixel_quality > 0 else 0

            quality[x][y] = quality_value

    return quality


def assess_quality(safe_folder):
    metadata = get_metadata(safe_folder)

    scene_classification = raster_to_array(
        metadata["paths"]["20m"]["SCL"], filled=True, output_2d=True
    )

    sun_elevation = metadata["SUN_ELEVATION"]
    sun_adjust = (
        (-0.0012 * (sun_elevation * sun_elevation)) + (0.2778 * sun_elevation) - 10
    )

    b2 = raster_to_array(metadata["paths"]["20m"]["B02"], filled=True, output_2d=True)
    b12 = raster_to_array(metadata["paths"]["20m"]["B12"], filled=True, output_2d=True)
    cld_prop = raster_to_array(
        metadata["paths"]["QI"]["CLDPRB_20m"], filled=True, output_2d=True
    )
    quality = scl_to_quality(scene_classification, b2, b12, cld_prop, sun_adjust)

    quality = smooth_quality(quality)

    return quality
