import sys
import numpy as np
from numba import jit, prange

sys.path.append("../../")

from buteo.earth_observation.s2_utils import get_metadata
from buteo.raster.io import raster_to_array, array_to_raster
from buteo.filters.kernel_generator import create_kernel
from buteo.raster.resample import internal_resample_raster
from buteo.raster.align import align_rasters


@jit(nopython=True, parallel=True, nogil=True, fastmath=True, inline="always")
def hood_quantile(values, weights, quant):
    sort_mask = np.argsort(values)
    sorted_data = values[sort_mask]
    sorted_weights = weights[sort_mask]
    cumsum = np.cumsum(sorted_weights)
    intersect = (cumsum - 0.5 * sorted_weights) / cumsum[-1]
    return np.interp(quant, intersect, sorted_data)


@jit(nopython=True, parallel=True, nogil=True, fastmath=True, inline="always")
def hood_median_absolute_deviation(values, weights, median):
    absdeviation = np.abs(np.subtract(values, median))
    return hood_quantile(absdeviation, weights, 0.5)


@jit(nopython=True, parallel=True, nogil=True, fastmath=True, inline="always")
def mad_pan(values, weights, median):
    absdeviation = np.abs(np.subtract(values, median))
    return hood_quantile(absdeviation, weights, 0.5)


@jit(
    nopython=True,
    parallel=True,
    nogil=True,
    fastmath=True,
    error_model="numpy",
    cache=True,
)
def pansharpen_kernel(
    pan_band,
    target_band,
    offsets,
    weights,
):
    x_adj = pan_band.shape[0] - 1
    y_adj = pan_band.shape[1] - 1

    hood_size = len(offsets)
    result = np.empty_like(target_band)

    for x in prange(pan_band.shape[0]):
        for y in range(pan_band.shape[1]):

            hood_pan = np.zeros(hood_size, dtype="float32")
            hood_tar = np.zeros(hood_size, dtype="float32")

            hood_weights = np.zeros(hood_size, dtype="float32")
            weight_sum = np.array([0.0], dtype="float32")

            normalise = False
            for n in range(hood_size):
                offset_x = x + offsets[n][0]
                offset_y = y + offsets[n][1]

                outside = False

                if offset_x < 0:
                    offset_x = 0
                    outside = True
                elif offset_x > x_adj:
                    offset_x = x_adj
                    outside = True

                if offset_y < 0:
                    offset_y = 0
                    outside = True
                elif offset_y > y_adj:
                    offset_y = y_adj
                    outside = True

                hood_tar[n] = target_band[offset_x, offset_y]
                hood_pan[n] = pan_band[offset_x, offset_y]

                if outside == True:
                    normalise = True
                    hood_weights[n] = 0
                else:
                    hood_weights[n] = weights[n]
                    weight_sum[0] += weights[n]

            if normalise:
                hood_weights = np.divide(hood_weights, weight_sum[0])

            tar_median = hood_quantile(hood_tar, hood_weights, 0.5)
            tar_mad = hood_median_absolute_deviation(hood_tar, hood_weights, tar_median)

            pan_median = hood_quantile(hood_pan, hood_weights, 0.5)
            pan_mad = hood_median_absolute_deviation(hood_pan, hood_weights, pan_median)

            result[x][y] = (
                np.true_divide((target_band[x][y] - pan_median) * pan_mad, tar_mad)
                + tar_median
            )

    return result


def pansharpen(pan_path, tar_path, out_path):
    target = internal_resample_raster(tar_path, pan_path, resample_alg="bilinear")

    aligned = align_rasters(target, master=pan_path)
    target = aligned[0]

    tar_arr = raster_to_array(target, output_2d=True)
    tar_arr = (tar_arr - tar_arr.min()) / (tar_arr.max() - tar_arr.min())

    pan_arr = raster_to_array(pan_path, output_2d=True)

    _kernel, offsets, weights = create_kernel(
        [5, 5], sigma=2, output_2d=True, offsets=True
    )

    pan = pansharpen_kernel(pan_arr, tar_arr, offsets, weights.astype("float32"))
    array_to_raster(pan, reference=target, out_path=out_path)
