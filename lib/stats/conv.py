import numpy as np
from numba import jit, prange
from kernel_generator import create_kernel


@jit(nopython=True, parallel=True, nogil=True, fastmath=True)
def convolve_3d(arr, offsets, weights, operation="sum", border="valid", quantile=0.5):
    z_adj = (arr.shape[0] - 1) // 2
    x_adj = arr.shape[1] - 1
    y_adj = arr.shape[2] - 1

    hood_size = len(offsets)

    result = np.zeros(arr.shape[1:], dtype="float32")
    border = True if border == "valid" else False

    for x in prange(arr.shape[1]):
        for y in range(arr.shape[2]):

            hood_values = np.zeros(hood_size, dtype="float32")
            hood_weights = np.zeros(hood_size, dtype="float32")
            weight_sum = np.array([0.0], dtype="float32")
            normalise = False

            # for z in range(arr.shape[0]):
            for n in range(hood_size):
                offset_z = offsets[n][0]
                offset_x = x + offsets[n][1]
                offset_y = y + offsets[n][2]

                outside = False

                if offset_z < -z_adj:
                    offset_z = -z_adj
                    outside = True
                elif offset_z > z_adj:
                    offset_z = z_adj
                    outside = True

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
                
                hood_values[n] = arr[offset_z, offset_x, offset_y]

                if border == True and outside == True:
                    normalise = True
                    hood_weights[n] = 0
                else:
                    hood_weights[n] = weights[n]
                    weight_sum[0] += weights[n]

            if normalise:
                hood_weights = np.divide(hood_weights, weight_sum[0])

            if operation == "sum":
                result[x, y] = np.sum(np.multiply(hood_values, hood_weights))

            elif operation == "quantile" or operation == "median_absolute_deviation" or operation == "median":
                quantile_to_find = quantile if operation == "quantile" else 0.5

                sort_mask = np.argsort(hood_values)
                sorted_data = hood_values[sort_mask]
                sorted_weights = hood_weights[sort_mask]
                cumsum = np.cumsum(sorted_weights)
                intersect = (cumsum - 0.5 * sorted_weights) / cumsum[-1]
                quant = np.interp(quantile_to_find, intersect, sorted_data)

                if operation == "quantile" or operation == "median":
                    result[x, y] = quant
                else:
                    absolute_deviations = np.abs(np.subtract(hood_values, quant))

                    sort_mask = np.argsort(absolute_deviations)
                    sorted_data = absolute_deviations[sort_mask]
                    sorted_weights = hood_weights[sort_mask]
                    cumsum = np.cumsum(sorted_weights)
                    intersect = (cumsum - 0.5 * sorted_weights) / cumsum[-1]

                    mad = np.interp(quantile_to_find, intersect, sorted_data)

                    result[x, y] = mad

            elif operation == "standard_deviation":
                summed = np.sum(np.multiply(hood_values, hood_weights))
                variance = np.sum(np.multiply(np.power(np.subtract(hood_values, summed), 2), hood_weights))
                result[x, y] = np.sqrt(variance)

    return result


def mean_filter(arr, shape):
    if len(arr.shape) == 2:
        arr = arr[np.newaxis, :, :]

    kernel, offsets, weights = create_kernel(
        shape,
        sigma=1,
        spherical=True,
        edge_weights=True,
        offsets=True,
        normalised=True,
        distance_calc="gaussian",
        radius_method="ellipsoid",
    )

    return convolve_3d(arr, offsets, weights, operation="median")


if __name__ == "__main__":
    yellow_follow = 'C:/Users/caspe/Desktop/yellow/'
    np.set_printoptions(suppress=True)
    import sys; sys.path.append(yellow_follow)
    from lib.raster_io import raster_to_array, array_to_raster
    from lib.raster_align import align
    from glob import glob

    folder = "C:/Users/caspe/Desktop/numba_conv/"

    images = glob(folder + "*_vv*.tif")
    # images.reverse()

    arr = raster_to_array(images)

    # win = np.load(folder + "min.npy")
    # arr1 = np.ones((10, 10)) * 1
    # arr2 = np.ones((10, 10)) * 0
    # arr3 = np.ones((10, 10)) * 2

    # arr = np.stack([arr1, arr2, arr3])

    med = mean_filter(arr, (3, 5, 5))

    array_to_raster(med, folder + "3d_filter_vv_5x5.tif", images[0])

    # import pdb; pdb.set_trace()

    # raster = raster_to_array(folder + "spec_vh.tif")
    # # raster = np.arange(0, 25).reshape((1, 5, 5))

    # med = mean_filter(raster, (1, 7, 7))

    # array_to_raster(med, reference_raster=folder + "spec_vh.tif", out_raster=folder + "spec_vh_med5_sig1.tif")
