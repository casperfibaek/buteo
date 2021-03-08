import numpy as np
from numba import jit, prange

from kernel_generator import create_kernel
yellow_follow = 'C:/Users/caspe/Desktop/yellow/'
import sys; sys.path.append(yellow_follow)
from lib.raster_io import raster_to_array, array_to_raster


@jit(nopython=True, parallel=True, nogil=True)
def convolve_3d(arr, offsets, weights, operation="sum", border="valid"):
    z_adj = arr.shape[0] - 1
    x_adj = arr.shape[1] - 1
    y_adj = arr.shape[2] - 1

    hood_size = len(offsets)

    result = np.zeros(arr.shape, dtype="float32")
    border = True if border == "valid" else False

    for z in range(arr.shape[0]):
        for x in prange(arr.shape[1]):
            for y in range(arr.shape[2]):

                hood_values = np.zeros(hood_size, dtype="float32")
                hood_weights = np.zeros(hood_size, dtype="float32")
                weight_sum = np.array([0.0], dtype="float32")
                normalise = False

                for n in range(hood_size):
                    offset_z = z + offsets[n][0]
                    offset_x = x + offsets[n][1]
                    offset_y = y + offsets[n][2]

                    outside = False

                    if offset_z < 0:
                        offset_z = 0
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
                    result[z, x, y] = np.sum(np.multiply(hood_values, hood_weights))

                elif operation == "median":
                    sort_mask = np.argsort(hood_values)
                    sorted_data = hood_values[sort_mask]
                    sorted_weights = hood_weights[sort_mask]
                    cumsum = np.cumsum(sorted_weights)
                    intersect = (cumsum - 0.5 * sorted_weights) / cumsum[-1]

                    result[z, x, y] = np.interp(0.5, intersect, sorted_data)


    return result


def mean_filter(arr, shape):
    if len(arr.shape) == 2:
        arr = arr[np.newaxis, :, :]

    kernel, offsets, weights = create_kernel(
        shape,
        sigma=2,
        spherical=True,
        edge_weights=True,
        offsets=True,
        normalised=True,
        distance_calc="gaussian",
        radius_method="ellipsoid",
    )

    return convolve_3d(arr, offsets, weights, operation="median")[0]


if __name__ == "__main__":
    np.set_printoptions(suppress=True)

    folder = "C:/Users/caspe/Desktop/numba_conv/"
    raster = raster_to_array(folder + "fyn.tif")
    # raster = np.arange(0, 25).reshape((1, 5, 5))

    med = mean_filter(raster, (1, 5, 5))

    import pdb; pdb.set_trace()

    array_to_raster(med, reference_raster=folder + "fyn.tif", out_raster=folder + "fyn_smooth.tif")
