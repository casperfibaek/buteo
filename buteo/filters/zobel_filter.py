"""
    Implements a zobel filter on a 2D array.

    TODO:
    - Add documentation
    - Add to filters.py
"""

import numpy as np
from numba import jit, prange


def zobel_kernel(shape, norm=True, offsets=True, channel_last=True, output_2d=True):

    assert shape[0] == shape[1], "only works for square kernels"

    def sobel_col(size):

        count = (size // 2) + 1
        ret = []
        prev = 1
        for _ in range(count):
            new = prev * 2
            ret.append(new)
            prev = new

        ret = np.array(ret)
        ret = np.append(ret, np.flip(ret[0 : count - 1])) // 2

        return np.array([ret]).T

    def sobel_row(size):

        count = size // 2
        ret = []

        prev = 2 ** (count)

        for _ in range(count):
            new = prev / 2
            ret.append(new)
            prev = new

        ret = np.array(ret)
        ret = np.append(ret, 0)
        ret = np.append(ret, np.flip(ret[0:count]) * -1)

        return np.array([ret])

    col = sobel_col(shape[0])
    row = sobel_row(shape[0])

    kernel = np.matmul(col, row, dtype="float32")

    if norm:

        norm_calc_matrix = kernel[:, : (shape[0] // 2)]
        norm_calc_sum = norm_calc_matrix.sum()

        norm_calc_final_scalar = norm_calc_sum / 0.5
        kernel = kernel / norm_calc_final_scalar

    idx_offsets = []

    if offsets:

        if len(kernel.shape) == 2:

            offset_shape = [1, kernel.shape[0], kernel.shape[1]]
            offset_kernel = np.zeros(offset_shape, dtype="float32")

        for z in range(offset_kernel.shape[0]):
            for x in range(offset_kernel.shape[1]):
                for y in range(offset_kernel.shape[2]):

                    if channel_last:
                        if output_2d:
                            idx_offsets.append(
                                [
                                    x - (offset_kernel.shape[1] // 2),
                                    y - (offset_kernel.shape[2] // 2),
                                ]
                            )
                        else:
                            idx_offsets.append(
                                [
                                    x - (offset_kernel.shape[1] // 2),
                                    y - (offset_kernel.shape[2] // 2),
                                    z - (offset_kernel.shape[0] // 2),
                                ]
                            )
                    else:
                        if output_2d:
                            idx_offsets.append(
                                [
                                    x - (offset_kernel.shape[1] // 2),
                                    y - (offset_kernel.shape[2] // 2),
                                ]
                            )
                        else:
                            idx_offsets.append(
                                [
                                    z - (offset_kernel.shape[0] // 2),
                                    x - (offset_kernel.shape[1] // 2),
                                    y - (offset_kernel.shape[2] // 2),
                                ]
                            )
    if offsets:
        return (
            kernel,
            np.array(idx_offsets, dtype=int),
        )

    return kernel


@jit(nopython=True, parallel=True, nogil=True, fastmath=True)
def convolve_sobel_2D(arr, kernel, offsets):
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

            transformation = hood_values * kernel

            result[x, y] = np.sum(transformation)

    return result


def zobel_filter(arr, size=[3, 3], normalised=False):
    sobel_filter, offsets = zobel_kernel(size, norm=normalised, offsets=True)
    sobel_filter90 = np.rot90(sobel_filter, 3)

    sobel_flattened = sobel_filter.flatten()
    sobel_flattened90 = sobel_filter90.flatten()

    res1 = convolve_sobel_2D(arr.astype("float32"), sobel_flattened, offsets)
    res2 = convolve_sobel_2D(arr.astype("float32"), sobel_flattened90, offsets)

    filtered = (res1 ** 2 + res2 ** 2) ** 0.5

    return filtered
