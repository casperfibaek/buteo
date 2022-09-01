""" Create patches from rasters, used for machine learnign applications. """

import numpy as np
from numba import prange, jit


@jit(nopython=True, parallel=True, nogil=True, inline="always")
def weighted_median(arr, weight_arr):
    """ Calculate the weighted median of a multi-dimensional array along the last axis. """

    ret_arr = np.empty((arr.shape[0], arr.shape[1], 1), dtype="float32")

    for idx_x in prange(arr.shape[0]):
        for idx_y in range(arr.shape[1]):
            values = arr[idx_x, idx_y].flatten()
            weights = weight_arr[idx_x, idx_y].flatten()

            sort_mask = np.argsort(values)
            sorted_data = values[sort_mask]
            sorted_weights = weights[sort_mask]
            cumsum = np.cumsum(sorted_weights)
            intersect = (cumsum - 0.5 * sorted_weights) / cumsum[-1]
            ret_arr[idx_x, idx_y, 0] = np.interp(0.5, intersect, sorted_data)

    return ret_arr


@jit(nopython=True, parallel=True, nogil=True, inline="always")
def mad_merge(arr, weight_arr, mad_dist=1.0):
    """ Merge an array of predictions using the MAD-merge methodology. """

    ret_arr = np.empty((arr.shape[0], arr.shape[1], 1), dtype="float32")

    for idx_x in prange(arr.shape[0]):
        for idx_y in prange(arr.shape[1]):
            values = arr[idx_x, idx_y].flatten()
            weights = weight_arr[idx_x, idx_y].flatten()

            sort_mask = np.argsort(values)
            sorted_data = values[sort_mask]
            sorted_weights = weights[sort_mask]
            cumsum = np.cumsum(sorted_weights)
            intersect = (cumsum - 0.5 * sorted_weights) / cumsum[-1]

            median = np.interp(0.5, intersect, sorted_data)
            mad = np.median(np.abs(median - values))

            if mad == 0.0:
                ret_arr[idx_x, idx_y, 0] = median
                continue

            new_weights = np.zeros_like(sorted_weights)
            for idx_z in range(sorted_data.shape[0]):
                new_weights[idx_z] = 1.0 - (np.minimum(np.abs(sorted_data[idx_z] - median) / (mad * mad_dist), 1))

            cumsum = np.cumsum(new_weights)
            intersect = (cumsum - 0.5 * new_weights) / cumsum[-1]

            ret_arr[idx_x, idx_y, 0] = np.interp(0.5, intersect, sorted_data)

    return ret_arr


def get_offsets(size, number_of_offsets=3):
    """ Get offsets for a given array size. """
    assert number_of_offsets <= 9, "Number of offsets must be nine or less"

    offsets = [[0, 0]]

    if number_of_offsets == 0:
        return offsets

    mid = size // 2
    low = mid // 2
    high = mid + low

    additional_offsets = [
        [mid, mid],
        [0, mid],
        [mid, 0],
        [0, low],
        [low, 0],
        [high, 0],
        [0, high],
        [low, low],
        [high, high],
    ]

    offsets += additional_offsets[:number_of_offsets]

    return offsets


def array_to_patches(arr, tile_size, offset=None):
    """ Generate patches from an array. """

    if offset is None:
        offset = [0, 0]

    patches_y = (arr.shape[0] - offset[1]) // tile_size
    patches_x = (arr.shape[1] - offset[0]) // tile_size

    cut_y = -((arr.shape[0] - offset[1]) % tile_size)
    cut_x = -((arr.shape[1] - offset[0]) % tile_size)

    cut_y = None if cut_y == 0 else cut_y
    cut_x = None if cut_x == 0 else cut_x

    og_coords = [offset[1], cut_y, offset[0], cut_x]
    og_shape = list(arr[offset[1] : cut_y, offset[0] : cut_x].shape)

    reshaped = arr[offset[1] : cut_y, offset[0] : cut_x].reshape(
        patches_y,
        tile_size,
        patches_x,
        tile_size,
        arr.shape[2],
    )

    swaped = reshaped.swapaxes(1, 2)
    blocks = swaped.reshape(-1, tile_size, tile_size, arr.shape[2])

    return blocks, og_coords, og_shape


def patches_to_array(patches, og_shape, tile_size, offset=None):
    """ Reconstitute an array from patches. """

    if offset is None:
        offset = [0, 0]

    with np.errstate(invalid="ignore"):
        target = np.empty(og_shape, dtype="float32") * np.nan

    target_y = ((og_shape[0] - offset[1]) // tile_size) * tile_size
    target_x = ((og_shape[1] - offset[0]) // tile_size) * tile_size

    cut_y = -((og_shape[0] - offset[1]) % tile_size)
    cut_x = -((og_shape[1] - offset[0]) % tile_size)

    cut_x = None if cut_x == 0 else cut_x
    cut_y = None if cut_y == 0 else cut_y

    reshape = patches.reshape(
        target_y // tile_size,
        target_x // tile_size,
        tile_size,
        tile_size,
        patches.shape[3],
        1,
    )

    swap = reshape.swapaxes(1, 2)

    destination = swap.reshape(
        (target_y // tile_size) * tile_size,
        (target_x // tile_size) * tile_size,
        patches.shape[3],
    )

    target[offset[1] : cut_y, offset[0] : cut_x] = destination

    return target


def get_kernel_weights(tile_size=64, edge_distance=5, epsilon=1e-7):
    """ Weight a kernel according to how close to an edge a given pixel is. """

    arr = np.empty((tile_size, tile_size), dtype="float32")
    max_dist = edge_distance * 2
    for idx_y in range(0, arr.shape[0]):
        for idx_x in range(0, arr.shape[1]):
            val_y_top = max(edge_distance - idx_y, 0.0)
            val_y_bot = max((1 + edge_distance) - (tile_size - idx_y), 0.0)
            val_y = val_y_top + val_y_bot

            val_x_lef = max(edge_distance - idx_x, 0.0)
            val_x_rig = max((1 + edge_distance) - (tile_size - idx_x), 0.0)
            val_x = val_x_lef + val_x_rig

            val = (max_dist - abs(val_y + val_x)) / max_dist

            if val <= 0.0:
                val = epsilon

            arr[idx_y, idx_x] = val

    return arr


def get_patches(arr, tile_size, number_of_offsets=3, border_check=True):
    """ Generate patches from an array. Also outputs the offsets and the shapes of the offsets. """

    overlaps = []
    offsets = []
    shapes = []

    calc_borders = get_offsets(tile_size, number_of_offsets=number_of_offsets)
    if border_check:
        calc_borders.append([arr.shape[1] - tile_size, 0])
        calc_borders.append([0, arr.shape[0] - tile_size])
        calc_borders.append([arr.shape[1] - tile_size, arr.shape[0] - tile_size])

    for offset in calc_borders:
        blocks, og_coords, og_shape = array_to_patches(arr, tile_size, offset)

        shapes.append(og_shape)
        offsets.append(og_coords)
        overlaps.append(blocks)

    return overlaps, offsets, shapes
