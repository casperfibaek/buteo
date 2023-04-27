"""
### Generic utility functions ###

Functions that make interacting with the toolbox easier.
"""
# Internal
from typing import Tuple

# External
import numpy as np
from numba import jit, prange



def channel_first_to_last(arr: np.ndarray) -> np.ndarray:
    """
    Converts a numpy array from channel first to channel last format.
    (-batch-, channel, height, width) -> (-batch-, height, width, channel)
    
    If 4D, it is assumed that the input array is in batch, channel, height, width format.
    If 3D, it is assumed that the input array is in channel, height, width format.

    Args:
        arr (np.ndarray): The array to convert.

    Returns:
        (np.ndarray): The converted array.
    """
    if arr.ndim not in [3, 4]:
        raise ValueError("Input array should be 3 or 4-dimensional with shape (-batch-, channels, height, width)")

    # Swap the axes to change from channel first to channel last format
    if arr.ndim == 3:
        arr = np.transpose(arr, (1, 2, 0))
    else:
        arr = np.transpose(arr, (0, 2, 3, 1))

    return arr

def channel_last_to_first(arr: np.ndarray) -> np.ndarray:
    """
    Converts a numpy array from channel last to channel first format.

    (-batch-, height, width, channel) -> (-batch-, channel, height, width)
    
    If 4D, it is assumed that the input array is in batch, channel, height, width format.
    If 3D, it is assumed that the input array is in channel, height, width format.

    Args:
        arr(np.ndarray): The array to convert.

    Returns:
        (np.ndarray): The converted array.
    """
    if arr.ndim not in [3, 4]:
        raise ValueError("Input array should be 3 or 4-dimensional with shape (-batch-, height, width, channels)")

    # Swap the axes to change from channel last to channel first format
    if arr.ndim == 3:
        arr = np.transpose(arr, (2, 0, 1))
    else:
        arr = np.transpose(arr, (0, 3, 1, 2))

    return arr


@jit(nopython=True)
def create_grid(range_rows, range_cols):
    """ Create a grid of rows and columns """
    rows_grid = np.zeros((len(range_rows), len(range_cols)), dtype=np.int64)
    cols_grid = np.zeros((len(range_rows), len(range_cols)), dtype=np.int64)

    for i in range(len(range_rows)):
        for j in range(len(range_cols)):
            cols_grid[i, j] = range_rows[j]
            rows_grid[i, j] = range_cols[i]

    return rows_grid, cols_grid


@jit(nopython=True, parallel=True, fastmath=True, cache=True, nogil=True)
def calculate_pixel_distances(array, target=1, maximum_distance=None, pixel_width=1, pixel_height=1):
    """ Calculate the distance from each pixel to the nearest target pixel. """
    binary_array = np.sum(array == target, axis=2, dtype=np.uint8)

    if maximum_distance is None:
        maximum_distance = np.sqrt(binary_array.shape[0] ** 2 + binary_array.shape[1] ** 2)

    radius_cols = int(np.ceil(maximum_distance / pixel_height))
    radius_rows = int(np.ceil(maximum_distance / pixel_width))

    kernel_cols = radius_cols * 2
    kernel_rows = radius_rows * 2

    if kernel_cols % 2 == 0:
        kernel_cols += 1

    if kernel_rows % 2 == 0:
        kernel_rows += 1

    middle_cols = int(np.floor(kernel_cols / 2))
    middle_rows = int(np.floor(kernel_rows / 2))

    range_cols = np.arange(-middle_cols, middle_cols + 1)
    range_rows = np.arange(-middle_rows, middle_rows + 1)

    cols_grid, rows_grid = create_grid(range_rows, range_cols)
    coord_grid = np.empty((cols_grid.size, 2), dtype=np.int64)
    coord_grid[:, 0] = cols_grid.flatten()
    coord_grid[:, 1] = rows_grid.flatten()

    coord_grid_projected = np.empty_like(coord_grid, dtype=np.float32)
    coord_grid_projected[:, 0] = coord_grid[:, 0] * pixel_height
    coord_grid_projected[:, 1] = coord_grid[:, 1] * pixel_width

    coord_grid_values = np.sqrt((coord_grid_projected[:, 0] ** 2) + (coord_grid_projected[:, 1] ** 2))

    selected_range = np.arange(coord_grid.shape[0])
    selected_range = selected_range[np.argsort(coord_grid_values)][1:]
    selected_range = selected_range[coord_grid_values[selected_range] <= maximum_distance]

    coord_grid = coord_grid[selected_range]
    coord_grid_values = coord_grid_values[selected_range]

    distances = np.full_like(binary_array, maximum_distance, dtype=np.float32)
    for col in prange(binary_array.shape[0]):
        for row in range(binary_array.shape[1]):
            if binary_array[col, row] == target:
                distances[col, row] = 0
            else:
                for idx, (col_adj, row_adj) in enumerate(coord_grid):
                    if (col + col_adj) >= 0 and (col + col_adj) < binary_array.shape[0] and \
                        (row + row_adj) >= 0 and (row + row_adj) < binary_array.shape[1] and \
                        binary_array[col + col_adj, row + row_adj] == target:

                        distances[col, row] = coord_grid_values[idx]
                        break

    return np.expand_dims(distances, axis=2)


@jit(nopython=True, parallel=True, fastmath=True, cache=True, nogil=True)
def fill_nodata_with_nearest_average(array, nodata_value, mask=None, max_iterations=None, channel=0):
    """ Calculate the distance from each pixel to the nearest target pixel. """
    kernel_size = 3

    range_rows = np.arange(-(kernel_size // 2), (kernel_size // 2) + 1)
    range_cols = np.arange(-(kernel_size // 2), (kernel_size // 2) + 1)

    cols_grid, rows_grid = create_grid(range_rows, range_cols)
    coord_grid = np.empty((cols_grid.size, 2), dtype=np.int64)
    coord_grid[:, 0] = cols_grid.flatten()
    coord_grid[:, 1] = rows_grid.flatten()

    coord_grid_values = np.sqrt((coord_grid[:, 0] ** 2) + (coord_grid[:, 1] ** 2))
    coord_grid_values_sort = np.argsort(coord_grid_values)[1:]
    coord_grid_values = coord_grid_values[coord_grid_values_sort]

    coord_grid = coord_grid[coord_grid_values_sort]

    weights = 1 / coord_grid_values
    weights = weights / np.sum(weights)
    weights = weights.astype(np.float32)

    main_filled = np.copy(array)

    if mask is None:
        mask = np.ones_like(main_filled, dtype=np.uint8)
    else:
        mask = (mask == 1).astype(np.uint8)

    main_filled = main_filled[:, :, channel]
    mask = mask[:, :, channel]

    nodata_value = np.array(nodata_value, dtype=array.dtype)
    uint8_1 = np.array(1, dtype=np.uint8)

    iterations = 0
    while True:
        local_filled = np.copy(main_filled)
        for row in prange(main_filled.shape[0]):
            for col in prange(main_filled.shape[1]):
                if main_filled[row, col] != nodata_value:
                    continue
                if mask[row, col] != uint8_1:
                    continue

                count = 0
                weights_sum = 0.0
                value_sum = 0.0

                for idx, (col_adj, row_adj) in enumerate(coord_grid):
                    if (row + row_adj) >= 0 and (row + row_adj) < main_filled.shape[0] and \
                        (col + col_adj) >= 0 and (col + col_adj) < main_filled.shape[1] and \
                        main_filled[row + row_adj, col + col_adj] != nodata_value and \
                        mask[row + row_adj, col + col_adj] == uint8_1:

                        weight = weights[idx]
                        value = main_filled[row + row_adj, col + col_adj]

                        value_sum += value * weight
                        weights_sum += weight
                        count += 1

                if count == 0:
                    local_filled[row, col] = nodata_value
                else:
                    local_filled[row, col] = value_sum * (1.0 / weights_sum)

        main_filled = local_filled
        iterations += 1

        if max_iterations is not None and iterations >= max_iterations:
            break

        if np.sum((main_filled == nodata_value) & (mask == uint8_1)) == 0:
            break

    return np.expand_dims(main_filled, axis=2)
