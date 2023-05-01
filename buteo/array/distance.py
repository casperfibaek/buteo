"""
### Generic utility functions ###

Functions that make interacting with the toolbox easier.
"""
# Internal
from helpers import _create_grid

# External
import numpy as np
from numba import jit, prange


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

    cols_grid, rows_grid = _create_grid(range_rows, range_cols)
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
