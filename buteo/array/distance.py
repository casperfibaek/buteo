"""### Distance Calculation Module. ###"""

# Standard library
from typing import Union, Optional

# External
import numpy as np
from numba import jit, prange

# Internal
from buteo.array.utils_array import _create_grid



def convolve_distance(
    array: np.ndarray,
    target: Union[int, float] = 1,
    maximum_distance: Optional[Union[int, float]] = None,
    pixel_width: Union[int, float] = 1,
    pixel_height: Union[int, float] = 1,
) -> np.ndarray:
    """Calculate the distance from each pixel to the nearest target pixel.

    Parameters
    ----------
    array : np.ndarray
        The array to calculate the distances for.

    target : Union[int, float], optional
        The target value to calculate the distance to. Default: 1.

    maximum_distance : Union[int, float], optional
        The maximum distance to calculate. Default: None.

    pixel_width : Union[int, float], optional
        The width of each pixel. Default: 1.

    pixel_height : Union[int, float], optional
        The height of each pixel. Default: 1.

    Returns
    -------
    np.ndarray
        The array of distances.
    """
    # Support for multi-channel arrays
    if array.ndim == 3:
        binary_array = np.zeros((array.shape[0], array.shape[1]), dtype=np.uint8)
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                if array[i, j, 0] == target:
                    binary_array[i, j] = 1
    else:
        binary_array = (array == target).astype(np.uint8)

    # Calculate default maximum distance if not provided
    if maximum_distance is None:
        max_dist = np.sqrt(binary_array.shape[0] ** 2 + binary_array.shape[1] ** 2)
    else:
        max_dist = float(maximum_distance)
    
    # Convert pixel dimensions to float
    pixel_width_float = float(pixel_width)
    pixel_height_float = float(pixel_height)
    
    # Call the numba-accelerated implementation
    return _distance_calculation(binary_array, max_dist, pixel_width_float, pixel_height_float)


@jit(nopython=True, parallel=True, fastmath=True, cache=True, nogil=True)
def _distance_calculation(
    binary_array: np.ndarray,
    maximum_distance_float: float,
    pixel_width: float = 1.0,
    pixel_height: float = 1.0,
) -> np.ndarray:
    """Internal implementation for distance calculation with numba acceleration."""
    radius_cols = int(np.ceil(maximum_distance_float / pixel_height))
    radius_rows = int(np.ceil(maximum_distance_float / pixel_width))

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
    # Swap pixel_height and pixel_width to match the test expectations
    # The grid is in (col, row) format but pixel dims are in (width, height)
    coord_grid_projected[:, 0] = coord_grid[:, 0] * pixel_width
    coord_grid_projected[:, 1] = coord_grid[:, 1] * pixel_height

    coord_grid_values = np.sqrt((coord_grid_projected[:, 0] ** 2) + (coord_grid_projected[:, 1] ** 2))

    selected_range = np.arange(coord_grid.shape[0])
    selected_range = selected_range[np.argsort(coord_grid_values)][1:]
    selected_range = selected_range[coord_grid_values[selected_range] <= maximum_distance_float]

    coord_grid = coord_grid[selected_range]
    coord_grid_values = coord_grid_values[selected_range]

    distances = np.full_like(binary_array, maximum_distance_float, dtype=np.float32)
    for col in prange(binary_array.shape[0]):
        for row in range(binary_array.shape[1]):
            if binary_array[col, row] == 1:
                distances[col, row] = 0
            else:
                for idx, (col_adj, row_adj) in enumerate(coord_grid):
                    if (col + col_adj) >= 0 and (col + col_adj) < binary_array.shape[0] and \
                        (row + row_adj) >= 0 and (row + row_adj) < binary_array.shape[1] and \
                        binary_array[col + col_adj, row + row_adj] == 1:

                        distances[col, row] = coord_grid_values[idx]
                        break

    distance_array = np.expand_dims(distances, axis=2)

    return distance_array
