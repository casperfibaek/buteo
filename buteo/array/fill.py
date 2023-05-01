"""
### Generic utility functions ###

Functions that make interacting with the toolbox easier.
"""
# Standard library
from typing import Union, List, Optional

# Internal
from helpers import _create_grid

# External
import numpy as np
from numba import jit, prange



# TODO: Multichannel support, Split assert
@jit(nopython=True, parallel=True, fastmath=True, cache=True, nogil=True)
def fill_nodata_with_nearest_average(
    array: np.ndarray,
    nodata_value: Union[int, float],
    mask: Optional[np.ndarray] = None,
    max_iterations: Optional[Union[int, float]] = None,
    channel: int = 0,
):
    """
    Fill in nodata values with the average of the nearest values.
    
    Parameters
    ----------
    array : np.ndarray
        The array to fill in the nodata values for.

    nodata_value : Union[int, float]
        The value to use as the nodata value.

    mask : Optional[np.ndarray], optional
        The mask to use. Default: None.

    max_iterations : Optional[Union[int, float]], optional
        The maximum number of iterations to run. Default: None.

    channel : int, optional
        The channel to use. Default: 0.

    Returns
    -------
    np.ndarray
        The filled in array.
    """
    kernel_size = 3

    range_rows = np.arange(-(kernel_size // 2), (kernel_size // 2) + 1)
    range_cols = np.arange(-(kernel_size // 2), (kernel_size // 2) + 1)

    cols_grid, rows_grid = _create_grid(range_rows, range_cols)
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
