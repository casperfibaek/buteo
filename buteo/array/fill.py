"""### Fill nearest or nodata values. ###"""

# Standard library
from typing import Union, Optional

# External
import numpy as np
from numba import jit, prange

# Internal
from buteo.array.utils_array import _create_grid


# TODO: Mask is broken in class approach

# TODO: Multichannel support, Split assert
@jit(nopython=True, fastmath=True, cache=True, nogil=True)
def convolve_fill_nearest(
    array: np.ndarray,
    nodata_value: Union[int, float],
    mask: Optional[np.ndarray] = None,
    max_iterations: Optional[Union[int, float]] = None,
    channel: int = 0,
):
    """Fill in nodata values with the average of the nearest values.

    Parameters
    ----------
    array : np.ndarray
        The array to fill in the nodata values for. Expected to be channel-first.
    nodata_value : Union[int, float]
        The value to use as the nodata value.
    mask : Optional[np.ndarray], optional
        The mask to use. Default: None. (should be uint8)
    max_iterations : Optional[Union[int, float]], optional
        The maximum number of iterations to run. Default: None.
    channel : int, optional
        The channel to use. Default: 0.

    Returns
    -------
    np.ndarray
        The filled in array in channel-first order.
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

    weights = 1 / (coord_grid_values ** 2)
    weights = weights / np.sum(weights)
    weights = weights.astype(np.float32)

    main_filled = np.copy(array)

    if mask is None:
        mask = np.ones_like(main_filled, dtype=np.uint8)

    # Since the array is channel-first, extract the selected channel accordingly.
    main_filled = main_filled[channel, :, :]
    mask = mask[channel, :, :]

    nodata_value = np.array(nodata_value, dtype=array.dtype).item()
    uint8_1 = np.array(1, dtype=np.uint8)

    iterations = 0
    while True:
        local_filled = np.copy(main_filled)
        for row in prange(main_filled.shape[0]):
            for col in range(main_filled.shape[1]):
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

    return np.expand_dims(main_filled, axis=0)


@jit(nopython=True, fastmath=True, cache=True, nogil=True)
def convolve_fill_nearest_classes(
    array: np.ndarray,
    nodata_value: Union[int, float],
    mask: np.ndarray,
    max_iterations: Optional[Union[int, float]] = None,
    channel: int = 0,
):
    """Fill in nodata values with the most frequent class of the nearest values (channel-first).

    Parameters
    ----------
    array : np.ndarray
        The array to fill in the nodata values for.
    nodata_value : Union[int, float]
        The value to use as the nodata value.
    mask : np.ndarray
        The mask to use.
    max_iterations : Optional[Union[int, float]], optional
        The maximum number of iterations to run. Default: None.
    channel : int, optional
        The channel to use. Default: 0.

    Returns
    -------
    np.ndarray
        The filled in array in channel-first order.
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

    weights = 1 / (coord_grid_values ** 2)
    weights = weights / np.sum(weights)
    weights = weights.astype(np.float32)

    # TODO: Could these be filled wtih the create kernel functions instead.

    main_filled = np.copy(array)
    # Extract unique classes from the selected channel (channel-first).
    channel_data = main_filled[channel, :, :]
    classes = np.unique(channel_data)
    if nodata_value in classes:
        classes = classes[classes != nodata_value]

    idx_to_class = dict(enumerate(classes))
    class_to_idx = {v: k for k, v in idx_to_class.items()}

    # Use the selected channel for main_filled and mask.
    main_filled = main_filled[channel, :, :]
    mask = mask[channel, :, :]

    nodata_value = np.array(nodata_value, dtype=array.dtype).item()
    uint8_1 = np.array(1, dtype=np.uint8)

    iterations = 0
    while True:
        local_filled = np.copy(main_filled)
        for row in prange(main_filled.shape[0]):
            for col in range(main_filled.shape[1]):
                if main_filled[row, col] != nodata_value:
                    continue
                if mask[row, col] != uint8_1:
                    continue

                count = np.zeros(classes.size, dtype=np.float32)

                for i, (col_adj, row_adj) in enumerate(coord_grid):
                    new_row = row + row_adj
                    new_col = col + col_adj
                    if (new_row >= 0 and new_row < main_filled.shape[0] and
                        new_col >= 0 and new_col < main_filled.shape[1] and
                        main_filled[new_row, new_col] != nodata_value and
                        mask[new_row, new_col] == uint8_1):
                        class_ = main_filled[new_row, new_col]
                        class_idx = class_to_idx[class_]
                        count[class_idx] += weights[i]

                if count.sum() == 0:
                    local_filled[row, col] = nodata_value
                else:
                    local_filled[row, col] = idx_to_class[int(np.argmax(count))]

        main_filled = local_filled
        iterations += 1

        if max_iterations is not None and iterations >= max_iterations:
            break

        if np.sum((main_filled == nodata_value) & (mask == uint8_1)) == 0:
            break

    return np.expand_dims(main_filled, axis=0)
