"""
### Generic utility functions ###

Functions that make interacting with the toolbox easier.
"""

# External
import numpy as np
from numba import jit


def channel_first_to_last(arr: np.ndarray) -> np.ndarray:
    """
    Converts a numpy array from channel first to channel last format.
    `(-batch-, channel, height, width)` -> `(-batch-, height, width, channel)`
    
    If 4D, it is assumed that the input array is in batch, channel, height, width format.
    If 3D, it is assumed that the input array is in channel, height, width format.

    Parameters
    ----------
    arr : np.ndarray
        The array to convert.

    Returns
    -------
    np.ndarray
        The converted array.
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

    `(-batch-, height, width, channel)` -> `(-batch-, channel, height, width)`
    
    If 4D, it is assumed that the input array is in batch, channel, height, width format.
    If 3D, it is assumed that the input array is in channel, height, width format.

    Parameters
    ----------
    arr : np.ndarray
        The array to convert.

    Returns
    -------
    np.ndarray
        The converted array.
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
def _create_grid(range_rows, range_cols):
    """
    Create a grid of rows and columns.
    
    Parameters
    ----------
    range_rows : np.ndarray
        The rows to create the grid from.

    range_cols : np.ndarray
        The columns to create the grid from.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The rows and columns grid.
    """
    rows_grid = np.zeros((len(range_rows), len(range_cols)), dtype=np.int64)
    cols_grid = np.zeros((len(range_rows), len(range_cols)), dtype=np.int64)

    for i in range(len(range_rows)):
        for j in range(len(range_cols)):
            cols_grid[i, j] = range_rows[j]
            rows_grid[i, j] = range_cols[i]

    return rows_grid, cols_grid
