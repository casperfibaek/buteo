"""### Generic utility functions for arrays. ### """

# External
import numpy as np
from numba import jit, prange

def channel_first_to_last(arr: np.ndarray) -> np.ndarray:
    """
    Convert a channel-first array to channel-last format.
    
    Parameters
    ----------
    arr : np.ndarray
        The input array in channel-first format (channels, height, width) or
        (batch, channels, height, width) for 4D arrays
        
    Returns
    -------
    np.ndarray
        The output array in channel-last format (height, width, channels) or
        (batch, height, width, channels) for 4D arrays
    """
    if len(arr.shape) == 3:
        return np.transpose(arr, (1, 2, 0))
    elif len(arr.shape) == 4:
        return np.transpose(arr, (0, 2, 3, 1))
    else:
        raise ValueError(f"Input array must be 3D or 4D, got {len(arr.shape)}D")

def channel_last_to_first(arr: np.ndarray) -> np.ndarray:
    """
    Convert a channel-last array to channel-first format.

    Parameters
    ----------
    arr : np.ndarray
        The input array in channel-last format (height, width, channels) or
        (batch, height, width, channels) for 4D arrays

    Returns
    -------
    np.ndarray
        The output array in channel-first format (channels, height, width) or
        (batch, channels, height, width) for 4D arrays
    """
    if len(arr.shape) == 3:
        return np.transpose(arr, (2, 0, 1))
    elif len(arr.shape) == 4:
        return np.transpose(arr, (0, 3, 1, 2))
    else:
        raise ValueError(f"Input array must be 3D or 4D, got {len(arr.shape)}D")



@jit(nopython=True, nogil=True, parallel=True, fastmath=True, inline="always")
def _create_grid(range_rows: np.ndarray, range_cols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Create a grid of row and column indices based on input ranges.

    Parameters
    ----------
    range_rows : np.ndarray
        1D array containing the row indices to create grid from.
    range_cols : np.ndarray
        1D array containing the column indices to create grid from.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing two 2D arrays:
            - rows_grid: Array containing row indices arranged in a grid
            - cols_grid: Array containing column indices arranged in a grid
        Both arrays have shape (len(range_rows), len(range_cols)) and dtype np.int64.

    Examples
    --------
    >>> rows = np.array([0, 1])
    >>> cols = np.array([0, 1, 2])
    >>> rows_grid, cols_grid = _create_grid(rows, cols)
    >>> rows_grid
    array([[0, 0, 0],
           [1, 1, 1]])
    >>> cols_grid
    array([[0, 1, 2],
           [0, 1, 2]])
    """
    rows_grid = np.zeros((len(range_rows), len(range_cols)), dtype=np.int64)
    cols_grid = np.zeros((len(range_rows), len(range_cols)), dtype=np.int64)

    for i in prange(len(range_rows)):
        for j in range(len(range_cols)):
            cols_grid[i, j] = range_rows[j]
            rows_grid[i, j] = range_cols[i]

    return rows_grid, cols_grid
