""" ### Utility functions for working with patches. ###"""

# Standard library
from typing import List, Tuple, Optional, Union

# External
import numpy as np
from numba import prange, jit


def _get_kernel_weights(
    tile_size: int = 64,
    edge_distance: int = 5,
    epsilon: float = 1e-7,
) -> np.ndarray:
    """Weight a kernel according to how close to an edge a given pixel is.

    Parameters
    ----------
    tile_size : int, optional
        The size of the square kernel. Default: 64

    edge_distance : int, optional
        The distance from the edge to consider for weighting. Default: 5

    epsilon : float, optional
        A small value to prevent division by zero. Default: 1e-7

    Returns
    -------
    np.ndarray
        A 2D NumPy array of shape (tile_size, tile_size) with the kernel weights.
    """
    assert tile_size > 0, "Tile size must be greater than zero."
    assert edge_distance < tile_size // 2, "Edge distance must be less than half the tile size."
    assert edge_distance >= 0, "Edge distance must be greater than or equal to zero."

    arr = np.zeros((tile_size, tile_size), dtype="float32")
    max_dist = edge_distance * 2

    # Iterate through the kernel array
    for idx_y in range(0, arr.shape[0]):
        for idx_x in range(0, arr.shape[1]):

            # Calculate vertical distance to the closest edge
            val_y_top = max(edge_distance - idx_y, 0.0)
            val_y_bot = max((1 + edge_distance) - (tile_size - idx_y), 0.0)
            val_y = val_y_top + val_y_bot

            # Calculate horizontal distance to the closest edge
            val_x_lef = max(edge_distance - idx_x, 0.0)
            val_x_rig = max((1 + edge_distance) - (tile_size - idx_x), 0.0)
            val_x = val_x_lef + val_x_rig

            # Calculate the weight based on the distance to the closest edge
            val = (max_dist - abs(val_y + val_x)) / max_dist

            # Set a minimum weight to avoid division by zero
            if val <= 0.0:
                val = epsilon

            # Assign the calculated weight to the kernel array
            arr[idx_y, idx_x] = val

    return arr


def _get_offsets(
    tile_size: int,
    n_offsets: int,
) -> List[Tuple[int, int]]:
    """Generate a list of offset pairs for a given tile size and number of offsets in y and x dimensions.

    Parameters
    ----------
    tile_size : int
        The size of each tile.

    n_offsets : int
        The desired number of offsets to be calculated in the y and x dimensions.

    Returns
    -------
    List[Tuple[int, int]]
        A list of tuples containing offset pairs for y and x dimensions.
        order is (y, x)
    """
    offset_props = np.arange(0, 1, 1 / (n_offsets + 1))[1:].tolist()
    offsets = [(0, 0)]

    assert tile_size > n_offsets, f"Too many offsets ({n_offsets}) requested for tile_size {tile_size}"

    for val in offset_props:
        offset = int(round((val * tile_size), 2))
        offsets.append((offset, offset))

    return offsets


def _borders_are_necessary(
    arr: np.ndarray,
    tile_size: int,
    offset: Union[List[int], Tuple[int, int]],
) -> Tuple[bool, bool]:
    """Checks if borders are necessary for the given array.
    Width and height are returned as a tuple.
    order is (y, x).

    Parameters
    ----------
    arr : np.ndarray
        The array to be checked.

    tile_size : int
        The size of each tile.

    offset : List[int]
        The offset to be used.

    Returns
    -------
    Tuple[bool, bool]
        A tuple containing of borders are needed in (height, width) dims.
    """
    if arr.ndim == 2:
        height, width = arr.shape
    else:
        height, width, _ = arr.shape

    if (height - offset[0]) % tile_size == 0:
        height_border = False
    else:
        height_border = True

    if (width - offset[1]) % tile_size == 0:
        width_border = False
    else:
        width_border = True

    return height_border, width_border


def _borders_are_necessary_list(
    arr: np.ndarray,
    tile_size: int,
    offsets: List[Tuple[int, int]],
) -> Tuple[bool, bool]:
    """Checks if borders are necessary for the given array.
    Width and height are returned as a tuple.

    Parameters
    ----------
    arr : np.ndarray
        The array to be checked.

    tile_size : int
        The size of each tile.

    offsets : List[List[int]]
        The offsets to be used.

    Returns
    -------
    Tuple[bool, bool]
        A tuple containing of borders are needed in (height, width) dims.
    """
    height_border = True
    width_border = True

    for offset in offsets:
        offset_height_border, offset_width_border = _borders_are_necessary(
            arr, tile_size, offset
        )
        if not offset_height_border:
            height_border = False

        if not offset_width_border:
            width_border = False

        if not height_border and not width_border:
            break

    return height_border, width_border


def _patches_to_weights(
    patches: np.ndarray,
    edge_distance: int,
) -> np.ndarray:
    """Calculate the weights for each patch based on the distance to the edge.
    
    Parameters
    ----------
    patches : np.ndarray
        The patches to be weighted.
        
    edge_distance : int
        The distance from the edge to consider for weighting.
        
    Returns
    -------
    np.ndarray
        The weights for each patch.
    """
    assert len(patches.shape) == 4, "Patches must be a 4D array"
    assert patches.shape[1] == patches.shape[2], "Patches must be square"

    # Calculate the distance to the edge for each patch
    weights = _get_kernel_weights(patches.shape[1], edge_distance)

    # Expand the weights to match the number of patches
    weights = np.repeat(weights[np.newaxis, ...], patches.shape[0], axis=0)[..., np.newaxis]

    return weights


@jit(nopython=True, nogil=True)
def _unique_values(arr: np.ndarray) -> np.ndarray:
    """Find the unique values in a 1D NumPy array.

    Parameters
    ----------
    arr : np.ndarray
        The input array.

    Returns
    -------
    np.ndarray
        A 1D NumPy array with the unique values.
    """
    unique = np.empty(arr.size, dtype=arr.dtype)
    unique_count = 0
    for i in range(arr.shape[0]):
        if arr[i] not in unique[:unique_count]:
            unique[unique_count] = arr[i]
            unique_count += 1

    return unique[:unique_count]
