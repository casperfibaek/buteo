""" ### Functions for extracting patches from arrays. ###"""

# Standard library
from typing import Union, List, Tuple, Optional

# External
import numpy as np

# Internal
from buteo.array.utils_array import channel_first_to_last, channel_last_to_first
from buteo.array.patches.util import (
    _get_offsets,
    _borders_are_necessary_list,
)


def _array_to_patches_single(
    arr: np.ndarray,
    tile_size: int,
    offset: Optional[Union[List[int], Tuple[int, int]]] = None,
) -> np.ndarray:
    """Generate patches from an array. Offsets in (y, x) order.

    Parameters
    ----------
    arr : np.ndarray
        The array to be divided into patches.

    tile_size : int
        The size of each tile/patch, e.g., 64 for 64x64 tiles.

    offset : Optional[Union[List[int], Tuple[int, int]]], optional
        The y and x offset values for the input array. If not provided, defaults to [0, 0].

    Returns
    -------
    np.ndarray
        A numpy array containing the patches.
    """
    assert arr.ndim in [2, 3], "Array must be 2D or 3D"
    assert tile_size > 0, "Tile size must be greater than 0"
    assert offset is None or len(offset) == 2, "Offset must be a list or tuple of length 2"

    # Set default offset to [0, 0] if not provided
    if offset is None:
        offset = [0, 0]

    # Calculate the number of patches in the y and x dimensions
    patches_y = (arr.shape[0] - offset[0]) // tile_size
    patches_x = (arr.shape[1] - offset[1]) // tile_size

    # Calculate cut dimensions for the y and x dimensions
    cut_y = -((arr.shape[0] - offset[0]) % tile_size)
    cut_x = -((arr.shape[1] - offset[1]) % tile_size)

    # Set cut dimensions to None if they are 0
    cut_y = None if cut_y == 0 else cut_y
    cut_x = None if cut_x == 0 else cut_x

    # Reshape the array to separate the patches
    reshaped = arr[offset[0]:cut_y, offset[1]:cut_x].reshape(
        patches_y,
        tile_size,
        patches_x,
        tile_size,
        arr.shape[2],
    )

    # Swap axes to rearrange patches in the correct order
    swaped = reshaped.swapaxes(1, 2)

    # Combine the patches into a single array
    blocks = swaped.reshape(-1, tile_size, tile_size, arr.shape[2])

    return blocks


def _patches_to_array_single(
    patches: np.ndarray,
    shape: Union[List, Tuple],
    tile_size: int,
    offset: Optional[Union[List, Tuple]] = None,
    background_value: Optional[Union[int, float]] = None,
) -> np.ndarray:
    """Reconstitute an array from patches.

    Given an array of patches, this function stitches them back together
    to form the original array of the specified shape.

    Parameters
    ----------
    patches : np.ndarray
        A numpy array containing the patches to be stitched together.

    shape : Union[List, Tuple]
        The desired shape of the output array.

    tile_size : int
        The size of each tile/patch, e.g., 64 for 64x64 tiles.

    offset : Optional[Union[List, Tuple]], optional
        The y and x offset values for the target array. If not provided, defaults to [0, 0].

    background_value : Optional[Union[int, float]], optional
        The value to use for the background. If not provided, defaults to np.nan.

    Returns
    -------
    np.ndarray
        A numpy array with the original shape, formed by stitching together the provided patches.
    """
    assert len(shape) in [2, 3], "Shape must be a tuple or list of length 2 or 3"
    assert len(patches.shape) == 4, "Patches must be a 4D array"
    assert patches.shape[1] == tile_size, "Patches must be of size tile_size"
    assert patches.shape[2] == tile_size, "Patches must be of size tile_size"
    assert offset is None or len(offset) == 2, "Offset must be a tuple or list of length 2"

    # Set default offset to [0, 0] if not provided
    if offset is None:
        offset = [0, 0]

    # Create an empty target array of the specified shape
    if background_value is None:

        # check if patches.dtype is integer types
        if patches.dtype.kind in "ui":
            target = np.full(shape, np.iinfo(patches.dtype).min, dtype=patches.dtype)
        else:
            target = np.full(shape, np.nan, dtype=patches.dtype)

    else:
        target = np.full(shape, background_value, dtype=patches.dtype)

    # Calculate target dimensions
    target_y = ((shape[0] - offset[0]) // tile_size) * tile_size
    target_x = ((shape[1] - offset[1]) // tile_size) * tile_size

    # Calculate cut dimensions
    cut_y = -((shape[0] - offset[0]) % tile_size)
    cut_x = -((shape[1] - offset[1]) % tile_size)

    # Set cut dimensions to None if they are 0
    cut_y = None if cut_y == 0 else cut_y
    cut_x = None if cut_x == 0 else cut_x

    # Calculate the number of tiles in the y and x dimensions
    num_tiles_y = target_y // tile_size
    num_tiles_x = target_x // tile_size

    # Reshape the patches for stitching
    reshape = patches.reshape(
        num_tiles_y,
        num_tiles_x,
        tile_size,
        tile_size,
        patches.shape[3],
        1,
    )

    # Swap axes to rearrange patches in the correct order for stitching
    swap = reshape.swapaxes(1, 2)

    # Combine the patches into a single array
    destination = swap.reshape(
        num_tiles_y * tile_size,
        num_tiles_x * tile_size,
        patches.shape[3],
    )

    # Assign the combined patches to the target array
    target[offset[0]:cut_y, offset[1]:cut_x] = destination

    return target


def array_to_patches(
    arr: np.ndarray,
    tile_size: int, *,
    n_offsets: int = 0,
    border_check: bool = True,
    channel_last: bool = True,
) -> np.ndarray:
    """Generate patches from an array based on the specified parameters.

    Parameters
    ----------
    arr : np.ndarray
        A numpy array to be divided into patches.

    tile_size : int
        The size of each tile/patch, e.g., 64 for 64x64 tiles.

    n_offsets : int, optional
        The desired number of offsets to be calculated. Default: 0

    border_check : bool, optional
        Whether or not to include border patches. Default: True

    channel_last : bool, optional
        Whether or not the channel dimension is the last dimension. Default: True

    Returns
    -------
    np.ndarray
        The concatenate patches along axis 0. In the order (patches, y, x, channels)
    """
    assert len(arr.shape) == 3, "Array must be 3D"
    if channel_last:
        assert arr.shape[0] >= tile_size, "Array must be larger or equal to tile_size"
        assert arr.shape[1] >= tile_size, "Array must be larger or equal to tile_size"
    else:
        assert arr.shape[1] >= tile_size, "Array must be larger or equal to tile_size"
        assert arr.shape[2] >= tile_size, "Array must be larger or equal to tile_size"

    assert tile_size > 0, "Tile size must be greater than 0"
    assert n_offsets >= 0, "Number of offsets must be greater than or equal to 0"
    assert isinstance(border_check, bool), "Border check must be a boolean"
    assert isinstance(n_offsets, int), "Number of offsets must be an integer"

    if not channel_last:
        arr = channel_first_to_last(arr)

    # Get the list of offsets for both x and y dimensions
    offsets = _get_offsets(tile_size, n_offsets)

    if border_check:
        borders_y, borders_x = _borders_are_necessary_list(arr, tile_size, offsets)

        # TODO: Investigate how to handle smarter. Currently we might get duplicates.
        if borders_y or borders_x:
            offsets.append((0, arr.shape[1] - tile_size))
            offsets.append((arr.shape[0] - tile_size, 0))
            offsets.append((arr.shape[0] - tile_size, arr.shape[1] - tile_size))

    # Initialize an empty list to store the generated patches
    patches = []

    # Iterate through the offsets and generate patches for each offset
    for offset in offsets:
        patches.append(
            _array_to_patches_single(arr, tile_size, offset),
        )

    patches = np.concatenate(patches, axis=0)

    if not channel_last:
        patches = channel_last_to_first(patches)

    return patches
