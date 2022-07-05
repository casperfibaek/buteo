"""
This module contains utility functions for the patch creation process.

TODO:
    - Improve documentation
    - Change defaults to None and add checks
"""

import numpy as np


def array_to_blocks(arr, tile_size, offset=[0, 0]):
    blocks_y = (arr.shape[0] - offset[1]) // tile_size
    blocks_x = (arr.shape[1] - offset[0]) // tile_size

    cut_y = -((arr.shape[0] - offset[1]) % tile_size)
    cut_x = -((arr.shape[1] - offset[0]) % tile_size)

    cut_y = None if cut_y == 0 else cut_y
    cut_x = None if cut_x == 0 else cut_x

    reshaped = arr[offset[1] : cut_y, offset[0] : cut_x].reshape(
        blocks_y,
        tile_size,
        blocks_x,
        tile_size,
        arr.shape[2],
    )

    swaped = reshaped.swapaxes(1, 2)
    merge = swaped.reshape(-1, tile_size, tile_size, arr.shape[2])

    return merge


def blocks_to_array(blocks, og_shape, tile_size, offset=[0, 0]):
    with np.errstate(invalid="ignore"):
        target = np.empty(og_shape) * np.nan

    target_y = ((og_shape[0] - offset[1]) // tile_size) * tile_size
    target_x = ((og_shape[1] - offset[0]) // tile_size) * tile_size

    cut_y = -((og_shape[0] - offset[1]) % tile_size)
    cut_x = -((og_shape[1] - offset[0]) % tile_size)

    cut_x = None if cut_x == 0 else cut_x
    cut_y = None if cut_y == 0 else cut_y

    reshape = blocks.reshape(
        target_y // tile_size,
        target_x // tile_size,
        tile_size,
        tile_size,
        blocks.shape[3],
        1,
    )

    swap = reshape.swapaxes(1, 2)

    destination = swap.reshape(
        (target_y // tile_size) * tile_size,
        (target_x // tile_size) * tile_size,
        blocks.shape[3],
    )

    target[offset[1] : cut_y, offset[0] : cut_x] = destination

    return target


def get_overlaps(arr, offsets, tile_size, border_check=True):
    arr_offsets = []

    for offset in offsets:
        arr_offsets.append(array_to_blocks(arr, tile_size, offset))

    if border_check:
        found = False
        border = None
        for end in ["right", "bottom", "corner"]:
            if end == "right" and (arr.shape[1] % tile_size) != 0:
                found = True
                border = arr[:, -tile_size:]
            elif end == "bottom" and (arr.shape[0] % tile_size) != 0:
                found = True
                border = arr[-tile_size:, :]
            elif (
                end == "corner"
                and ((arr.shape[1] % tile_size) != 0)
                and ((arr.shape[1] % tile_size) != 0)
            ):
                found = True
                border = arr[-tile_size:, -tile_size:]

            if found:
                arr_offsets.append(array_to_blocks(border, tile_size, offset=[0, 0]))

    return arr_offsets
