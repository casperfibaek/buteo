""" Tests for raster/convolution.py """
# pylint: disable=missing-function-docstring

# Standard library
import sys; sys.path.append("../")

# Internal
from buteo.raster.core_offsets import (
    _get_chunk_offsets,
    _get_chunk_offsets_fixed_size,
)



def test_get_chunk_offsets_without_overlap():
    image_shape = (100, 100)
    num_chunks = 4
    chunk_offsets = _get_chunk_offsets(image_shape, num_chunks)

    expected_chunk_offsets = [
        (0, 0, 50, 50),
        (50, 0, 50, 50),
        (0, 50, 50, 50),
        (50, 50, 50, 50),
    ]

    used = []

    for offset in chunk_offsets:
        assert offset in expected_chunk_offsets and offset not in used
        used.append(offset)

    assert len(used) == len(expected_chunk_offsets)

def test_get_chunk_offsets_with_overlap():
    image_shape = (100, 100, 3)
    num_chunks = 4
    overlap = 10
    chunk_offsets = _get_chunk_offsets(image_shape, num_chunks, overlap=overlap)

    expected_chunk_offsets = [
        (0, 0, 55, 55),
        (45, 0, 55, 55),
        (0, 45, 55, 55),
        (45, 45, 55, 55),
    ]

    used = []
    for offset in chunk_offsets:
        assert offset in expected_chunk_offsets and offset not in used
        used.append(offset)

    assert len(used) == len(expected_chunk_offsets)

def test_get_chunk_offsets_fixed_size():
    # Test case 1: Perfectly divisible image with border_strategy = 1
    img_shape = (12, 12)
    chunk_size_x = 3
    chunk_size_y = 4
    border_strategy = 1
    channel_last = True
    expected_chunk_offsets = [
        (0, 0, 3, 4), (3, 0, 3, 4), (6, 0, 3, 4), (9, 0, 3, 4),
        (0, 4, 3, 4), (3, 4, 3, 4), (6, 4, 3, 4), (9, 4, 3, 4),
        (0, 8, 3, 4), (3, 8, 3, 4), (6, 8, 3, 4), (9, 8, 3, 4),
    ]
    chunk_offsets = _get_chunk_offsets_fixed_size(img_shape, chunk_size_x, chunk_size_y, border_strategy, channel_last=channel_last)

    used = []
    for offset in chunk_offsets:
        assert offset in expected_chunk_offsets and offset not in used
        used.append(offset)

    assert len(used) == len(expected_chunk_offsets)


    # Test case 2: Image with extra pixels on the border, border_strategy = 1
    img_shape = (10, 10)
    chunk_size_x = 4
    chunk_size_y = 4
    border_strategy = 1
    channel_last = True
    expected_chunk_offsets = [
        (0, 0, 4, 4), (4, 0, 4, 4),
        (0, 4, 4, 4), (4, 4, 4, 4),
    ]
    chunk_offsets = _get_chunk_offsets_fixed_size(img_shape, chunk_size_x, chunk_size_y, border_strategy, channel_last=channel_last)

    used = []
    for offset in chunk_offsets:
        assert offset in expected_chunk_offsets and offset not in used
        used.append(offset)

    assert len(used) == len(expected_chunk_offsets)


    # Test case 3: Image with extra pixels on the border, border_strategy = 2
    img_shape = (10, 10)
    chunk_size_x = 4
    chunk_size_y = 4
    border_strategy = 2
    channel_last = True
    expected_chunk_offsets = [
        (0, 0, 4, 4), (4, 0, 4, 4), (6, 0, 4, 4),
        (0, 4, 4, 4), (4, 4, 4, 4), (6, 4, 4, 4),
        (0, 6, 4, 4), (4, 6, 4, 4), (6, 6, 4, 4),
    ]
    chunk_offsets = _get_chunk_offsets_fixed_size(img_shape, chunk_size_x, chunk_size_y, border_strategy, channel_last=channel_last)

    used = []
    for offset in chunk_offsets:
        assert offset in expected_chunk_offsets and offset not in used
        used.append(offset)

    assert len(used) == len(expected_chunk_offsets)
