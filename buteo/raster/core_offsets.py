""" ### Create offsets to read rasters in chunks. ### """

# Standard library
from typing import List, Tuple
from math import ceil



def _get_chunk_offsets(
    image_shape: Tuple[int, int],
    num_chunks: int,
    overlap: int = 0,
    channel_last: bool = True,
) -> List[Tuple[int, int, int, int]]:
    """Calculate chunk offsets for dividing an image into a specified number of chunks with minimal circumference.

    The function finds the optimal configuration of chunks to minimize the circumference and ensure the whole image
    is captured.

    Parameters
    ----------
    image_shape : Tuple[int, int]
        A tuple containing the height and width of the image. (Height, Width)

    num_chunks : int
        The number of chunks to divide the image into.

    overlap : int, optional
        The amount of overlap to apply to each chunk, in pixels. Default: 0.

    channel_last : bool, optional
        Whether the image has the channels as the last dimension. Default: True.

    Returns
    -------
    List[Tuple[int, int, int, int]]
        A list of tuples, each containing the chunk offsets and dimensions in the format: `(x_start, y_start, x_pixels, y_pixels)`.

    Raises
    ------
    ValueError
        If the number of chunks is too high for the given image size.
    """
    height, width = image_shape[:2] if channel_last else image_shape[1:]

    best_factors = None
    min_aspect_ratio_diff = float("inf")

    for i in range(1, num_chunks + 1):
        if num_chunks % i == 0:
            num_h_chunks = i
            num_w_chunks = num_chunks // i

            chunk_height = height // num_h_chunks
            chunk_width = width // num_w_chunks

            aspect_ratio = chunk_width / chunk_height
            aspect_ratio_diff = abs(aspect_ratio - 1)

            if aspect_ratio_diff < min_aspect_ratio_diff:
                min_aspect_ratio_diff = aspect_ratio_diff
                best_factors = (num_h_chunks, num_w_chunks)

    num_h_chunks, num_w_chunks = best_factors

    chunk_offsets = []

    for h in range(num_h_chunks):
        for w in range(num_w_chunks):
            h_start = h * (height // num_h_chunks)
            w_start = w * (width // num_w_chunks)

            h_end = height if h == num_h_chunks - 1 else (h + 1) * (height // num_h_chunks)
            w_end = width if w == num_w_chunks - 1 else (w + 1) * (width // num_w_chunks)

            x_pixels = w_end - w_start
            y_pixels = h_end - h_start

            chunk_offsets.append((w_start, h_start, x_pixels, y_pixels))

    if overlap > 0:
        overlap_half = ceil(overlap / 2)

        new_offsets = []
        for start_x, start_y, size_x, size_y in chunk_offsets:
            new_start_x = max(0, start_x - overlap_half)
            new_start_y = max(0, start_y - overlap_half)

            new_size_x = size_x + overlap_half
            new_size_y = size_y + overlap_half

            # If we are over the adjust, bring it back.
            if new_size_x + new_start_x > width:
                new_size_x = new_size_x - ((new_size_x + new_start_x) - width)

            if new_size_y + new_start_y > height:
                new_size_y = new_size_y - ((new_size_y + new_start_y) - height)

            new_offsets.append((
                new_start_x, new_start_y, new_size_x, new_size_y,
            ))

        return new_offsets

    return chunk_offsets


def _get_chunk_offsets_fixed_size(
    image_shape,
    chunk_size_x: int,
    chunk_size_y: int,
    border_strategy: int = 1,
    overlap: int = 0,
    *,
    channel_last: bool = True,
) -> List[Tuple[int, int, int, int]]:
    """Get chunk offsets for a fixed chunk size.

    Parameters
    ----------
    image_shape : Tuple[int, int]
        A tuple containing the height and width of the image. (Height, Width)

    chunk_size_x : int
        The size of the chunks in the x-direction.

    chunk_size_y : int
        The size of the chunks in the y-direction.

    border_strategy : int, optional
        The border strategy to use when splitting the raster into chunks.
        border_strategy ignored when chunk_size and overlaps are provided.
        Only applied when chunk_size is provided. Can be 1 or 2. Default: 1.
        1. Ignore the border chunks if they do not fit the chunk size.
        2. Oversample the border chunks to fit the chunk size.
        3. Shrink the last chunk to fit the image size. (Creates uneven chunks.)

    overlap : int, optional
        The amount of overlap to apply to each chunk, in pixels. Default: 0.

    channel_last : bool, optional
        Whether the image has the channels as the last dimension. Default: True.

    Returns
    -------
    List[Tuple[int, int, int, int]]
        A list of tuples, each containing the chunk offsets and dimensions in the format: `(x_start, y_start, x_pixels, y_pixels)`.
    """
    assert isinstance(image_shape, (list, tuple)), "image_shape must be a list or tuple."
    assert isinstance(chunk_size_x, int), "chunk_size_x must be an integer."
    assert isinstance(chunk_size_y, int), "chunk_size_y must be an integer."
    assert isinstance(border_strategy, int), "border_strategy must be an integer."
    assert isinstance(channel_last, bool), "channel_last must be a boolean."
    assert chunk_size_x > 0, "chunk_size_x must be > 0."
    assert border_strategy in [1, 2, 3], "border_strategy must be 1 or 2."
    height, width = image_shape[:2] if channel_last else image_shape[1:]

    num_chunks_x = width // chunk_size_x + (1 if width % chunk_size_x > 0 else 0)
    num_chunks_y = height // chunk_size_y + (1 if height % chunk_size_y > 0 else 0)

    chunk_offsets = []

    for y in range(num_chunks_y):
        for x in range(num_chunks_x):
            x_start = x * chunk_size_x + overlap
            y_start = y * chunk_size_y + overlap

            x_end = (x + 1) * chunk_size_x + overlap
            y_end = (y + 1) * chunk_size_y + overlap

            if x_end > width or y_end > height:
                if border_strategy == 1:
                    if x_end > width or y_end > height:
                        continue
                elif border_strategy == 2:
                    x_end = min(x_end, width)
                    y_end = min(y_end, height)

            x_pixels = x_end - x_start
            y_pixels = y_end - y_start

            if border_strategy in [1, 2] and (x_pixels < chunk_size_x or y_pixels < chunk_size_y):
                x_pixels = chunk_size_x
                y_pixels = chunk_size_y

            if x_start + x_pixels > width:
                if border_strategy == 1:
                    continue
                elif border_strategy == 2:
                    x_start = width - x_pixels
                elif border_strategy == 3:
                    x_pixels = width - x_start

            if y_start + y_pixels > height:
                if border_strategy == 1:
                    continue
                elif border_strategy == 2:
                    y_start = height - y_pixels
                elif border_strategy == 3:
                    y_pixels = height - y_start

            new_offset = (x_start, y_start, x_pixels, y_pixels)

            if border_strategy in [1, 2] and (new_offset[2] != chunk_size_x or new_offset[3] != chunk_size_y):
                raise RuntimeError("Parsing error in offsets.")

            if new_offset not in chunk_offsets:
                chunk_offsets.append(new_offset)

    if overlap > 0:
        num_chunks_x = width // chunk_size_x + (1 if width % chunk_size_x > 0 else 0)
        num_chunks_y = height // chunk_size_y + (1 if height % chunk_size_y > 0 else 0)

        for x in range(num_chunks_x):
            x_start = x * chunk_size_x
            x_end = (x + 1) * chunk_size_x

            x_pixels = chunk_size_x

            if x_start < x_pixels and border_strategy == 1:
                continue

            if x_end > width:
                if border_strategy == 1:
                    continue
                elif border_strategy == 2:
                    x_start = width - chunk_size_x
                elif border_strategy == 3:
                    x_pixels = width - x_start

            new_offset = (x_start, 0, x_pixels, chunk_size_y)

            if border_strategy in [1, 2] and (new_offset[2] != chunk_size_x or new_offset[3] != chunk_size_y):
                raise RuntimeError("Parsing error in offsets.")

            if new_offset not in chunk_offsets:
                chunk_offsets.append(new_offset)

        for y in range(num_chunks_y):
            y_start = y * chunk_size_y
            y_end = (y + 1) * chunk_size_y

            y_pixels = chunk_size_y

            if y_start < y_pixels and border_strategy == 1:
                continue

            if y_end > height:
                if border_strategy == 1:
                    continue
                elif border_strategy == 2:
                    y_start = height - chunk_size_y
                elif border_strategy == 3:
                    y_pixels = height - y_start

            new_offset = (0, y_start, chunk_size_x, y_pixels)

            if border_strategy in [1, 2] and (new_offset[2] != chunk_size_x or new_offset[3] != chunk_size_y):
                raise RuntimeError("Parsing error in offsets.")

            if new_offset not in chunk_offsets:
                chunk_offsets.append(new_offset)

    return chunk_offsets
