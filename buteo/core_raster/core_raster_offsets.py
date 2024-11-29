""" ### Create offsets to read rasters in chunks. ### """

# Standard library
from typing import List, Tuple, Sequence
from math import ceil

from buteo.utils.utils_base import _type_check



def _find_optimal_chunk_factors(
    num_chunks: int,
    width: int,
    height: int,
) -> Tuple[int, int]:
    """
    Find optimal factors for splitting image into chunks with minimal aspect ratio difference.

    Parameters
    ----------
    num_chunks : int
        Number of chunks to split into
    width : int
        Image width
    height : int
        Image height

    Returns
    -------
    Tuple[int, int]
        Number of chunks in height and width direction

    Raises
    ------
    ValueError
        If num_chunks is less than 1
    """
    _type_check(num_chunks, [int], "num_chunks")
    _type_check(width, [int], "width")
    _type_check(height, [int], "height")

    if num_chunks < 1:
        raise ValueError("num_chunks must be greater than 0")

    if num_chunks == 1:
        return (1, 1)

    best_factors = (1, 1)
    min_score = float("inf")
    target_aspect = width / height

    for i in range(1, num_chunks + 1):
        if num_chunks % i == 0:
            h_chunks = i
            w_chunks = num_chunks // i

            chunk_width = width / w_chunks
            chunk_height = height / h_chunks
            chunk_aspect = chunk_width / chunk_height
            aspect_diff = abs(chunk_aspect - target_aspect)

            # Calculate coverage efficiency (how many pixels are wasted)
            total_pixels = width * height
            covered_pixels = (int(chunk_width) * w_chunks) * (int(chunk_height) * h_chunks)
            coverage_diff = (total_pixels - covered_pixels) / total_pixels

            # Combined score (weighted sum of aspect difference and coverage)
            score = aspect_diff + coverage_diff

            if score < min_score:
                min_score = score
                best_factors = (h_chunks, w_chunks)

    return best_factors


def _get_chunk_offsets(
    image_shape: Sequence[int],
    num_chunks: int,
    overlap: int = 0,
) -> List[Tuple[int, int, int, int]]:
    """Calculate chunk offsets dividing image into specified number of chunks.
    Expects channel-first format.

    Parameters
    ----------
    image_shape : Tuple[int, ...]
        Image shape (channels, height, width)
    num_chunks : int
        Number of chunks to divide into
    overlap : int, optional
        Overlap in pixels, by default 0

    Returns
    -------
    List[Tuple[int, int, int, int]]
        List of (x_start, y_start, x_pixels, y_pixels)

    Raises
    ------
    ValueError
        If image_shape is invalid or overlap is negative

    TypeError
        If image_shape is not a list or tuple, num_chunks is not an integer, or overlap is not an integer
    """
    _type_check(image_shape, [list, tuple], "image_shape")
    _type_check(num_chunks, [int], "num_chunks")
    _type_check(overlap, [int], "overlap")

    if len(image_shape) != 3:
        raise ValueError("image_shape must have at least 3 dimensions")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")

    _channels, height, width = image_shape

    num_h_chunks, num_w_chunks = _find_optimal_chunk_factors(num_chunks, width, height)

    # Calculate base offsets
    offsets = []
    for h in range(num_h_chunks):
        for w in range(num_w_chunks):
            h_start = h * (height // num_h_chunks)
            w_start = w * (width // num_w_chunks)
            h_end = height if h == num_h_chunks - 1 else (h + 1) * (height // num_h_chunks)
            w_end = width if w == num_w_chunks - 1 else (w + 1) * (width // num_w_chunks)
            offsets.append((w_start, h_start, w_end - w_start, h_end - h_start))

    # Apply overlap if specified
    if overlap > 0:
        overlap_half = ceil(overlap / 2)
        return [(
            max(0, x - overlap_half),
            max(0, y - overlap_half),
            min(size_x + overlap_half, width - max(0, x - overlap_half)),
            min(size_y + overlap_half, height - max(0, y - overlap_half))
        ) for x, y, size_x, size_y in offsets]

    return offsets


def _compute_chunk_positions(
    length: int,
    chunk_size: int,
    overlap: int,
    border_strategy: int,
) -> List[int]:
    """
    Compute the positions of chunks along an axis.

    Parameters
    ----------
    length : int
        The length of the axis.
    chunk_size : int
        The size of the chunks.
    overlap : int
        The amount of overlap between chunks.
    border_strategy : int
        The border strategy to use when splitting the raster into chunks. Can be 1, 2, or 3.
        1. Ignore border chunks if they do not fit the chunk size.
        2. Oversample border chunks to fit the chunk size.
        3. Shrink the last chunk to fit the image size (creates uneven chunks).

    Returns
    -------
    List[int]
        A list of positions along the axis.

    Raises
    ------
    TypeError
        If input parameters have incorrect types.

    ValueError
        If input parameters have invalid values.
    """
    _type_check(length, [int], "length")
    _type_check(chunk_size, [int], "chunk_size")
    _type_check(overlap, [int], "overlap")
    _type_check(border_strategy, [int], "border_strategy")

    positions = []
    step = chunk_size - overlap
    if step <= 0:
        raise ValueError("Overlap must be smaller than chunk size.")

    pos = 0
    while pos < length:
        if pos + chunk_size > length:
            if border_strategy == 1:
                break
            elif border_strategy == 2:
                pos = length - chunk_size
            elif border_strategy == 3:
                pass  # Last chunk may be smaller than chunk_size
        if pos not in positions:
            positions.append(pos)
        if pos + chunk_size >= length:
            break
        pos += step

    return positions


def _get_chunk_offsets_fixed_size(
    image_shape: Sequence[int],
    chunk_size_x: int,
    chunk_size_y: int,
    border_strategy: int = 1,
    overlap: int = 0,
) -> List[Tuple[int, int, int, int]]:
    """Get chunk offsets for a fixed chunk size.
    Expects channel-first format.

    Parameters
    ----------
    image_shape : Sequence[int]
        A tuple or list containing the image shape in channel-first format (channels, height, width).
    chunk_size_x : int
        The size of the chunks in the x-direction.
    chunk_size_y : int
        The size of the chunks in the y-direction.
    border_strategy : int, optional
        The border strategy to use when splitting the raster into chunks. Can be 1, 2, or 3. Default: 1.
        1. Ignore border chunks if they do not fit the chunk size.
        2. Oversample border chunks to fit the chunk size.
        3. Shrink the last chunk to fit the image size (creates uneven chunks).
    overlap : int, optional
        The amount of overlap to apply to each chunk, in pixels. Default: 0.

    Returns
    -------
    List[Tuple[int, int, int, int]]
        A list of tuples, each containing the chunk offsets and dimensions in the format:
        `(x_start, y_start, x_pixels, y_pixels)`.

    Raises
    ------
    TypeError
        If input parameters have incorrect types.
    ValueError
        If input parameters have invalid values.
    RuntimeError
        If chunk sizes are inconsistent when using border strategies 1 or 2.
    """
    _type_check(image_shape, [list, tuple], "image_shape")
    _type_check(chunk_size_x, [int], "chunk_size_x")
    _type_check(chunk_size_y, [int], "chunk_size_y")
    _type_check(border_strategy, [int], "border_strategy")
    _type_check(overlap, [int], "overlap")

    if chunk_size_x <= 0 or chunk_size_y <= 0:
        raise ValueError("chunk sizes must be greater than 0.")
    if overlap < 0:
        raise ValueError("overlap must be non-negative.")
    if border_strategy not in [1, 2, 3]:
        raise ValueError("border_strategy must be 1, 2, or 3.")
    if len(image_shape) != 3:
        raise ValueError("image_shape must have at least 3 dimensions (channels, height, width).")
    if overlap >= chunk_size_x or overlap >= chunk_size_y:
        raise ValueError("overlap must be smaller than chunk sizes.")

    _channels, height, width = image_shape

    # Compute positions along x and y
    x_positions = _compute_chunk_positions(width, chunk_size_x, overlap, border_strategy)
    y_positions = _compute_chunk_positions(height, chunk_size_y, overlap, border_strategy)

    chunk_offsets = []
    for y_start in y_positions:
        for x_start in x_positions:
            x_end = x_start + chunk_size_x
            y_end = y_start + chunk_size_y

            if x_end > width:
                if border_strategy == 3:
                    x_pixels = width - x_start
                else:
                    x_pixels = chunk_size_x
            else:
                x_pixels = chunk_size_x

            if y_end > height:
                if border_strategy == 3:
                    y_pixels = height - y_start
                else:
                    y_pixels = chunk_size_y
            else:
                y_pixels = chunk_size_y

            # Adjust for overlap
            x_pixels = min(x_pixels, width - x_start)
            y_pixels = min(y_pixels, height - y_start)

            # Ensure consistent chunk sizes for border strategies 1 and 2
            if border_strategy in [1, 2]:
                if x_pixels != chunk_size_x or y_pixels != chunk_size_y:
                    raise RuntimeError("Parsing error in offsets.")

            chunk_offsets.append((x_start, y_start, x_pixels, y_pixels))

    return chunk_offsets
