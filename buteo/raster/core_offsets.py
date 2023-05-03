"""
Create offsets to read rasters in chunks.
"""

# Standard library
from typing import List, Union, Tuple



def _split_shape_into_offsets(
    shape: Union[List[int], Tuple[int]],
    offsets_x: int = 2,
    offsets_y: int = 2,
    overlap_x: int = 0,
    overlap_y: int = 0,
) -> List[int]:
    """
    Split a shape into offsets. Usually used for splitting an image into offsets to reduce RAM needed.
    
    Parameters
    ----------
    shape : tuple or list
        The shape to split into offsets. (height, width, ...)

    offsets_x : int, optional
        The number of offsets to split the shape into in the x-direction. Default: 2.

    offsets_y : int, optional
        The number of offsets to split the shape into in the y-direction. Default: 2.

    overlap_x : int, optional
        The number of pixels to overlap in the x-direction. Default: 0.

    overlap_y : int, optional
        The number of pixels to overlap in the y-direction. Default: 0.

    Returns
    -------
    list
        The offsets. `[x_offset, y_offset, x_size, y_size]`
    """
    print("WARNING: This is deprecated and will be removed in a future update. Please use get_chunk_offsets instead.")
    height = shape[0]
    width = shape[1]

    x_remainder = width % offsets_x
    y_remainder = height % offsets_y

    x_offsets = [0]
    x_sizes = []
    for _ in range(offsets_x - 1):
        x_offsets.append(x_offsets[-1] + (width // offsets_x) - overlap_x)
    x_offsets[-1] -= x_remainder

    for idx, _ in enumerate(x_offsets):
        if idx == len(x_offsets) - 1:
            x_sizes.append(width - x_offsets[idx])
        elif idx == 0:
            x_sizes.append(x_offsets[1] + overlap_x)
        else:
            x_sizes.append(x_offsets[idx + 1] - x_offsets[idx] + overlap_x)

    y_offsets = [0]
    y_sizes = []
    for _ in range(offsets_y - 1):
        y_offsets.append(y_offsets[-1] + (height // offsets_y) - overlap_y)
    y_offsets[-1] -= y_remainder

    for idx, _ in enumerate(y_offsets):
        if idx == len(y_offsets) - 1:
            y_sizes.append(height - y_offsets[idx])
        elif idx == 0:
            y_sizes.append(y_offsets[1] + overlap_y)
        else:
            y_sizes.append(y_offsets[idx + 1] - y_offsets[idx] + overlap_y)

    offsets = []

    for idx_col, _ in enumerate(y_offsets):
        for idx_row, _ in enumerate(x_offsets):
            offsets.append([
                x_offsets[idx_row],
                y_offsets[idx_col],
                x_sizes[idx_row],
                y_sizes[idx_col],
            ])

    return offsets


def _apply_overlap_to_offsets(
    list_of_offsets: List[Tuple[int, int, int, int]],
    overlap: int,
    shape: Tuple[int, int],
) -> List[List[int]]:
    """
    Apply an overlap to a list of chunk offsets.
    
    The function adjusts the starting position and size of each chunk to apply the specified overlap.
    
    Parameters
    ----------
    list_of_offsets : List[Tuple[int, int, int, int]]
        A list of tuples containing the chunk offsets and dimensions in the format `(x_start, y_start, x_pixels, y_pixels)`.

    overlap : int
        The amount of overlap to apply to each chunk, in pixels.

    shape : Tuple[int, int]
        A tuple containing the height and width of the image: `(Height, Width)`.
        
    Returns
    -------
    List[List[int]]
        A list of lists, each containing the adjusted chunk offsets and dimensions in the format `[x_start, y_start, x_pixels, y_pixels]`.
    """
    height, width = shape
    new_offsets = []
    for start_x, start_y, size_x, size_y in list_of_offsets:
        new_start_x = max(0, start_x - (overlap // 2))
        new_start_y = max(0, start_y - (overlap // 2))

        new_size_x = size_x + overlap
        new_size_y = size_y + overlap

        # If we are over the adjust, bring it back.
        if new_size_x + new_start_x > width:
            new_size_x = new_size_x - ((new_size_x + new_start_x) - width)

        if new_size_y + new_start_y > height:
            new_size_y = new_size_y - ((new_size_y + new_start_y) - height)

        new_offsets.append([
            new_start_x, new_start_y, new_size_x, new_size_y,
        ])

    return new_offsets


def _get_chunk_offsets(
    image_shape: Tuple[int, int],
    num_chunks: int,
    overlap: int = 0,
) -> List[Tuple[int, int, int, int]]:
    """
    Calculate chunk offsets for dividing an image into a specified number of chunks with minimal circumference.

    The function finds the optimal configuration of chunks to minimize the circumference and ensure the whole image
    is captured.

    Parameters
    ----------
    image_shape : Tuple[int, int]
        A tuple containing the height and width of the image. (Height, Width)

    num_chunks : int
        The number of chunks to divide the image into.

    Returns
    -------
    List[Tuple[int, int, int, int]]
        A list of tuples, each containing the chunk offsets and dimensions in the format: `(x_start, y_start, x_pixels, y_pixels)`.

    Raises
    ------
    ValueError
        If the number of chunks is too high for the given image size.
    """
    height, width = image_shape[:2]

    # Find the factors of num_chunks that minimize the circumference of the chunks
    min_circumference = float("inf")
    best_factors = (1, num_chunks)

    for i in range(1, num_chunks + 1):
        if num_chunks % i == 0:
            num_h_chunks = i
            num_w_chunks = num_chunks // i

            chunk_height = height // num_h_chunks
            chunk_width = width // num_w_chunks

            # Calculate the circumference of the current chunk configuration
            circumference = 2 * (chunk_height + chunk_width)

            if circumference < min_circumference:
                min_circumference = circumference
                best_factors = (num_h_chunks, num_w_chunks)

    num_h_chunks, num_w_chunks = best_factors

    # Initialize an empty list to store the chunk offsets
    chunk_offsets = []

    # Iterate through the image and create chunk offsets
    for h in range(num_h_chunks):
        for w in range(num_w_chunks):
            h_start = h * (height // num_h_chunks)
            w_start = w * (width // num_w_chunks)

            # If the current chunk is the last one in its row or column, adjust its size
            h_end = height if h == num_h_chunks - 1 else (h + 1) * (height // num_h_chunks)
            w_end = width if w == num_w_chunks - 1 else (w + 1) * (width // num_w_chunks)

            x_pixels = w_end - w_start
            y_pixels = h_end - h_start

            chunk_offsets.append((w_start, h_start, x_pixels, y_pixels))

    if overlap > 0:
        return _apply_overlap_to_offsets(chunk_offsets, overlap, image_shape)

    return chunk_offsets
