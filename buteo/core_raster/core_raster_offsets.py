""" ### Create offsets to read rasters in patches. ### """

# Standard library
from typing import List, Tuple, Sequence
from math import ceil

from buteo.utils.utils_base import _type_check



def _find_optimal_patch_factors(
    num_patches: int,
    width: int,
    height: int,
) -> Tuple[int, int]:
    """
    Find optimal factors for splitting image into patches with minimal aspect ratio difference.

    Parameters
    ----------
    num_patches : int
        Number of patches to split into
    width : int
        Image width
    height : int
        Image height

    Returns
    -------
    Tuple[int, int]
        Number of patches in height and width direction

    Raises
    ------
    ValueError
        If num_patches is less than 1
    """
    _type_check(num_patches, [int], "num_patches")
    _type_check(width, [int], "width")
    _type_check(height, [int], "height")

    if num_patches < 1:
        raise ValueError("num_patches must be greater than 0")

    if num_patches == 1:
        return (1, 1)

    best_factors = (1, 1)
    min_score = float("inf")
    target_aspect = width / height

    for i in range(1, num_patches + 1):
        if num_patches % i == 0:
            h_patches = i
            w_patches = num_patches // i

            patch_width = width / w_patches
            patch_height = height / h_patches
            patch_aspect = patch_width / patch_height
            aspect_diff = abs(patch_aspect - target_aspect)

            # Calculate coverage efficiency (how many pixels are wasted)
            total_pixels = width * height
            covered_pixels = (int(patch_width) * w_patches) * (int(patch_height) * h_patches)
            coverage_diff = (total_pixels - covered_pixels) / total_pixels

            # Combined score (weighted sum of aspect difference and coverage)
            score = aspect_diff + coverage_diff

            if score < min_score:
                min_score = score
                best_factors = (h_patches, w_patches)

    return best_factors


def _get_patch_offsets(
    image_shape: Sequence[int],
    num_patches: int,
    overlap: int = 0,
) -> List[Tuple[int, int, int, int]]:
    """Calculate patch offsets dividing image into specified number of patches.
    Expects channel-first format.

    Parameters
    ----------
    image_shape : Tuple[int, ...]
        Image shape (channels, height, width)
    num_patches : int
        Number of patches to divide into
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
        If image_shape is not a list or tuple, num_patches is not an integer, or overlap is not an integer
    """
    _type_check(image_shape, [list, tuple], "image_shape")
    _type_check(num_patches, [int], "num_patches")
    _type_check(overlap, [int], "overlap")

    if len(image_shape) != 3:
        raise ValueError("image_shape must have at least 3 dimensions")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")

    _channels, height, width = image_shape

    num_h_patches, num_w_patches = _find_optimal_patch_factors(num_patches, width, height)

    # Calculate base offsets
    offsets = []
    for h in range(num_h_patches):
        for w in range(num_w_patches):
            h_start = h * (height // num_h_patches)
            w_start = w * (width // num_w_patches)
            h_end = height if h == num_h_patches - 1 else (h + 1) * (height // num_h_patches)
            w_end = width if w == num_w_patches - 1 else (w + 1) * (width // num_w_patches)
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


def _compute_patch_positions(
    length: int,
    patch_size: int,
    overlap: int,
    border_strategy: int,
) -> List[int]:
    """
    Compute the positions of patches along an axis.

    Parameters
    ----------
    length : int
        The length of the axis.
    patch_size : int
        The size of the patches.
    overlap : int
        The amount of overlap between patches.
    border_strategy : int
        The border strategy to use when splitting the raster into patches. Can be 1, 2, or 3.
        1. Ignore border patches if they do not fit the patch size.
        2. Oversample border patches to fit the patch size.
        3. Shrink the last patch to fit the image size (creates uneven patches).

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
    _type_check(patch_size, [int], "patch_size")
    _type_check(overlap, [int], "overlap")
    _type_check(border_strategy, [int], "border_strategy")

    positions = []
    step = patch_size - overlap
    if step <= 0:
        raise ValueError("Overlap must be smaller than patch size.")

    pos = 0
    while pos < length:
        if pos + patch_size > length:
            if border_strategy == 1:
                break
            elif border_strategy == 2:
                pos = length - patch_size
            elif border_strategy == 3:
                pass  # Last patch may be smaller than patch_size
        if pos not in positions:
            positions.append(pos)
        if pos + patch_size >= length:
            break
        pos += step

    return positions


def _get_patch_offsets_fixed_size(
    image_shape: Sequence[int],
    patch_size_x: int,
    patch_size_y: int,
    border_strategy: int = 1,
    overlap: int = 0,
) -> List[Tuple[int, int, int, int]]:
    """Get patch offsets for a fixed patch size.
    Expects channel-first format.

    Parameters
    ----------
    image_shape : Sequence[int]
        A tuple or list containing the image shape in channel-first format (channels, height, width).
    patch_size_x : int
        The size of the patches in the x-direction.
    patch_size_y : int
        The size of the patches in the y-direction.
    border_strategy : int, optional
        The border strategy to use when splitting the raster into patches. Can be 1, 2, or 3. Default: 1.
        1. Ignore border patches if they do not fit the patch size.
        2. Oversample border patches to fit the patch size.
        3. Shrink the last patch to fit the image size (creates uneven patches).
    overlap : int, optional
        The amount of overlap to apply to each patch, in pixels. Default: 0.

    Returns
    -------
    List[Tuple[int, int, int, int]]
        A list of tuples, each containing the patch offsets and dimensions in the format:
        `(x_start, y_start, x_pixels, y_pixels)`.

    Raises
    ------
    TypeError
        If input parameters have incorrect types.
    ValueError
        If input parameters have invalid values.
    RuntimeError
        If patch sizes are inconsistent when using border strategies 1 or 2.
    """
    _type_check(image_shape, [list, tuple], "image_shape")
    _type_check(patch_size_x, [int], "patch_size_x")
    _type_check(patch_size_y, [int], "patch_size_y")
    _type_check(border_strategy, [int], "border_strategy")
    _type_check(overlap, [int], "overlap")

    if patch_size_x <= 0 or patch_size_y <= 0:
        raise ValueError("patch sizes must be greater than 0.")
    if overlap < 0:
        raise ValueError("overlap must be non-negative.")
    if border_strategy not in [1, 2, 3]:
        raise ValueError("border_strategy must be 1, 2, or 3.")
    if len(image_shape) != 3:
        raise ValueError("image_shape must have at least 3 dimensions (channels, height, width).")
    if overlap >= patch_size_x or overlap >= patch_size_y:
        raise ValueError("overlap must be smaller than patch sizes.")

    _channels, height, width = image_shape

    # Compute positions along x and y
    x_positions = _compute_patch_positions(width, patch_size_x, overlap, border_strategy)
    y_positions = _compute_patch_positions(height, patch_size_y, overlap, border_strategy)

    patch_offsets = []
    for y_start in y_positions:
        for x_start in x_positions:
            x_end = x_start + patch_size_x
            y_end = y_start + patch_size_y

            if x_end > width:
                if border_strategy == 3:
                    x_pixels = width - x_start
                else:
                    x_pixels = patch_size_x
            else:
                x_pixels = patch_size_x

            if y_end > height:
                if border_strategy == 3:
                    y_pixels = height - y_start
                else:
                    y_pixels = patch_size_y
            else:
                y_pixels = patch_size_y

            # Adjust for overlap
            x_pixels = min(x_pixels, width - x_start)
            y_pixels = min(y_pixels, height - y_start)

            # Ensure consistent patch sizes for border strategies 1 and 2
            if border_strategy in [1, 2]:
                if x_pixels != patch_size_x or y_pixels != patch_size_y:
                    raise RuntimeError("Parsing error in offsets.")

            patch_offsets.append((x_start, y_start, x_pixels, y_pixels))

    return patch_offsets
