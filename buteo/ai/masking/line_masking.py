""" ### Line masking functions for images. ###"""

# External
import numpy as np
from numba import jit, prange


@jit(nopython=True, nogil=True, cache=True, fastmath=True)
def mask_lines_2d(
    arr: np.ndarray,
    p: float = 0.05,
    max_height: float = 1.0,
    max_width: float = 1.0,
    min_height: float = 0.1,
    min_width: float = 0.1,
    max_size: int = 3,
    min_size: int = 1,
    channel_last: bool = True,
):
    """
    Creates mask with random horizontal and vertical lines.
    
    Parameters
    ----------
    arr : np.ndarray
        The image to create a line mask for.
    p : float, optional
        The probability of creating a line at each row/column, default: 0.05.
    max_height : float, optional
        Maximum height of vertical lines as a proportion of image height, default: 1.0.
    max_width : float, optional
        Maximum width of horizontal lines as a proportion of image width, default: 1.0.
    min_height : float, optional
        Minimum height of vertical lines as a proportion of image height, default: 0.1.
    min_width : float, optional
        Minimum width of horizontal lines as a proportion of image width, default: 0.1.
    max_size : int, optional
        Maximum thickness of lines in pixels, default: 3.
    min_size : int, optional
        Minimum thickness of lines in pixels, default: 1.
    channel_last : bool, optional
        Whether the image is in (height, width, channels) or (channels, height, width) format.
        
    Returns
    -------
    np.ndarray
        A mask where 0 indicates masked pixels and 1 indicates unmasked pixels.
    """
    height, width, channels = arr.shape
    mask_shape = (height, width, channels)

    if not channel_last:
        channels, height, width = arr.shape
        mask_shape = (channels, height, width)

    mask = np.ones(mask_shape, dtype=np.uint8)
    zero = np.array(0, dtype=np.uint8)

    mask_height = (np.random.uniform(0.0, 1.0, size=(height, )) < p).astype(np.uint8)
    mask_width = (np.random.uniform(0.0, 1.0, size=(width, )) < p).astype(np.uint8)

    mask_height_indices = np.where(mask_height == 1)[0]
    mask_width_indices = np.where(mask_width == 1)[0]

    vertical_lines = mask_height.sum()
    horizontal_lines = mask_width.sum()

    if vertical_lines == 0 and horizontal_lines == 0:
        return mask
    
    vertical_start = np.random.randint(0, height, size=(vertical_lines, ))
    vertical_end = vertical_start + np.floor(np.random.uniform(min_height, max_height, size=(vertical_lines, )) * height).astype(np.int64)

    for i in range(vertical_end.shape[0]):
        if vertical_end[i] > height:
            vertical_end[i] = height

    horizontal_start = np.random.randint(0, width, size=(horizontal_lines, ))
    horizontal_end = horizontal_start + np.floor(np.random.uniform(min_width, max_width, size=(horizontal_lines, )) * width).astype(np.int64)

    for j in range(horizontal_end.shape[0]):
        if horizontal_end[j] > width:
            horizontal_end[j] = width

    if channel_last:
        for idx, pos_y in enumerate(mask_height_indices):
            size = np.random.randint(min_size, max_size + 1)
            size_half = size // 2

            mask[vertical_start[idx]:vertical_end[idx], pos_y - size_half:pos_y + size_half + 1 , :] = zero

        for idx, pos_x in enumerate(mask_width_indices):
            size = np.random.randint(min_size, max_size + 1)
            size_half = size // 2

            mask[pos_x - size_half: pos_x + size_half + 1, horizontal_start[idx]:horizontal_end[idx], :] = zero
    else:
        for idx, pos_y in enumerate(mask_height_indices):
            size = np.random.randint(min_size, max_size + 1)
            size_half = size // 2
            mask[:, vertical_start[idx]:vertical_end[idx], pos_y - size_half:pos_y + size_half + 1] = zero

        for idx, pos_x in enumerate(mask_width_indices):
            size = np.random.randint(min_size, max_size + 1)
            size_half = size // 2
            mask[:, pos_x - size_half: pos_x + size_half + 1, horizontal_start[idx]:horizontal_end[idx]] = zero

    return mask


class MaskLines2D():
    """Masks random lines from an image. The same pixels are masked on each channel.

    Parameters
    ----------
    p : float, optional
        The probability of masking a line at each row/column, default: 0.05.
    max_height : float, optional
        Maximum height of vertical lines as a proportion of image height, default: 1.0.
    max_width : float, optional
        Maximum width of horizontal lines as a proportion of image width, default: 1.0.
    min_height : float, optional
        Minimum height of vertical lines as a proportion of image height, default: 0.1.
    min_width : float, optional
        Minimum width of horizontal lines as a proportion of image width, default: 0.1.
    max_size : int, optional
        Maximum thickness of lines in pixels, default: 3.
    min_size : int, optional
        Minimum thickness of lines in pixels, default: 1.
    channel_last : bool, optional
        Whether the image is (channels, height, width) or (height, width, channels), default: True.
    """
    def __init__(
        self, 
        p: float = 0.05, 
        max_height: float = 1.0,
        max_width: float = 1.0,
        min_height: float = 0.1,
        min_width: float = 0.1,
        max_size: int = 3,
        min_size: int = 1,
        channel_last: bool = True
    ):
        self.p = p
        self.max_height = max_height
        self.max_width = max_width
        self.min_height = min_height
        self.min_width = min_width
        self.max_size = max_size
        self.min_size = min_size
        self.channel_last = channel_last

    def __call__(self, arr: np.ndarray) -> np.ndarray:
        return mask_lines_2d(
            arr, 
            p=self.p,
            max_height=self.max_height,
            max_width=self.max_width,
            min_height=self.min_height,
            min_width=self.min_width,
            max_size=self.max_size,
            min_size=self.min_size,
            channel_last=self.channel_last
        )


@jit(nopython=True, nogil=True, cache=True, fastmath=True, parallel=True)
def mask_lines_3d(
    arr: np.ndarray,
    p: float = 0.05,
    channel_last: bool = True,
):
    """
    Creates mask with random horizontal and vertical lines for each channel independently.
    
    Parameters
    ----------
    arr : np.ndarray
        The image to create a line mask for.
    p : float, optional
        The probability of creating a line at each row/column, default: 0.05.
    channel_last : bool, optional
        Whether the image is in (height, width, channels) or (channels, height, width) format.
        
    Returns
    -------
    np.ndarray
        A mask where 0 indicates masked pixels and 1 indicates unmasked pixels.
    """
    height, width, channels = arr.shape

    if not channel_last:
        channels, height, width = arr.shape

    mask = np.ones(arr.shape, dtype=np.uint8)
    zero = np.array(0, dtype=np.uint8)

    mask_height = (np.random.uniform(0.0, 1.0, size=(height, channels, )) < p).astype(np.uint8)
    mask_width = (np.random.uniform(0.0, 1.0, size=(width, channels, )) < p).astype(np.uint8)

    if channel_last:
        for col in prange(height):
            for row in prange(width):
                for channel in prange(channels):
                    if mask_height[col, channel]:
                        mask[col, row, channel] = zero
                    if mask_width[row, channel]:
                        mask[col, row, channel] = zero

    else:
        for col in prange(height):
            for row in prange(width):
                for channel in prange(channels):
                    if mask_height[col, channel]:
                        mask[channel, col, row] = zero
                    if mask_width[row, channel]:
                        mask[channel, col, row] = zero

    return mask


class MaskLines3D():
    """Masks random lines from an image independently for each channel.

    Parameters
    ----------
    p : float, optional
        The probability of masking a line at each row/column, default: 0.05.
    channel_last : bool, optional
        Whether the image is (channels, height, width) or (height, width, channels), default: True.
    """
    def __init__(self, p: float = 0.05, channel_last: bool = True):
        self.p = p
        self.channel_last = channel_last

    def __call__(self, arr: np.ndarray) -> np.ndarray:
        return mask_lines_3d(arr, p=self.p, channel_last=self.channel_last)


@jit(nopython=True, nogil=True, cache=True, fastmath=True)
def _round_converter(idx, height, width):
    """
    Convert a 1D index to a 2D coordinate on the border of an image.
    
    Parameters
    ----------
    idx : int
        1D index to convert
    height : int
        Height of the image
    width : int
        Width of the image
        
    Returns
    -------
    Tuple[int, int]
        (y, x) coordinates
    """
    # Top side
    if idx < width:
        return 0, idx

    # Right side
    elif idx < width + height - 1:
        return idx - width + 1, width - 1

    # Bottom side
    elif idx < 2 * width + height - 2:
        return height - 1, width - 1 - (idx - width - height + 2)

    # Left side
    else:
        return height - 1 - (idx - 2 * width - height + 3), 0


@jit(nopython=True, nogil=True, cache=True, fastmath=True)
def _quadratic_bezier(px0, py0, px1, py1, px2, py2, t):
    """
    Calculate point on a quadratic Bezier curve using individual components.
    
    Parameters
    ----------
    px0, py0 : float
        First control point
    px1, py1 : float
        Second control point
    px2, py2 : float
        Third control point
    t : float
        Parameter value (0-1)
        
    Returns
    -------
    Tuple[int, int]
        (x, y) coordinates of the point on the curve
    """
    a = (1-t)
    b = t
    x = a*(a*px0 + b*px1) + b*(a*px1 + b*px2)
    y = a*(a*py0 + b*py1) + b*(a*py1 + b*py2)
    
    return int(x + 0.5), int(y + 0.5)  # Using +0.5 and int for rounding


@jit(nopython=True, nogil=True, cache=True, fastmath=True, parallel=True)
def mask_lines_2d_bezier(
    arr: np.ndarray,
    p: float = 0.05,
    channel_last: bool = True,
) -> np.ndarray:
    """
    Creates mask with random Bezier curve lines. The same lines are drawn on each channel.
    
    Parameters
    ----------
    arr : np.ndarray
        The image to create a curve mask for.
    p : float, optional
        The probability of creating a curve for each border point, default: 0.05.
    channel_last : bool, optional
        Whether the image is in (height, width, channels) or (channels, height, width) format.
        
    Returns
    -------
    np.ndarray
        A mask where 0 indicates masked pixels and 1 indicates unmasked pixels.
    """
    if channel_last:
        height, width, channels = arr.shape
    else:
        channels, height, width = arr.shape

    mask = np.ones((height, width, channels), dtype=np.uint8)

    border_pixels = (2 * (height + width)) - 4
    lines_count = (np.random.rand(border_pixels) < p).sum()

    if lines_count == 0:
        return mask

    _sy = np.zeros(lines_count, dtype=np.int64)
    _sx = np.zeros(lines_count, dtype=np.int64)
    _ey = np.zeros(lines_count, dtype=np.int64)
    _ex = np.zeros(lines_count, dtype=np.int64)
    _my = np.zeros(lines_count, dtype=np.int64)
    _mx = np.zeros(lines_count, dtype=np.int64)

    for i in prange(lines_count):
        start_idx = np.random.randint(0, border_pixels)
        end_idx = np.random.randint(0, border_pixels)

        _sy[i], _sx[i] = _round_converter(start_idx, height, width)
        _ey[i], _ex[i] = _round_converter(end_idx, height, width)
        _my[i] = np.random.randint(0, height)
        _mx[i] = np.random.randint(0, width)

    for i in range(lines_count):
        # 1. Calculate the Bounding Box for Each Curve:
        min_x = min(_sx[i], _mx[i], _ex[i])
        max_x = max(_sx[i], _mx[i], _ex[i])
        min_y = min(_sy[i], _my[i], _ey[i])
        max_y = max(_sy[i], _my[i], _ey[i])
        
        # 2. Determine the `linspace` Resolution for Each Curve:
        bounding_box_diagonal = int(np.sqrt((max_x - min_x)**2 + (max_y - min_y)**2))
        space = np.linspace(0.0, 1.0, bounding_box_diagonal)
    
        for t_idx in prange(space.shape[0]):
            t = space[t_idx]
            
            y, x = _quadratic_bezier(_sx[i], _sy[i], _mx[i], _my[i], _ex[i], _ey[i], t)

            if channel_last:
                mask[y, x, :] = 0
            else:
                mask[:, y, x] = 0

    return mask


class MaskLines2DBezier():
    """Masks random bezier lines from an image. The same lines are masked on each channel.

    Parameters
    ----------
    p : float, optional
        The probability of masking a line for each border point, default: 0.05.
    channel_last : bool, optional
        Whether the image is (channels, height, width) or (height, width, channels), default: True.
    """
    def __init__(self, p: float = 0.05, channel_last: bool = True):
        self.p = p
        self.channel_last = channel_last

    def __call__(self, arr: np.ndarray) -> np.ndarray:
        return mask_lines_2d_bezier(arr, p=self.p, channel_last=self.channel_last)
