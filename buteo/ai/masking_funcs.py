"""This module contains functions for augmenting images that are
suited to remote sensing imagery.
"""
# Standard library
import sys; sys.path.append("../../")

# External
import numpy as np
from numba import jit, prange

# Internal
from buteo.array.convolution import convolve_array_simple
from buteo.array.convolution_kernels import _simple_blur_kernel_2d_3x3

EPSILON = 1e-7


@jit(nopython=True, nogil=True, cache=True, fastmath=True)
def _get_matched_noise_2d(
    arr: np.ndarray,
    val_min: float = 0.0,
    val_max: float = 1.0,
    channel_last: bool = True,
) -> np.ndarray:
    """
    Matches the noise of an array to the noise of another array.
    """
    size = (1, arr.shape[0], arr.shape[1])
    if channel_last:
        size = (arr.shape[0], arr.shape[1], 1)

    noise = np.random.uniform(val_min, val_max, size=size).astype(np.float32)

    return noise


@jit(nopython=True, nogil=True, cache=True, fastmath=True)
def _get_matched_noise_2d_binary(
    arr: np.ndarray,
    val_min: float = 0.0,
    val_max: float = 1.0,
    channel_last: bool = True,
) -> np.ndarray:
    """Matches the noise of an array to the noise of another array."""
    size = (1, arr.shape[0], arr.shape[1])
    if channel_last:
        size = (arr.shape[0], arr.shape[1], 1)

    binary_noise = np.random.uniform(0.0, 1.0, size=size) > 0.5
    noise = arr.copy()

    _val_min = np.array(val_min, dtype=np.float32)
    _val_max = np.array(val_max, dtype=np.float32)

    if channel_last:
        for col in prange(arr.shape[0]):
            for row in prange(arr.shape[1]):
                if binary_noise[col, row, 0]:
                    noise[col, row, :] = _val_max
                else:
                    noise[col, row, :] = _val_min
    else:
        for col in prange(arr.shape[1]):
            for row in prange(arr.shape[2]):
                if binary_noise[0, col, row]:
                    noise[:, col, row] = _val_max
                else:
                    noise[:, col, row] = _val_min

    return noise


@jit(nopython=True, nogil=True, cache=True, fastmath=True)
def _get_matched_noise_3d(
    arr: np.ndarray,
    val_min: float = 0.0,
    val_max: float = 1.0,
    channel_last: bool = True,
) -> np.ndarray:
    """Matches the noise of an array to the noise of another array."""
    noise = np.random.uniform(val_min, val_max, size=arr.shape).astype(np.float32)

    return noise


@jit(nopython=True, nogil=True, cache=True, fastmath=True, parallel=True)
def _get_matched_noise_3d_binary(
    arr: np.ndarray,
    val_min: float = 0.0,
    val_max: float = 1.0,
    channel_last: bool = True,
) -> np.ndarray:
    """Matches the noise of an array to the noise of another array."""
    binary_noise = np.random.uniform(0.0, 1.0, size=arr.shape) > 0.5
    noise = arr.copy()

    _val_min = np.array(val_min, dtype=np.float32)
    _val_max = np.array(val_max, dtype=np.float32)

    if channel_last:
        for col in prange(arr.shape[0]):
            for row in prange(arr.shape[1]):
                for channel in prange(arr.shape[2]):
                    if binary_noise[col, row, channel]:
                        noise[col, row, channel] = _val_max
                    else:
                        noise[col, row, channel] = _val_min
    else:
        for channel in prange(arr.shape[0]):
            for col in prange(arr.shape[1]):
                for row in prange(arr.shape[2]):
                    if binary_noise[channel, col, row]:
                        noise[channel, col, row] = _val_max
                    else:
                        noise[channel, col, row] = _val_min

    return noise


@jit(nopython=True, nogil=True, cache=True, fastmath=True, parallel=True)
def _get_blurred_image(
    X: np.ndarray,
    channel_last: bool = True,
) -> np.ndarray:
    """Blurs an image at random. Input should be (height, width, channels) or (channels, height, width).

    NOTE: Beware of datatypes. Consider casting to float32 before adding noise.

    Parameters
    ----------
    X : np.ndarray
        The image to blur.

    channel_to_adjust : int, optional
        Weather to only apply the blur to a specific channel.

    channel_last : bool, optional
        Whether the image is (channels, height, width) or (height, width, channels), default: True.

    inplace : bool, optional
        Whether to perform the rotation in-place, default: False.

    Returns
    -------
    np.ndarray
        The blurred image.
    """
    X = X.copy()

    offsets, weights = _simple_blur_kernel_2d_3x3()

    if channel_last:
        for channel in prange(X.shape[2]):
            X[:, :, channel] = convolve_array_simple(
                X[:, :, channel],
                offsets,
                weights
            )
    else:
        for channel in prange(X.shape[0]):
            X[channel, :, :] = convolve_array_simple(
                X[channel, :, :],
                offsets,
                weights,
            )

    return X


@jit(nopython=True, nogil=True, cache=True, fastmath=True)
def mask_pixels_2d(
    arr: np.ndarray,
    p: float = 0.05,
    channel_last: bool = True,
) -> np.ndarray:
    height, width, channels = arr.shape
    mask_shape = (height, width, 1)

    if not channel_last:
        channels, height, width = arr.shape
        mask_shape = (1, height, width)

    mask = (np.random.uniform(0.0, 1.0, size=mask_shape) > p).astype(np.uint8)
    mask = mask.repeat(channels).reshape(arr.shape)

    return mask


class MaskPixels2D():
    """Masks random pixels from an image. The same pixels are masked on each channel.

    Parameters
    ----------
    arr : np.ndarray
        The image to mask a pixel from.

    p : float, optional
        The probability of masking a pixel, default: 0.05.

    channel_last : bool, optional
        Whether the image is (channels, height, width) or (height, width, channels), default: True.

    Returns
    -------
    np.ndarray
        The image with masked pixels.
    """
    def __init__(self, p: float = 0.05, channel_last: bool = True):
        self.p = p
        self.channel_last = channel_last

    def __call__(self, arr: np.ndarray) -> np.ndarray:
        return mask_pixels_2d(arr, p=self.p, channel_last=self.channel_last)

@jit(nopython=True, nogil=True, cache=True, fastmath=True)
def mask_pixels_3d(
    arr: np.ndarray,
    p: float = 0.05,
    channel_last: bool = True,
) -> np.ndarray:
    mask = (np.random.uniform(0.0, 1.0, size=arr.shape) > p).astype(np.uint8)

    return mask


class MaskPixels3D():
    """Masks random pixels from an image.

    Parameters
    ----------
    arr : np.ndarray
        The image to mask a pixel from.

    p : float, optional
        The probability of masking a pixel, default: 0.05.

    Returns
    -------
    np.ndarray
        The image with masked pixels.
    """
    def __init__(self, p: float = 0.05, channel_last: bool = True):
        self.p = p
        self.channel_last = channel_last

    def __call__(self, arr: np.ndarray) -> np.ndarray:
        return mask_pixels_3d(arr, p=self.p, channel_last=self.channel_last)

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
            mask[:, vertical_start[idx]:vertical_end[idx], pos_y + size] = zero

        for idx, pos_x in enumerate(mask_width_indices):
            mask[:, pos_x + size, horizontal_start[idx]:horizontal_end[idx]] = zero

    return mask


class MaskLines2D():
    """Masks random lines from an image. The same pixels are masked on each channel.

    Parameters
    ----------
    arr : np.ndarray
        The image to mask a pixel from.

    p : float, optional
        The probability of masking a pixel, default: 0.05.

    channel_last : bool, optional
        Whether the image is (channels, height, width) or (height, width, channels), default: True.

    Returns
    -------
    np.ndarray
        The image with masked pixels.
    """
    def __init__(self, p: float = 0.05, channel_last: bool = True):
        self.p = p
        self.channel_last = channel_last

    def __call__(self, arr: np.ndarray) -> np.ndarray:
        return mask_lines_2d(arr, p=self.p, channel_last=self.channel_last)


@jit(nopython=True, nogil=True, cache=True, fastmath=True, parallel=True)
def mask_lines_3d(
    arr: np.ndarray,
    p: float = 0.05,
    channel_last: bool = True,
):

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
    """Masks random lines from an image.

    Parameters
    ----------
    arr : np.ndarray
        The image to mask a pixel from.

    p : float, optional
        The probability of masking a pixel, default: 0.05.

    channel_last : bool, optional
        Whether the image is (channels, height, width) or (height, width, channels), default: True.

    Returns
    -------
    np.ndarray
        The image with masked pixels.
    """
    def __init__(self, p: float = 0.05, channel_last: bool = True):
        self.p = p
        self.channel_last = channel_last

    def __call__(self, arr: np.ndarray) -> np.ndarray:
        return mask_lines_3d(arr, p=self.p, channel_last=self.channel_last)


@jit(nopython=True, nogil=True, cache=True, fastmath=True)
def _round_converter(idx, height, width):
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
    """Calculate point on a quadratic Bezier curve using individual components."""
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
    """Masks random bezier lines from an image.

    Parameters
    ----------
    arr : np.ndarray
        The image to mask a pixel from.

    p : float, optional
        The probability of masking a pixel, default: 0.05.

    channel_last : bool, optional
        Whether the image is (channels, height, width) or (height, width, channels), default: True.

    Returns
    -------
    np.ndarray
        The image with masked pixels.
    """
    def __init__(self, p: float = 0.05, channel_last: bool = True):
        self.p = p
        self.channel_last = channel_last

    def __call__(self, arr: np.ndarray) -> np.ndarray:
        return mask_lines_2d_bezier(arr, p=self.p, channel_last=self.channel_last)


@jit(nopython=True, nogil=True, cache=True, fastmath=True, parallel=True)
def mask_lines_3d_bezier(
    arr: np.ndarray,
    p: float = 0.05,
    channel_last: bool = True,
) -> np.ndarray:
    if channel_last:
        height, width, channels = arr.shape
    else:
        channels, height, width = arr.shape

    mask = np.ones((height, width, channels), dtype=np.uint8)

    border_pixels = (2 * (height + width)) - 4

    for c in prange(channels):
        lines_count = (np.random.rand(border_pixels) < p).sum()

        if lines_count == 0:
            continue

        _sy = np.zeros(lines_count, dtype=np.int64)
        _sx = np.zeros(lines_count, dtype=np.int64)
        _ey = np.zeros(lines_count, dtype=np.int64)
        _ex = np.zeros(lines_count, dtype=np.int64)
        _my = np.zeros(lines_count, dtype=np.int64)
        _mx = np.zeros(lines_count, dtype=np.int64)

        for i in range(lines_count):
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
                    mask[y, x, c] = 0
                else:
                    mask[c, y, x] = 0

    return mask


class MaskLines3DBezier():
    """Masks random bezier lines from an image.

    Parameters
    ----------
    arr : np.ndarray
        The image to mask a pixel from.

    p : float, optional
        The probability of masking a pixel, default: 0.05.

    channel_last : bool, optional
        Whether the image is (channels, height, width) or (height, width, channels), default: True.

    Returns
    -------
    np.ndarray
        The image with masked pixels.
    """
    def __init__(self, p: float = 0.05, channel_last: bool = True):
        self.p = p
        self.channel_last = channel_last

    def __call__(self, arr: np.ndarray) -> np.ndarray:
        return mask_lines_3d_bezier(arr, p=self.p, channel_last=self.channel_last)


@jit(nopython=True, nogil=True, cache=True, fastmath=True)
def _point_within_elipse(
    centroid: np.ndarray,
    a: float,
    b: float,
    theta: float,
    point: np.ndarray,
) -> bool:
    x, y = point
    x0, y0 = centroid

    return ((x - x0) * np.cos(theta) + (y - y0) * np.sin(theta)) ** 2 / a ** 2 + ((x - x0) * np.sin(theta) - (y - y0) * np.cos(theta)) ** 2 / b ** 2 <= 1


@jit(nopython=True, nogil=True, cache=True, fastmath=True, parallel=True)
def mask_elipse_2d(
    arr: np.ndarray,
    p: float = 0.05,
    max_height: float = 0.4,
    max_width: float = 0.4,
    min_height: float = 0.1,
    min_width: float = 0.1,
    channel_last: bool = True,
):
    height, width, channels = arr.shape
    mask_shape = (height, width, channels)

    if not channel_last:
        channels, height, width = arr.shape
        mask_shape = (channels, height, width)

    mask = np.ones(mask_shape, dtype=np.uint8)
    zero = np.array(0, dtype=np.uint8)

    if np.random.uniform(0.0, 1.0 + EPSILON) > p:
        return mask

    # Randomly select the centroid of the ellipse
    centroid = (np.random.uniform(0, height), np.random.uniform(0, width))
    
    # Randomly select the height and width of the ellipse
    # Ensure that the ellipse fits within the rectangle
    min_x = int(width * min_width)
    min_y = int(height * min_height)
    max_x = int(width * max_width)
    max_y = int(height * max_height)
    a = np.random.uniform(min_y, max_y)  # semi-major axis
    b = np.random.uniform(min_x, max_x)  # semi-minor axis
    
    # Randomly select the orientation of the ellipse
    theta = np.random.uniform(0, 2 * np.pi)

    # Calculate lazy bounds
    max_side = np.ceil(np.maximum(a, b))
    max_y = centroid[0] + max_side
    min_y = centroid[0] - max_side
    max_x = centroid[1] + max_side
    min_x = centroid[1] - max_side

    for col in prange(height):
        if col < min_y or col > max_y:
            continue
        for row in prange(width):
            if row < min_x or row > max_x:
                continue

            if _point_within_elipse(centroid, a, b, theta, np.array([col, row], dtype=np.float32)):
                if channel_last:
                    mask[col, row, :] = zero
                else:
                    mask[:, col, row] = zero

    return mask


class MaskElipse2D():
    """Masks a random elipse within the image.

    Parameters
    ----------
    arr : np.ndarray
        The image to mask a pixel from.

    p : float, optional
        The probability of masking a pixel, default: 0.05.

    channel_last : bool, optional
        Whether the image is (channels, height, width) or (height, width, channels), default: True.

    Returns
    -------
    np.ndarray
        The image with masked pixels.
    """
    def __init__(self,
        p: float = 0.05,
        channel_last: bool = True,
        max_height: float = 0.4,
        max_width: float = 0.4,
        min_height: float = 0.1,
        min_width: float = 0.1,
    ):
        self.p = p
        self.channel_last = channel_last
        self.max_height = max_height
        self.max_width = max_width
        self.min_height = min_height
        self.min_width = min_width

        assert self.max_height >= self.min_height, "max_height must be greater than or equal to min_height"
        assert self.max_width >= self.min_width, "max_width must be greater than or equal to min_width"
        assert p >= 0.0 and p <= 1.0, "p must be between 0.0 and 1.0"
        assert max_height >= 0.0 and max_height <= 1.0, "max_height must be between 0.0 and 1.0"
        assert max_width >= 0.0 and max_width <= 1.0, "max_width must be between 0.0 and 1.0"
        assert min_height >= 0.0 and min_height <= 1.0, "min_height must be between 0.0 and 1.0"
        assert min_width >= 0.0 and min_width <= 1.0, "min_width must be between 0.0 and 1.0"

    def __call__(self, arr: np.ndarray) -> np.ndarray:
        return mask_elipse_2d(
            arr,
            p=self.p,
            max_height=self.max_height,
            max_width=self.max_width,
            min_height=self.min_height,
            min_width=self.min_width,
            channel_last=self.channel_last,
        )


@jit(nopython=True, nogil=True, cache=True, fastmath=True, parallel=True)
def mask_elipse_3d(
    arr: np.ndarray,
    p: float = 0.05,
    max_height: float = 0.4,
    max_width: float = 0.4,
    min_height: float = 0.1,
    min_width: float = 0.1,
    channel_last: bool = True,
):
    height, width, channels = arr.shape
    mask_shape = (height, width, channels)

    if not channel_last:
        channels, height, width = arr.shape
        mask_shape = (channels, height, width)

    mask = np.ones(mask_shape, dtype=np.uint8)
    zero = np.array(0, dtype=np.uint8)

    for c in range(channels):
        if np.random.uniform(0.0, 1.0 + EPSILON) > p:
            continue

        # Randomly select the centroid of the ellipse
        centroid = (np.random.uniform(0, height), np.random.uniform(0, width))
        
        # Randomly select the height and width of the ellipse
        # Ensure that the ellipse fits within the rectangle
        min_x = int(width * min_width)
        min_y = int(height * min_height)
        max_x = int(width * max_width)
        max_y = int(height * max_height)
        a = np.random.uniform(min_y, max_y)  # semi-major axis
        b = np.random.uniform(min_x, max_x)  # semi-minor axis
        
        # Randomly select the orientation of the ellipse
        theta = np.random.uniform(0, 2 * np.pi)

        # Calculate lazy bounds
        max_side = np.ceil(np.maximum(a, b))
        max_y = centroid[0] + max_side
        min_y = centroid[0] - max_side
        max_x = centroid[1] + max_side
        min_x = centroid[1] - max_side

        for col in prange(height):
            if col < min_y or col > max_y:
                continue
            for row in prange(width):
                if row < min_x or row > max_x:
                    continue

                if _point_within_elipse(centroid, a, b, theta, np.array([col, row], dtype=np.float32)):
                    if channel_last:
                        mask[col, row, c] = zero
                    else:
                        mask[c, col, row] = zero

    return mask


class MaskElipse3D():
    """Masks a random elipse within the image.

    Parameters
    ----------
    arr : np.ndarray
        The image to mask a pixel from.

    p : float, optional
        The probability of masking a pixel, default: 0.05.

    channel_last : bool, optional
        Whether the image is (channels, height, width) or (height, width, channels), default: True.

    Returns
    -------
    np.ndarray
        The image with masked pixels.
    """
    def __init__(self,
        p: float = 0.05,
        max_height: float = 0.4,
        max_width: float = 0.4,
        min_height: float = 0.1,
        min_width: float = 0.1,
        channel_last: bool = True,
    ):
        self.p = p
        self.channel_last = channel_last
        self.max_height = max_height
        self.max_width = max_width
        self.min_height = min_height
        self.min_width = min_width

        assert self.max_height >= self.min_height, "max_height must be greater than or equal to min_height"
        assert self.max_width >= self.min_width, "max_width must be greater than or equal to min_width"
        assert p >= 0.0 and p <= 1.0, "p must be between 0.0 and 1.0"
        assert max_height >= 0.0 and max_height <= 1.0, "max_height must be between 0.0 and 1.0"
        assert max_width >= 0.0 and max_width <= 1.0, "max_width must be between 0.0 and 1.0"
        assert min_height >= 0.0 and min_height <= 1.0, "min_height must be between 0.0 and 1.0"
        assert min_width >= 0.0 and min_width <= 1.0, "min_width must be between 0.0 and 1.0"

    def __call__(self, arr: np.ndarray) -> np.ndarray:
        return mask_elipse_3d(
            arr,
            p=self.p,
            channel_last=self.channel_last,
            max_height=self.max_height,
            max_width=self.max_width,
            min_height=self.min_height,
            min_width=self.min_width,
        )


@jit(nopython=True, nogil=True, cache=True, fastmath=True, parallel=True)
def mask_rectangle_2d(
    arr: np.ndarray,
    p: float = 0.05,
    max_height: float = 0.5,
    max_width: float = 0.5,
    min_height: float = 0.1,
    min_width: float = 0.1,
    channel_last: bool = True,
) -> np.ndarray:
    
    height, width, _channels = arr.shape

    if not channel_last:
        _channels, height, width = arr.shape

    max_height = int(max_height * height)
    max_width = int(max_width * width)
    min_height = int(min_height * height)
    min_width = int(min_width * width)

    mask = np.ones(arr.shape, dtype=np.uint8)
    zero = np.array(0, dtype=np.uint8)

    if np.random.uniform(0.0, 1.0 + EPSILON) > p:
        return mask

    if max_height == -1:
        mask_height = np.random.randint(min_height, (height // 2) + 1)
    else:
        mask_height = np.random.randint(min_height, max_height + 1)
    
    if max_width == -1:
        mask_width = np.random.randint(min_width, (width // 2) + 1)
    else:
        mask_width = np.random.randint(min_width, max_width + 1)

    col = np.random.randint(0, height - mask_height)
    row = np.random.randint(0, width - mask_width)

    if channel_last:
        mask[col:col + mask_height, row:row + mask_width, :] = zero
    else:
        mask[:, col:col + mask_height, row:row + mask_width] = zero

    return mask


class MaskRectangle2D():
    """Masks a random rectangle from an image. The same pixels are masked on each channel.

    Parameters
    ----------
    arr : np.ndarray
        The image to mask a pixel from.

    p : float, optional
        The probability of a rectangle being masked, default: 0.05.

    max_height : int, optional
        The maximum height (proportion of total_height) of the rectangle, default: 0.5.

    max_width : int, optional
        The maximum width (proportion of total_width) of the rectangle, default: 0.5.

    min_height : int, optional
        The minimum height of the rectangle (proportion of total_height), default: 0.1

    min_width : int, optional
        The minimum width of the rectangle (proportion of total_width), default: 0.1

    channel_last : bool, optional
        Whether the image is (channels, height, width) or (height, width, channels), default: True.

    Returns
    -------
    np.ndarray
        The image with masked pixels.
    """
    def __init__(
        self,
        p: float = 0.05,
        max_height: float = 0.5,
        max_width: float = 0.5,
        min_height: float = 0.1,
        min_width: float = 0.1,
        channel_last: bool = True,
    ):
        self.p = p
        self.max_height = max_height
        self.max_width = max_width
        self.min_height = min_height
        self.min_width = min_width
        self.channel_last = channel_last

        assert self.max_height >= self.min_height, "max_height must be greater than or equal to min_height"
        assert self.max_width >= self.min_width, "max_width must be greater than or equal to min_width"
        assert p >= 0.0 and p <= 1.0, "p must be between 0.0 and 1.0"
        assert max_height >= 0.0 and max_height <= 1.0, "max_height must be between 0.0 and 1.0"
        assert max_width >= 0.0 and max_width <= 1.0, "max_width must be between 0.0 and 1.0"
        assert min_height >= 0.0 and min_height <= 1.0, "min_height must be between 0.0 and 1.0"
        assert min_width >= 0.0 and min_width <= 1.0, "min_width must be between 0.0 and 1.0"

    def __call__(self, arr: np.ndarray) -> np.ndarray:
        return mask_rectangle_2d(
            arr,
            p=self.p,
            max_height=self.max_height,
            max_width=self.max_width,
            min_height=self.min_height,
            min_width=self.min_width,
            channel_last=self.channel_last,
        )
    

@jit(nopython=True, nogil=True, cache=True, fastmath=True, parallel=True)
def mask_rectangle_3d(
    arr: np.ndarray,
    p: float = 0.05,
    max_height: float = 0.5,
    max_width: float = 0.5,
    min_height: float = 0.1,
    min_width: float = 0.1,
    channel_last: bool = True,
) -> np.ndarray:
    height, width, channels = arr.shape

    if not channel_last:
        channels, height, width = arr.shape

    mask = np.ones(arr.shape, dtype=np.uint8)
    zero = np.array(0, dtype=np.uint8)

    max_height = int(max_height * height)
    max_width = int(max_width * width)
    min_height = int(min_height * height)
    min_width = int(min_width * width)

    for c in prange(channels):
        if np.random.uniform(0.0, 1.0 + EPSILON) > p:
            continue

        if max_height == -1:
            mask_height = np.random.randint(min_height, (height // 2) + 1)
        else:
            mask_height = np.random.randint(min_height, max_height + 1)
        
        if max_width == -1:
            mask_width = np.random.randint(min_width, (width // 2) + 1)
        else:
            mask_width = np.random.randint(min_width, max_width + 1)

        col = np.random.randint(0, height - mask_height)
        row = np.random.randint(0, width - mask_width)

        if channel_last:
            mask[col:col + mask_height, row:row + mask_width, c] = zero
        else:
            mask[c, col:col + mask_height, row:row + mask_width] = zero

    return mask


class MaskRectangle3D():
    """Masks a random rectangle from an image.

    Parameters
    ----------
    arr : np.ndarray
        The image to mask a pixel from.

    p : float, optional
        The probability of a rectangle being masked, default: 0.05.

    max_height : int, optional
        The maximum height (proportion of total_height) of the rectangle, default: 0.5.

    max_width : int, optional
        The maximum width (proportion of total_width) of the rectangle, default: 0.5.

    min_height : int, optional
        The minimum height of the rectangle (proportion of total_height), default: 0.1

    min_width : int, optional
        The minimum width of the rectangle (proportion of total_width), default: 0.1

    channel_last : bool, optional
        Whether the image is (channels, height, width) or (height, width, channels), default: True.

    Returns
    -------
    np.ndarray
        The image with masked pixels.
    """
    def __init__(
        self,
        p: float = 0.05,
        max_height: float = 0.5,
        max_width: float = 0.5,
        min_height: float = 0.1,
        min_width: float = 0.1,
        channel_last: bool = True,
    ):
        self.p = p
        self.max_height = max_height
        self.max_width = max_width
        self.min_height = min_height
        self.min_width = min_width
        self.channel_last = channel_last

        assert self.max_height >= self.min_height, "max_height must be greater than or equal to min_height"
        assert self.max_width >= self.min_width, "max_width must be greater than or equal to min_width"
        assert p >= 0.0 and p <= 1.0, "p must be between 0.0 and 1.0"
        assert max_height >= 0.0 and max_height <= 1.0, "max_height must be between 0.0 and 1.0"
        assert max_width >= 0.0 and max_width <= 1.0, "max_width must be between 0.0 and 1.0"
        assert min_height >= 0.0 and min_height <= 1.0, "min_height must be between 0.0 and 1.0"
        assert min_width >= 0.0 and min_width <= 1.0, "min_width must be between 0.0 and 1.0"

    def __call__(self, arr: np.ndarray) -> np.ndarray:
        return mask_rectangle_3d(
            arr,
            p=self.p,
            max_height=self.max_height,
            max_width=self.max_width,
            min_height=self.min_height,
            min_width=self.min_width,
            channel_last=self.channel_last,
        )


@jit(nopython=True, nogil=True, cache=True, fastmath=True, parallel=True)
def mask_channels(
    arr: np.ndarray,
    p: float = 0.05,
    channel_last: bool = True,
    max_channels: int = 1,
) -> np.ndarray:
    _height, _width, channels = arr.shape

    if not channel_last:
        channels, _height, _width = arr.shape

    mask = np.ones(arr.shape, dtype=np.uint8)
    zero = np.array(0, dtype=np.uint8)

    for _ in prange(max_channels):
        if np.random.uniform(0.0, 1.0) < p:
            random_channel = np.random.randint(0, channels)

            if channel_last:
                mask[:, :, random_channel] = zero
            else:
                mask[random_channel, :, :] = zero

    return mask


class MaskChannels():
    """Masks random channels from an image.

    Parameters
    ----------
    arr : np.ndarray
        The image to mask a pixel from.

    p : float, optional
        The probability of masking a pixel, default: 0.05.

    max_channels : int, optional
        The maximum number of channels to mask, default: 1.

    channel_last : bool, optional
        Whether the image is (channels, height, width) or (height, width, channels), default: True.

    Returns
    -------
    np.ndarray
        The image with masked pixels.
    """
    def __init__(self, p: float = 0.05, max_channels: int = 1, channel_last: bool = True):
        self.p = p
        self.max_channels = max_channels
        self.channel_last = channel_last

        assert p >= 0.0 and p <= 1.0, "p must be between 0.0 and 1.0"
        if channel_last:
            assert max_channels >= 1, "max_channels must be between 1 and the number of channels in the image"

    def __call__(self, arr: np.ndarray) -> np.ndarray:
        return mask_channels(arr, p=self.p, max_channels=self.max_channels, channel_last=self.channel_last)


@jit(nopython=True, nogil=True, cache=True, fastmath=True)
def mask_replace_2d(
    arr: np.ndarray,
    mask: np.ndarray,
    method: int = 0,
    val: float = 0.0,
    min_val: float = 0.0,
    max_val: float = 1.0,
    channel_last: bool = True,
    inplace: bool = False
):
    """Replaces pixels in an array with values using a mask and a method.
    NOTE: The mask should have the same shape as the array and
    contain values either 0 for masked and 1 for valid.

    Parameters
    ----------
    arr : np.ndarray
        The array to replace pixels in.

    mask : np.ndarray
        The mask to use for replacing pixels.

    method : int, optional
        The method to use for replacing pixels.
        0. replace with 0
        1. replace with 1
        2. replace with val
        3. replace with random value between min_val and max_val
        4. replace with random value binary value, either min_val or max_val
        5. replace with a blurred version of the image

    channel_last : bool, optional
        Whether the image is (channels, height, width) or (height, width, channels), default: True.

    Returns
    -------
    np.ndarray
        The array with replaced pixels.
    """
    replace_arr_needed = False
    replace_val = 0.0
    replace_arr = None

    if inplace:
        X = arr
    else:
        X = arr.copy()

    if method == 0:
        replace_arr_needed = False
        replace_val = 0.0
    elif method == 1:
        replace_arr_needed = False
        replace_val = 1.0
    elif method == 2:
        replace_arr_needed = False
        replace_val = val
    elif method == 3:
        replace_arr_needed = True
        replace_val = 0.0
        replace_arr = _get_matched_noise_2d(X, min_val, max_val, channel_last)
    elif method == 4:
        replace_arr_needed = True
        replace_val = 0.0
        replace_arr = _get_matched_noise_2d_binary(X, min_val, max_val, channel_last)
    elif method == 5:
        replace_arr_needed = True
        replace_val = 0.0
        replace_arr = _get_blurred_image(X, channel_last)
    else:
        raise ValueError("method must be between 0 and 4.")
    
    if channel_last:
        for col in prange(X.shape[0]):
            for row in prange(X.shape[1]):
                if mask[col, row, 0] == 0:
                    if replace_arr_needed:
                        X[col, row, :] = replace_arr[col, row, :]
                    else:
                        X[col, row, :] = replace_val
    else:
        for col in prange(X.shape[1]):
            for row in prange(X.shape[2]):
                if mask[0, col, row] == 0:
                    if replace_arr_needed:
                        X[:, col, row] = replace_arr[:, col, row]
                    else:
                        X[:, col, row] = replace_val

    return X


@jit(nopython=True, nogil=True, cache=True, fastmath=True)
def mask_replace_3d(
    arr: np.ndarray,
    mask: np.ndarray,
    method: int = 0,
    val: float = 0.0,
    min_val: float = 0.0,
    max_val: float = 1.0,
    channel_last: bool = True,
    inplace: bool = False
):
    """Replaces pixels in an array with values using a mask and a method.
    NOTE: The mask should have the same shape as the array and
    contain values either 0 for masked and 1 for valid.

    Parameters
    ----------
    arr : np.ndarray
        The array to replace pixels in.

    mask : np.ndarray
        The mask to use for replacing pixels.

    method : int, optional
        The method to use for replacing pixels.
        0. replace with 0
        1. replace with 1
        2. replace with val
        3. replace with random value between min_val and max_val
        4. replace with random value binary value, either min_val or max_val
        5. replace with a blurred version of the image

    channel_last : bool, optional
        Whether the image is (channels, height, width) or (height, width, channels), default: True.

    Returns
    -------
    np.ndarray
        The array with replaced pixels.
    """
    replace_arr_needed = False
    replace_val = 0.0
    replace_arr = None

    if inplace:
        X = arr
    else:
        X = arr.copy()

    if method == 0:
        replace_arr_needed = False
        replace_val = 0.0
    elif method == 1:
        replace_arr_needed = False
        replace_val = 1.0
    elif method == 2:
        replace_arr_needed = False
        replace_val = val
    elif method == 3:
        replace_arr_needed = True
        replace_val = 0.0
        replace_arr = _get_matched_noise_3d(X, min_val, max_val, channel_last)
    elif method == 4:
        replace_arr_needed = True
        replace_val = 0.0
        replace_arr = _get_matched_noise_3d_binary(X, min_val, max_val, channel_last)
    elif method == 5:
        replace_arr_needed = True
        replace_val = 0.0
        replace_arr = _get_blurred_image(X, channel_last)
    else:
        raise ValueError("method must be between 0 and 4.")
    
    if channel_last:
        for col in prange(X.shape[0]):
            for row in prange(X.shape[1]):
                for channel in prange(X.shape[2]):
                    if mask[col, row, channel] == 0:
                        if replace_arr_needed:
                            X[col, row, channel] = replace_arr[col, row, channel]
                        else:
                            X[col, row, channel] = replace_val
    else:
        for channel in prange(X.shape[0]):
            for col in prange(X.shape[1]):
                for row in prange(X.shape[2]):
                    if mask[channel, col, row] == 0:
                        if replace_arr_needed:
                            X[channel, col, row] = replace_arr[channel, col, row]
                        else:
                            X[channel, col, row] = replace_val

    return X
