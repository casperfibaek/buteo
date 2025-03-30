""" ### Shape-based masking functions for images. ###"""

# External
import numpy as np
from numba import jit, prange

# Define epsilon for float comparisons
EPSILON = 1e-7


@jit(nopython=True, nogil=True, cache=True, fastmath=True)
def _point_within_elipse(
    centroid: tuple,
    a: float,
    b: float,
    theta: float,
    point: np.ndarray,
) -> bool:
    """
    Check if a point is within an ellipse.
    
    Parameters
    ----------
    centroid : tuple
        The center point (x0, y0) of the ellipse
    a : float
        Semi-major axis length
    b : float
        Semi-minor axis length
    theta : float
        Rotation angle of the ellipse in radians
    point : np.ndarray
        The (x, y) point to check
        
    Returns
    -------
    bool
        True if the point is within the ellipse, False otherwise
    """
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
    epsilon: float = 1e-7,
) -> np.ndarray:
    """
    Creates a mask with a random ellipse. The same ellipse is applied to all channels.
    
    Parameters
    ----------
    arr : np.ndarray
        The image to create an ellipse mask for
    p : float, optional
        The probability of creating an ellipse, default: 0.05
    max_height : float, optional
        Maximum height of ellipse as a proportion of image height, default: 0.4
    max_width : float, optional
        Maximum width of ellipse as a proportion of image width, default: 0.4
    min_height : float, optional
        Minimum height of ellipse as a proportion of image height, default: 0.1
    min_width : float, optional
        Minimum width of ellipse as a proportion of image width, default: 0.1
    channel_last : bool, optional
        Whether the image is in (height, width, channels) or (channels, height, width) format
    epsilon : float, optional
        Small value to avoid floating point issues, default: 1e-7
        
    Returns
    -------
    np.ndarray
        A mask where 0 indicates masked pixels and 1 indicates unmasked pixels
    """
    height, width, channels = arr.shape
    mask_shape = (height, width, channels)

    if not channel_last:
        channels, height, width = arr.shape
        mask_shape = (channels, height, width)

    mask = np.ones(mask_shape, dtype=np.uint8)
    zero = np.array(0, dtype=np.uint8)

    if np.random.uniform(0.0, 1.0 + epsilon) > p:
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
    """Masks a random ellipse within the image. The same ellipse is applied to all channels.

    Parameters
    ----------
    p : float, optional
        The probability of masking with an ellipse, default: 0.05
    max_height : float, optional
        Maximum height of ellipse as a proportion of image height, default: 0.4
    max_width : float, optional
        Maximum width of ellipse as a proportion of image width, default: 0.4
    min_height : float, optional
        Minimum height of ellipse as a proportion of image height, default: 0.1
    min_width : float, optional
        Minimum width of ellipse as a proportion of image width, default: 0.1
    channel_last : bool, optional
        Whether the image is (channels, height, width) or (height, width, channels), default: True
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
    epsilon: float = 1e-7,
) -> np.ndarray:
    """
    Creates a mask with random ellipses, independently for each channel.
    
    Parameters
    ----------
    arr : np.ndarray
        The image to create ellipse masks for
    p : float, optional
        The probability of creating an ellipse per channel, default: 0.05
    max_height : float, optional
        Maximum height of ellipse as a proportion of image height, default: 0.4
    max_width : float, optional
        Maximum width of ellipse as a proportion of image width, default: 0.4
    min_height : float, optional
        Minimum height of ellipse as a proportion of image height, default: 0.1
    min_width : float, optional
        Minimum width of ellipse as a proportion of image width, default: 0.1
    channel_last : bool, optional
        Whether the image is in (height, width, channels) or (channels, height, width) format
    epsilon : float, optional
        Small value to avoid floating point issues, default: 1e-7
        
    Returns
    -------
    np.ndarray
        A mask where 0 indicates masked pixels and 1 indicates unmasked pixels
    """
    height, width, channels = arr.shape
    mask_shape = (height, width, channels)

    if not channel_last:
        channels, height, width = arr.shape
        mask_shape = (channels, height, width)

    mask = np.ones(mask_shape, dtype=np.uint8)
    zero = np.array(0, dtype=np.uint8)

    for c in range(channels):
        if np.random.uniform(0.0, 1.0 + epsilon) > p:
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
    """Masks random ellipses independently for each channel.

    Parameters
    ----------
    p : float, optional
        The probability of masking with an ellipse per channel, default: 0.05
    max_height : float, optional
        Maximum height of ellipse as a proportion of image height, default: 0.4
    max_width : float, optional
        Maximum width of ellipse as a proportion of image width, default: 0.4
    min_height : float, optional
        Minimum height of ellipse as a proportion of image height, default: 0.1
    min_width : float, optional
        Minimum width of ellipse as a proportion of image width, default: 0.1
    channel_last : bool, optional
        Whether the image is (channels, height, width) or (height, width, channels), default: True
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
    epsilon: float = 1e-7,
) -> np.ndarray:
    """
    Creates a mask with a random rectangle. The same rectangle is applied to all channels.
    
    Parameters
    ----------
    arr : np.ndarray
        The image to create a rectangle mask for
    p : float, optional
        The probability of creating a rectangle, default: 0.05
    max_height : float, optional
        Maximum height of rectangle as a proportion of image height, default: 0.5
    max_width : float, optional
        Maximum width of rectangle as a proportion of image width, default: 0.5
    min_height : float, optional
        Minimum height of rectangle as a proportion of image height, default: 0.1
    min_width : float, optional
        Minimum width of rectangle as a proportion of image width, default: 0.1
    channel_last : bool, optional
        Whether the image is in (height, width, channels) or (channels, height, width) format
    epsilon : float, optional
        Small value to avoid floating point issues, default: 1e-7
        
    Returns
    -------
    np.ndarray
        A mask where 0 indicates masked pixels and 1 indicates unmasked pixels
    """
    if channel_last:
        height, width, channels = arr.shape
        mask_shape = (height, width, channels)
    else:
        channels, height, width = arr.shape
        mask_shape = (channels, height, width)

    # Create mask filled with ones (not masked)
    mask = np.ones(mask_shape, dtype=np.uint8)
    zero = np.array(0, dtype=np.uint8)

    # Skip mask creation with probability (1-p)
    if np.random.uniform(0.0, 1.0 + epsilon) > p:
        return mask

    # Calculate rectangle size constraints
    max_h = int(max_height * height)
    max_w = int(max_width * width)
    min_h = max(1, int(min_height * height))
    min_w = max(1, int(min_width * width))

    # Determine the rectangle dimensions
    mask_height = np.random.randint(min_h, max_h + 1)
    mask_width = np.random.randint(min_w, max_w + 1)

    # Determine the rectangle position (top-left corner)
    col = np.random.randint(0, height - mask_height + 1)
    row = np.random.randint(0, width - mask_width + 1)

    # Apply the mask (set masked area to 0)
    if channel_last:
        mask[col:col + mask_height, row:row + mask_width, :] = zero
    else:
        mask[:, col:col + mask_height, row:row + mask_width] = zero

    return mask


class MaskRectangle2D():
    """Masks a random rectangle within the image. The same rectangle is applied to all channels.

    Parameters
    ----------
    p : float, optional
        The probability of masking with a rectangle, default: 0.05
    max_height : float, optional
        Maximum height of rectangle as a proportion of image height, default: 0.5
    max_width : float, optional
        Maximum width of rectangle as a proportion of image width, default: 0.5
    min_height : float, optional
        Minimum height of rectangle as a proportion of image height, default: 0.1
    min_width : float, optional
        Minimum width of rectangle as a proportion of image width, default: 0.1
    channel_last : bool, optional
        Whether the image is (channels, height, width) or (height, width, channels), default: True
    """
    def __init__(self,
        p: float = 0.05,
        channel_last: bool = True,
        max_height: float = 0.5,
        max_width: float = 0.5,
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
    epsilon: float = 1e-7,
) -> np.ndarray:
    """
    Creates a mask with random rectangles, independently for each channel.
    
    Parameters
    ----------
    arr : np.ndarray
        The image to create rectangle masks for
    p : float, optional
        The probability of creating a rectangle per channel, default: 0.05
    max_height : float, optional
        Maximum height of rectangle as a proportion of image height, default: 0.5
    max_width : float, optional
        Maximum width of rectangle as a proportion of image width, default: 0.5
    min_height : float, optional
        Minimum height of rectangle as a proportion of image height, default: 0.1
    min_width : float, optional
        Minimum width of rectangle as a proportion of image width, default: 0.1
    channel_last : bool, optional
        Whether the image is in (height, width, channels) or (channels, height, width) format
    epsilon : float, optional
        Small value to avoid floating point issues, default: 1e-7
        
    Returns
    -------
    np.ndarray
        A mask where 0 indicates masked pixels and 1 indicates unmasked pixels
    """
    if channel_last:
        height, width, channels = arr.shape
        mask_shape = (height, width, channels)
    else:
        channels, height, width = arr.shape
        mask_shape = (channels, height, width)

    # Create mask filled with ones (not masked)
    mask = np.ones(mask_shape, dtype=np.uint8)
    zero = np.array(0, dtype=np.uint8)

    # Calculate rectangle size constraints
    max_h = int(max_height * height)
    max_w = int(max_width * width)
    min_h = max(1, int(min_height * height))
    min_w = max(1, int(min_width * width))

    # For each channel, apply a random rectangle with probability p
    for c in range(channels):
        if np.random.uniform(0.0, 1.0 + epsilon) > p:
            continue

        # Determine the rectangle dimensions
        mask_height = np.random.randint(min_h, max_h + 1)
        mask_width = np.random.randint(min_w, max_w + 1)

        # Determine the rectangle position (top-left corner)
        col = np.random.randint(0, height - mask_height + 1)
        row = np.random.randint(0, width - mask_width + 1)

        # Apply the mask for this channel
        if channel_last:
            mask[col:col + mask_height, row:row + mask_width, c] = zero
        else:
            mask[c, col:col + mask_height, row:row + mask_width] = zero

    return mask


class MaskRectangle3D():
    """Masks random rectangles independently for each channel.

    Parameters
    ----------
    p : float, optional
        The probability of masking with a rectangle per channel, default: 0.05
    max_height : float, optional
        Maximum height of rectangle as a proportion of image height, default: 0.5
    max_width : float, optional
        Maximum width of rectangle as a proportion of image width, default: 0.5
    min_height : float, optional
        Minimum height of rectangle as a proportion of image height, default: 0.1
    min_width : float, optional
        Minimum width of rectangle as a proportion of image width, default: 0.1
    channel_last : bool, optional
        Whether the image is (channels, height, width) or (height, width, channels), default: True
    """
    def __init__(self,
        p: float = 0.05,
        max_height: float = 0.5,
        max_width: float = 0.5,
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
        return mask_rectangle_3d(
            arr,
            p=self.p,
            max_height=self.max_height,
            max_width=self.max_width,
            min_height=self.min_height,
            min_width=self.min_width,
            channel_last=self.channel_last,
        )
