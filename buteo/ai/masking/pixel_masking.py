""" ### Pixel-level masking functions for images. ###"""

# External
import numpy as np
from numba import jit, prange

# Set epsilon for float comparisons
EPSILON = 1e-7


@jit(nopython=True, nogil=True, cache=True, fastmath=True)
def mask_pixels_2d(
    arr: np.ndarray,
    p: float = 0.05,
    channel_last: bool = True,
) -> np.ndarray:
    """
    Masks random pixels from an image. The same pixels are masked on each channel.
    
    Parameters
    ----------
    arr : np.ndarray
        The image to mask pixels from.
    p : float, optional
        The probability of masking a pixel, default: 0.05.
    channel_last : bool, optional
        Whether the image is in (height, width, channels) or (channels, height, width) format.
        
    Returns
    -------
    np.ndarray
        A mask where 0 indicates masked pixels and 1 indicates unmasked pixels.
    """
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
    p : float, optional
        The probability of masking a pixel, default: 0.05.
    channel_last : bool, optional
        Whether the image is (channels, height, width) or (height, width, channels), default: True.
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
    """
    Masks random pixels from an image independently for each channel.
    
    Parameters
    ----------
    arr : np.ndarray
        The image to mask pixels from.
    p : float, optional
        The probability of masking a pixel, default: 0.05.
    channel_last : bool, optional
        Whether the image is in (height, width, channels) or (channels, height, width) format.
        
    Returns
    -------
    np.ndarray
        A mask where 0 indicates masked pixels and 1 indicates unmasked pixels.
    """
    mask = (np.random.uniform(0.0, 1.0, size=arr.shape) > p).astype(np.uint8)
    return mask


class MaskPixels3D():
    """Masks random pixels from an image independently for each channel.

    Parameters
    ----------
    p : float, optional
        The probability of masking a pixel, default: 0.05.
    channel_last : bool, optional
        Whether the image is (channels, height, width) or (height, width, channels), default: True.
    """
    def __init__(self, p: float = 0.05, channel_last: bool = True):
        self.p = p
        self.channel_last = channel_last

    def __call__(self, arr: np.ndarray) -> np.ndarray:
        return mask_pixels_3d(arr, p=self.p, channel_last=self.channel_last)


@jit(nopython=True, nogil=True, cache=True, fastmath=True)
def mask_channels(
    arr: np.ndarray,
    p: float = 0.05,
    channel_last: bool = True,
    max_channels: int = 1,
) -> np.ndarray:
    """
    Masks random channels from an image.
    
    Parameters
    ----------
    arr : np.ndarray
        The image to mask channels from.
    p : float, optional
        The probability of masking a channel, default: 0.05.
    channel_last : bool, optional
        Whether the image is in (height, width, channels) or (channels, height, width) format.
    max_channels : int, optional
        The maximum number of channels to mask, default: 1.
        
    Returns
    -------
    np.ndarray
        A mask where 0 indicates masked pixels and 1 indicates unmasked pixels.
    """
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
    p : float, optional
        The probability of masking a channel, default: 0.05.
    max_channels : int, optional
        The maximum number of channels to mask, default: 1.
    channel_last : bool, optional
        Whether the image is (channels, height, width) or (height, width, channels), default: True.
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
