"""
This module contains functions for augmenting images that are
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
    """
    Matches the noise of an array to the noise of another array.
    """
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
    """
    Matches the noise of an array to the noise of another array.
    """
    noise = np.random.uniform(val_min, val_max, size=arr.shape).astype(np.float32)

    return noise


@jit(nopython=True, nogil=True, cache=True, fastmath=True, parallel=True)
def _get_matched_noise_3d_binary(
    arr: np.ndarray,
    val_min: float = 0.0,
    val_max: float = 1.0,
    channel_last: bool = True,
) -> np.ndarray:
    """
    Matches the noise of an array to the noise of another array.
    """
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
    """
    Blurs an image at random. Input should be (height, width, channels) or (channels, height, width).
    
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
    """
    Masks random pixels from an image. The same pixels are masked on each channel.

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
    """
    Masks random pixels from an image.

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
    channel_last: bool = True,
):
    height, width, channels = arr.shape
    mask_shape = (height, width, 1)

    if not channel_last:
        channels, height, width = arr.shape
        mask_shape = (1, height, width)

    mask = np.ones(mask_shape, dtype=np.uint8)
    zero = np.array(0, dtype=np.uint8)

    mask_height = (np.random.uniform(0.0, 1.0, size=(height, )) < p).astype(np.uint8)
    mask_width = (np.random.uniform(0.0, 1.0, size=(width, )) < p).astype(np.uint8)

    if channel_last:
        for col in prange(height):
            for row in prange(width):
                if mask_height[col]:
                    mask[col, row, :] = zero
                if mask_width[row]:
                    mask[col, row, :] = zero

    else:
        for col in prange(height):
            for row in prange(width):
                if mask_height[col]:
                    mask[:, col, row] = zero
                if mask_width[row]:
                    mask[:, col, row] = zero

    mask = mask.repeat(channels).reshape(arr.shape)

    return mask


class MaskLines2D():
    """
    Masks random lines from an image. The same pixels are masked on each channel.

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
    """
    Masks random lines from an image.

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

@jit(nopython=True, nogil=True, cache=True, fastmath=True, parallel=True)
def mask_rectangle_2d(
    arr: np.ndarray,
    p: float = 0.05,
    max_height: int = -1,
    max_width: int = -1,
    min_height: int = 10,
    min_width: int = 10,
    channel_last: bool = True,
) -> np.ndarray:
    
    height, width, _channels = arr.shape

    if not channel_last:
        _channels, height, width = arr.shape

    mask = np.ones(arr.shape, dtype=np.uint8)
    zero = np.array(0, dtype=np.uint8)

    if np.random.uniform(0.0, 1.0) < p:
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
    """
    Masks a random rectangle from an image. The same pixels are masked on each channel.

    Parameters
    ----------
    arr : np.ndarray
        The image to mask a pixel from.

    p : float, optional
        The probability of a rectangle being masked, default: 0.05.

    max_height : int, optional
        The maximum height of the rectangle, default: -1.
        -1 means that half the image height is used.

    max_width : int, optional
        The maximum width of the rectangle, default: -1.
        -1 means that half the image width is used.

    min_height : int, optional
        The minimum height of the rectangle, default: 10.

    min_width : int, optional
        The minimum width of the rectangle, default: 10.

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
        max_height: int = -1,
        max_width: int = -1,
        min_height: int = 10,
        min_width: int = 10,
        channel_last: bool = True,
    ):
        self.p = p
        self.max_height = max_height
        self.max_width = max_width
        self.min_height = min_height
        self.min_width = min_width
        self.channel_last = channel_last

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
    max_height: int = -1,
    max_width: int = -1,
    min_height: int = 10,
    min_width: int = 10,
    channel_last: bool = True,
) -> np.ndarray:
    height, width, channels = arr.shape

    if not channel_last:
        channels, height, width = arr.shape

    mask = np.ones(arr.shape, dtype=np.uint8)
    zero = np.array(0, dtype=np.uint8)

    for channel in prange(channels):
        if np.random.uniform(0.0, 1.0) > p:
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
            mask[col:col + mask_height, row:row + mask_width, channel] = zero
        else:
            mask[channel, col:col + mask_height, row:row + mask_width] = zero

    return mask


class MaskRectangle3D():
    """
    Masks a random rectangle from an image.

    Parameters
    ----------
    arr : np.ndarray
        The image to mask a pixel from.

    p : float, optional
        The probability of a rectangle being masked, default: 0.05.

    max_height : int, optional
        The maximum height of the rectangle, default: -1.
        -1 means that half the image height is used.

    max_width : int, optional
        The maximum width of the rectangle, default: -1.
        -1 means that half the image width is used.

    min_height : int, optional
        The minimum height of the rectangle, default: 10.

    min_width : int, optional
        The minimum width of the rectangle, default: 10.

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
        max_height: int = -1,
        max_width: int = -1,
        min_height: int = 10,
        min_width: int = 10,
        channel_last: bool = True,
    ):
        self.p = p
        self.max_height = max_height
        self.max_width = max_width
        self.min_height = min_height
        self.min_width = min_width
        self.channel_last = channel_last

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
    """
    Masks random channels from an image.

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
    """
    Replaces pixels in an array with values using a mask and a method.
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
    """
    Replaces pixels in an array with values using a mask and a method.
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
