""" ### Noise generation functions for masking. ###"""

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
