"""
This module contains functions for augmenting images that are
suited to remote sensing imagery.
"""
# Standard library
import sys; sys.path.append("../../")
from typing import Optional, Tuple, Any

# External
import numpy as np
from numba import jit, prange

# Internal
from buteo.ai.augmentation_utils import (
    fit_data_to_dtype,
    rotate_arr,
    mirror_arr,
    simple_blur_kernel_2d_3x3,
    simple_unsharp_kernel_2d_3x3,
    simple_shift_kernel_2d,
    convolution_simple,
)


@jit(nopython=True, nogil=True, cache=True, fastmath=True)
def augmentation_rotation(
    X: np.ndarray,
    y: Optional[np.ndarray],
    chance: float = 0.5,
    k: Optional[int] = None,
    channel_last: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Randomly rotate the image by 90 degrees intervals. Images
    can be (channels, height, width) or (height, width, channels).

    Args:
        X (np.ndarray): The image to rotate.

    Keyword Args:
        y (np.ndarray/none=None): The label to rotate.
        chance (float=0.5): The chance of rotating the image.
        k (int=None): The number of 90 degree intervals to rotate by.
        channel_last (bool=True): Whether the image is (channels, height, width) or (height, width, channels).

    Returns:
        Tuple[np.ndarray, Optional[np.ndarray]]: The rotated image and optionally the label.
    """
    if np.random.rand() > chance or k == 0:
        return X, y

    random_k = np.random.randint(1, 4) if k is None else k
    X_rot = rotate_arr(X, random_k, channel_last)

    if y is None:
        return X_rot, y

    y_rot = rotate_arr(y, random_k, channel_last)

    return X_rot, y_rot


@jit(nopython=True, nogil=True, cache=True, fastmath=True)
def augmentation_mirror(
    X: np.ndarray,
    y: Optional[np.ndarray],
    chance: float = 0.5,
    k: Optional[int] = None,
    channel_last: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Randomly mirrors the image. Images can be (channels, height, width) or (height, width, channels).

    Args:
        X (np.ndarray): The image to mirror.

    Keyword Args:
        y (np.ndarray/none=None): The label to mirror.
        chance (float=0.5): The chance of mirroring the image.
        k (int=None): If None, randomly mirrors the image along the horizontal or vertical axis.
            If 1, mirrors the image along the horizontal axis.
            If 2, mirrors the image along the vertical axis.
            If 3, mirrors the image along both the horizontal and vertical axis.
        channel_last (bool=True): Whether the image is (channels, height, width) or (height, width, channels).

    Returns:
        Tuple[np.ndarray, Optional[np.ndarray]]: The mirrored image and optionally the label.
    """
    if np.random.rand() > chance or k == 0:
        return X, y

    random_k = np.random.randint(1, 4) if k is None else k

    flipped_x = mirror_arr(X, random_k, channel_last)
    flipped_y = mirror_arr(y, random_k, channel_last) if y is not None else None

    return flipped_x, flipped_y


@jit(nopython=True, nogil=True, cache=True, fastmath=True)
def augmentation_noise(
    X: np.ndarray,
    y: Optional[np.ndarray],
    chance: float = 0.5,
    max_amount: float = 0.1,
    additive: bool = False,
    channel_last: Any = None, # pylint: disable=unused-argument
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Adds random noise seperately to each pixel of the image. The noise works
    for both channel first and last images.
    input should be (height, width, channels) or (channels, height, width).

    Args:
        X (np.ndarray): The image to add noise to.
    
    Keyword Args:
        y (np.ndarray/none=None): The label to add noise to. If None, no label is returned.
        chance (float=0.5): The chance of adding noise.
        max_amount (float=0.1): The maximum amount of noise to add, sampled uniformly.
        additive (bool=False): Whether to add or multiply the noise.
        channel_last (bool=True): Whether the image is (channels, height, width) or (height, width, channels).
            ignored for this function. Kept to keep the same function signature as other augmentations.

    Returns:
        Tuple[np.ndarray, Optional[np.ndarray]]: The image with noise and optionally the unmodified label.
    """
    if np.random.rand() > chance:
        return X, y

    amount = np.random.rand() * max_amount

    if additive:
        noisy_x = X + np.random.normal(0, amount, X.shape)
    else:
        noisy_x = X * np.random.normal(1, amount, X.shape)

    return fit_data_to_dtype(noisy_x, X.dtype), y


@jit(nopython=True, nogil=True, cache=True, fastmath=True, parallel=True)
def augmentation_channel_scale(
    X: np.ndarray,
    y: Optional[np.ndarray],
    chance: float = 0.5,
    max_amount: float = 0.1,
    additive: bool = False,
    channel_last: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Scales the channels of the image seperately by a fixed amount.
    input should be (height, width, channels) or (channels, height, width).

    Args:
        X (np.ndarray): The image to scale the channels of.
    
    Keyword Args:
        y (np.ndarray/none=None): The label to scale the channels of. If None, no label is returned.
        chance (float=0.5): The chance of scaling the channels.
        max_amount (float=0.1): The amount to possible scale the channels by. Sampled uniformly.
        additive (bool=False): Whether to add or multiply the scaling.
        channel_last (bool=True): Whether the image is (channels, height, width) or (height, width, channels).

    Returns:
        Tuple[np.ndarray, Optional[np.ndarray]]: The image with scaled channels and optionally the unmodified label.
    """
    if np.random.rand() > chance:
        return X, y

    x_scaled = X.astype(np.float32)
    y_scaled = y

    amount = np.random.rand() * max_amount

    if channel_last:
        for i in prange(X.shape[2]):
            if additive:
                random_amount = np.random.uniform(-amount, amount)
                x_scaled[:, :, i] += random_amount
            else:
                random_amount = np.random.uniform(1 - amount, 1 + amount)
                x_scaled[:, :, i] *= random_amount
    else:
        for i in prange(X.shape[0]):
            if additive:
                random_amount = np.random.uniform(-amount, amount)
                x_scaled[i, :, :] += random_amount
            else:
                random_amount = np.random.uniform(1 - amount, 1 + amount)
                x_scaled[i, :, :] *= random_amount

    return fit_data_to_dtype(x_scaled, X.dtype), y_scaled


@jit(nopython=True, nogil=True, cache=True, fastmath=True, parallel=True)
def augmentation_contrast(
    X: np.ndarray,
    y: Optional[np.ndarray],
    chance: float = 0.5,
    max_amount: float = 0.1,
    channel_last: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Changes the contrast of an image by a random amount, seperately for each channel.
    input should be (height, width, channels) or (channels, height, width).

    Args:
        X (np.ndarray): The image to change the contrast of.
    
    Keyword Args:
        y (np.ndarray/none=None): The label to change the contrast of. If None, no label is returned.
        chance (float=0.5): The chance of changing the contrast.
        max_amount (float=0.1): The max amount to change the contrast by.
        channel_last (bool=True): Whether the image is (channels, height, width) or (height, width, channels).

    Returns:
        Tuple[np.ndarray, Optional[np.ndarray]]: The image with changed contrast and optionally the unmodified label.
    """
    if np.random.rand() > chance:
        return X, y

    x_contrast = X.astype(np.float32)
    y_contrast = y

    amount = np.random.rand() * max_amount

    channels = X.shape[2] if channel_last else X.shape[0]
    mean_pixel = np.array([0.0] * channels, dtype=np.float32)

    # Numba workaround
    if channel_last:
        for i in prange(channels):
            mean_pixel[i] = np.mean(x_contrast[:, :, i])
    else:
        for i in prange(channels):
            mean_pixel[i] = np.mean(x_contrast[i])

    for i in prange(channels):
        x_contrast[:, :, i] = (x_contrast[:, :, i] - mean_pixel[i]) * (1 + amount) + mean_pixel[i]

    return fit_data_to_dtype(x_contrast, X.dtype), y_contrast


@jit(nopython=True, nogil=True, cache=True, fastmath=True, parallel=True)
def augmentation_drop_pixel(
    X: np.ndarray,
    y: Optional[np.ndarray],
    chance: float = 0.5,
    drop_probability: float = 0.01,
    drop_value: float = 0.0,
    channel_last: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Drops a random pixels from an image.
    input should be (height, width, channels) or (channels, height, width).
    Only drops pixels from features, not labels.

    Args:
        X (np.ndarray): The image to drop a pixel from.
    
    Keyword Args:
        y (np.ndarray/none=None): The label to drop a pixel from. If None, no label is returned.
        chance (float=0.5): The chance of dropping a pixel.
        drop_probability (float=0.05): The probability of dropping a pixel.
        drop_value (float=0.0): The value to drop the pixel to.
        apply_to_y (bool=False): Whether to apply the drop to the label.
        channel_last (bool=True): Whether the image is (channels, height, width) or (height, width, channels).

    Returns:
        Tuple[np.ndarray, Optional[np.ndarray]]: The image with the dropped pixels and optionally the unmodified label.
    """
    if np.random.rand() > chance:
        return X, y

    x_dropped = X.copy()
    y_dropped = y

    mask = np.random.random(size=x_dropped.shape)

    # Agreed. This looks terrible. But it's the only way to get numba to parallelize this.
    if channel_last:
        for col in prange(X.shape[0]):
            for row in prange(X.shape[1]):
                for channel in prange(X.shape[2]):
                    if mask[col, row, channel] <= drop_probability:
                        x_dropped[col, row, channel] = drop_value

    else:
        for channel in prange(X.shape[0]):
            for col in prange(X.shape[1]):
                for row in prange(X.shape[2]):
                    if mask[channel, col, row] <= drop_probability:
                        x_dropped[channel, col, row] = drop_value

    return x_dropped, y_dropped


@jit(nopython=True, nogil=True, cache=True, fastmath=True)
def augmentation_drop_channel(
    X: np.ndarray,
    y: Optional[np.ndarray],
    chance: float = 0.5,
    drop_probability: float = 0.1,
    drop_value: float = 0.0,
    channel_last: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Drops a random channel from an image.
    input should be (height, width, channels) or (channels, height, width).
    A maximum of one channel will be dropped.

    Args:
        X (np.ndarray): The image to drop a channel from.

    Keyword Args:
        y (np.ndarray/none=None): The label to drop a channel from. If None, no label is returned.
        chance (float=0.5): The chance of dropping a channel.
        drop_probability (float=0.1): The probability of dropping a channel.
        drop_value (float=0.0): The value to drop the channel to.
        channel_last (bool=True): Whether the image is (channels, height, width) or (height, width, channels).
    """
    if np.random.rand() > chance:
        return X, y

    x_dropped = X.copy()
    y_dropped = y

    channels = X.shape[2] if channel_last else X.shape[0]

    drop_a_channel = False
    for _ in range(channels):
        if np.random.rand() < drop_probability:
            drop_a_channel = True
            break

    if not drop_a_channel:
        return X, y

    channel_to_drop = np.random.randint(0, channels)

    if channel_last:
        x_dropped[:, :, channel_to_drop] = drop_value
    else:
        x_dropped[channel_to_drop, :, :] = drop_value

    return x_dropped, y_dropped


@jit(nopython=True, nogil=True, cache=True, fastmath=True, parallel=True)
def augmentation_blur(
    X: np.ndarray,
    y: Optional[np.ndarray],
    chance: float = 0.5,
    intensity: float = 1.0,
    apply_to_y: bool = False,
    channel_last: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Blurs an image at random.
    input should be (height, width, channels) or (channels, height, width).
    Same goes for the label if apply_to_y is True.

    Args:
        X (np.ndarray): The image to potentially blur.

    Keyword Args:
        y (np.ndarray/none=None): The label to blur a pixel in. If None, no label is returned.
        chance (float=0.5): The chance of blurring a pixel.
        intensity (float=1.0): The intensity of the blur. from 0.0 to 1.0.
        apply_to_y (bool=False): Whether to blur the label as well.
        channel_last (bool=True): Whether the image is (channels, height, width) or (height, width, channels).
    """
    if np.random.rand() > chance:
        return X, y

    x_blurred = X.astype(np.float32)
    y_blurred = y.astype(np.float32) if y is not None else None

    offsets, weights = simple_blur_kernel_2d_3x3()
    channels = X.shape[2] if channel_last else X.shape[0]

    if channel_last:
        for channel in prange(channels):
            x_blurred[:, :, channel] = convolution_simple(x_blurred[:, :, channel], offsets, weights, intensity)
            if apply_to_y and y is not None:
                y_blurred[:, :, channel] = convolution_simple(y_blurred[:, :, channel], offsets, weights, intensity)
    else:
        for channel in prange(channels):
            x_blurred[channel, :, :] = convolution_simple(x_blurred[channel, :, :], offsets, weights, intensity)
            if apply_to_y and y is not None:
                y_blurred[channel, :, :] = convolution_simple(y_blurred[channel, :, :], offsets, weights, intensity)

    x_blurred = fit_data_to_dtype(x_blurred, X.dtype)
    y_blurred = fit_data_to_dtype(y_blurred, y.dtype) if y is not None else None

    return x_blurred, y_blurred


@jit(nopython=True, nogil=True, cache=True, fastmath=True, parallel=True)
def augmentation_sharpen(
    X: np.ndarray,
    y: Optional[np.ndarray],
    chance: float = 0.5,
    intensity: float = 1.0,
    apply_to_y: bool = False,
    channel_last: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Sharpens an image at random.
    input should be (height, width, channels) or (channels, height, width).

    Args:
        X (np.ndarray): The image to potentially sharpen.

    Keyword Args:
        y (np.ndarray/none=None): The label to sharpen a pixel in. If None, no label is returned.
        chance (float=0.5): The chance of sharpening a pixel.
        intensity (float=1.0): The intensity of the sharpening. from 0.0 to 1.0.
        apply_to_y (bool=False): Whether to sharpen the label as well.
        channel_last (bool=True): Whether the image is (channels, height, width) or (height, width, channels).
    """
    if np.random.rand() > chance:
        return X, y

    x_sharpened = X.astype(np.float32)
    y_sharpened = y.astype(np.float32) if y is not None else None

    offsets, weights = simple_unsharp_kernel_2d_3x3()
    channels = X.shape[2] if channel_last else X.shape[0]

    if channel_last:
        for channel in prange(channels):
            x_sharpened[:, :, channel] = convolution_simple(x_sharpened[:, :, channel], offsets, weights, intensity)
            if apply_to_y and y is not None:
                y_sharpened[:, :, channel] = convolution_simple(y_sharpened[:, :, channel], offsets, weights, intensity)
    else:
        for channel in prange(channels):
            x_sharpened[channel, :, :] = convolution_simple(x_sharpened[channel, :, :], offsets, weights, intensity)
            if apply_to_y and y is not None:
                y_sharpened[channel, :, :] = convolution_simple(y_sharpened[channel, :, :], offsets, weights, intensity)

    x_sharpened = fit_data_to_dtype(x_sharpened, X.dtype)
    y_sharpened = fit_data_to_dtype(y_sharpened, y.dtype) if y is not None else None

    return x_sharpened, y_sharpened


def augmentation_misalign_pixels(
    X: np.ndarray,
    y: Optional[np.ndarray],
    chance: float = 0.5,
    max_offset: float = 0.5,
    channel_last: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Misaligns one channel in the image at random.
    input should be (height, width, channels) or (channels, height, width).

    Args:
        X (np.ndarray): The image to potentially misalign the channels of.

    Keyword Args:
        y (np.ndarray/none=None): The label to misalign the channels of a pixel in. If None, no label is returned.
        chance (float=0.5): The chance of misaligning the channels of a pixel.
        max_offset (float=0.5): The maximum offset to misalign the channels by.
        channel_last (bool=True): Whether the image is (channels, height, width) or (height, width, channels).
    """
    if np.random.rand() > chance:
        return X, y

    x_misaligned = X.astype(np.float32)
    y_misaligned = y.astype(np.float32) if y is not None else None

    offsets, weights = simple_shift_kernel_2d(
        min(np.random.rand(), max_offset),
        min(np.random.rand(), max_offset),
    )

    channels = X.shape[2] if channel_last else X.shape[0]
    channel_to_drop = np.random.randint(0, channels)

    if channel_last:
        x_misaligned[:, :, channel_to_drop] = convolution_simple(x_misaligned[:, :, channel_to_drop], offsets, weights)
    else:
        x_misaligned[channel_to_drop, :, :] = convolution_simple(x_misaligned[channel_to_drop, :, :], offsets, weights)

    x_misaligned = fit_data_to_dtype(x_misaligned, X.dtype)
    y_misaligned = fit_data_to_dtype(y_misaligned, y.dtype) if y is not None else None

    return x_misaligned, y_misaligned
