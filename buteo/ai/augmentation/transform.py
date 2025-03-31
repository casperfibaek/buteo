""" Transform augmentation functions. """

# Standard library
import random
from typing import Optional, Tuple

# External
import numpy as np
from numba import jit, prange

# Internal
from buteo.array.convolution import convolve_array_simple
from buteo.array.convolution.kernels import (
    kernel_base,
    kernel_shift,
    kernel_unsharp,
    kernel_sobel,
    kernel_get_offsets_and_weights
)


@jit(nopython=True, nogil=True, cache=True, fastmath=True, parallel=True)
def augmentation_channel_scale(
    X: np.ndarray,
    max_amount: float = 0.1,
    additive: bool = False,
    channel_last: bool = True,
    inplace: bool = False,
) -> np.ndarray:
    """Scales the channels of the image seperately by a fixed amount.
    Input should be (height, width, channels) or (channels, height, width).

    NOTE: Beware of datatypes. Consider casting to float32 before adding noise.

    Parameters
    ----------
    X : np.ndarray
        The image to scale the channels of.

    max_amount : float, optional
        The amount to possible scale the channels by. Sampled uniformly, default: 0.1.

    additive : bool, optional
        Whether to add or multiply the scaling, default: False.

    channel_last : bool, optional
        Whether the image is (channels, height, width) or (height, width, channels), default: True.

    inplace : bool, optional
        Whether to perform the rotation in-place, default: False.

    Returns
    -------
    np.ndarray
        The image with scaled channels.
    """
    if not inplace:
        X = X.copy()

    amount = np.random.rand() * max_amount

    if channel_last:
        for i in prange(X.shape[2]):
            if additive:
                random_amount = np.random.uniform(-amount, amount)
                X[:, :, i] += random_amount
            else:
                random_amount = np.random.uniform(1 - amount, 1 + amount)
                X[:, :, i] *= random_amount
    else:
        for i in prange(X.shape[0]):
            if additive:
                random_amount = np.random.uniform(-amount, amount)
                X[i, :, :] += random_amount
            else:
                random_amount = np.random.uniform(1 - amount, 1 + amount)
                X[i, :, :] *= random_amount

    return X


class AugmentationChannelScale:
    def __init__(self, *, p: float = 1.0, max_amount: float = 0.1, additive: bool = False, channel_last: bool = True, inplace: bool = False):
        self.p = p
        self.max_amount = max_amount
        self.additive = additive
        self.channel_last = channel_last
        self.inplace = inplace
        self.applies_to_features = True
        self.applies_to_labels = False
        self.requires_dataset = False

    def __call__(self, X: np.ndarray) -> np.ndarray:
        if random.random() > self.p:
            return X

        return augmentation_channel_scale(
            X,
            max_amount=self.max_amount,
            additive=self.additive,
            channel_last=self.channel_last,
            inplace=self.inplace,
        )


@jit(nopython=True, nogil=True, cache=True, fastmath=True, parallel=True)
def augmentation_contrast(
    X: np.ndarray,
    max_amount: float = 0.1,
    channel_last: bool = True,
    inplace: bool = False,
) -> np.ndarray:
    """Changes the contrast of an image by a random amount, seperately for each channel.
    Input should be (height, width, channels) or (channels, height, width).

    NOTE: Beware of datatypes. Consider casting to float32 before adding noise.

    Parameters
    ----------
    X : np.ndarray
        The image to change the contrast of.

    max_amount : float, optional
        The max amount to change the contrast by, default: 0.1.

    channel_last : bool, optional
        Whether the image is (channels, height, width) or (height, width, channels), default: True.

    inplace : bool, optional
        Whether to perform the rotation in-place, default: False.

    Returns
    -------
    np.ndarray
        The image with changed contrast.
    """
    if not inplace:
        X = X.copy()

    amount = np.random.rand() * max_amount

    channels = X.shape[2] if channel_last else X.shape[0]
    mean_pixel = np.zeros(channels, dtype=np.float32)

    # Numba workaround because numba does not support axis argument in np.mean
    if channel_last:
        for i in prange(channels):
            mean_pixel[i] = np.nanmean(X[:, :, i])
    else:
        for i in prange(channels):
            mean_pixel[i] = np.nanmean(X[i, :, :])

    for i in prange(channels):
        X[:, :, i] = (X[:, :, i] - mean_pixel[i]) * (1 + amount) + mean_pixel[i]

    return X


class AugmentationContrast:
    def __init__(self, *, p: float = 1.0, max_amount: float = 0.1, channel_last: bool = True, inplace: bool = False):
        self.p = p
        self.max_amount = max_amount
        self.channel_last = channel_last
        self.inplace = inplace
        self.applies_to_features = True
        self.applies_to_labels = False
        self.requires_dataset = False

    def __call__(self, X: np.ndarray) -> np.ndarray:
        if random.random() > self.p:
            return X

        return augmentation_contrast(
            X,
            max_amount=self.max_amount,
            channel_last=self.channel_last,
            inplace=self.inplace,
        )


@jit(nopython=True, nogil=True, cache=True, fastmath=True)
def augmentation_blur(
    X: np.ndarray,
    channel_to_adjust: int = -1,
    channel_last: bool = True,
    inplace: bool = False,
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
    if not inplace:
        X = X.copy()

    kernel = kernel_base(radius=1.0, circular=True, distance_weighted=True, method=3)
    offsets, weights = kernel_get_offsets_and_weights(kernel)

    if channel_last:
        if channel_to_adjust != -1:
            X[:, :, channel_to_adjust] = convolve_array_simple(X[:, :, channel_to_adjust], offsets, weights)
        else:
            for channel in prange(X.shape[2]):
                X[:, :, channel] = convolve_array_simple(
                    X[:, :, channel],
                    offsets,
                    weights
                )
    else:
        if channel_to_adjust != -1:
            X[channel_to_adjust, :, :] = convolve_array_simple(X[channel_to_adjust, :, :], offsets, weights)
        else:
            for channel in prange(X.shape[0]):
                X[channel, :, :] = convolve_array_simple(
                    X[channel, :, :],
                    offsets,
                    weights,
                )

    return X


class AugmentationBlur:
    def __init__(self, *, p: float = 1.0, channel_to_adjust: int = -1, channel_last: bool = True, inplace: bool = False):
        self.p = p
        self.channel_to_adjust = channel_to_adjust
        self.channel_last = channel_last
        self.inplace = inplace
        self.applies_to_features = True
        self.applies_to_labels = False
        self.requires_dataset = False

    def __call__(self, X: np.ndarray) -> np.ndarray:
        if random.random() > self.p:
            return X

        return augmentation_blur(
            X,
            channel_to_adjust=self.channel_to_adjust,
            channel_last=self.channel_last,
            inplace=self.inplace,
        )


@jit(nopython=True, nogil=True, cache=True, fastmath=True)
def augmentation_blur_xy(
    X: np.ndarray,
    y: np.ndarray,
    channel_last: bool = True,
    inplace: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Blurs an image at random. Input should be (height, width, channels) or (channels, height, width).
    The label is blurred by the same amount.

    NOTE: Beware of datatypes. Consider casting to float32 before adding noise.

    Parameters
    ----------
    X : np.ndarray
        The image to blur.

    y : np.ndarray
        The label to blur.

    channel_last : bool, optional
        Whether the image is (channels, height, width) or (height, width, channels), default: True.

    inplace : bool, optional
        Whether to perform the rotation in-place, default: False.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The blurred image and optionally the unmodified label.
    """
    x_blurred = augmentation_blur(X, -1, channel_last, inplace)
    y_blurred = augmentation_blur(y, -1, channel_last, inplace)

    return x_blurred, y_blurred


class AugmentationBlurXY:
    def __init__(self, *, p: float = 1.0, channel_last: bool = True, inplace: bool = False):
        self.p = p
        self.channel_last = channel_last
        self.inplace = inplace
        self.applies_to_features = True
        self.applies_to_labels = True
        self.requires_dataset = False

    def __call__(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() > self.p:
            return X, y

        return augmentation_blur_xy(
            X,
            y,
            channel_last=self.channel_last,
            inplace=self.inplace,
        )


@jit(nopython=True, nogil=True, cache=True, fastmath=True, parallel=True)
def augmentation_sharpen(
    X: np.ndarray,
    channel_to_adjust: int = -1,
    channel_last: bool = True,
    inplace: bool = False,
) -> np.ndarray:
    """Sharpens an image at random. Input should be (height, width, channels) or (channels, height, width).

    NOTE: Beware of datatypes. Consider casting to float32 before adding noise.

    Parameters
    ----------
    X : np.ndarray
        The image to sharpen.

    channel_to_adjust : int, default = -1
        Weather to apply the sharpen to a specific channel or all (-1).

    channel_last : bool, optional
        Whether the image is (channels, height, width) or (height, width, channels), default: True.

    inplace : bool, optional
        Whether to perform the rotation in-place, default: False.

    Returns
    -------
    np.ndarray
        The sharpened image.
    """
    if not inplace:
        X = X.copy()

    offsets, weights = kernel_unsharp()

    if channel_last:
        if channel_to_adjust == -1:
            for channel in prange(X.shape[2]):
                X[:, :, channel] = convolve_array_simple(
                    X[:, :, channel],
                    offsets,
                    weights
                )
        else:
            X[:, :, channel_to_adjust] = convolve_array_simple(
                X[:, :, channel_to_adjust],
                offsets,
                weights
            )
    else:
        if channel_to_adjust == -1:
            for channel in prange(X.shape[0]):
                X[channel, :, :] = convolve_array_simple(
                    X[channel, :, :],
                    offsets,
                    weights,
                )
        else:
            X[channel_to_adjust, :, :] = convolve_array_simple(
                X[channel_to_adjust, :, :],
                offsets,
                weights,
            )

    return X


class AugmentationSharpen:
    def __init__(self, *, p: float = 1.0, channel_to_adjust: int = -1, channel_last: bool = True, inplace: bool = False):
        self.p = p
        self.channel_to_adjust = channel_to_adjust
        self.channel_last = channel_last
        self.inplace = inplace
        self.applies_to_features = True
        self.applies_to_labels = False
        self.requires_dataset = False

    def __call__(self, X: np.ndarray) -> np.ndarray:
        if random.random() > self.p:
            return X

        return augmentation_sharpen(
            X,
            channel_to_adjust=self.channel_to_adjust,
            channel_last=self.channel_last,
            inplace=self.inplace,
        )


@jit(nopython=True, nogil=True, cache=True, fastmath=True)
def augmentation_sharpen_xy(
    X: np.ndarray,
    y: np.ndarray,
    channel_last: bool = True,
    inplace: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sharpens an image at random. Input should be (height, width, channels) or (channels, height, width).
    The label is sharpened by the same amount.

    NOTE: Beware of datatypes. Consider casting to float32 before adding noise.

    Parameters
    ----------
    X : np.ndarray
        The image to sharpen.

    y : np.ndarray
        The label to sharpen. If None, no label is returned

    channel_last : bool, optional
        Whether the image is (channels, height, width) or (height, width, channels), default: True.

    inplace : bool, optional
        Whether to perform the rotation in-place, default: False.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The sharpened image and label.
    """
    x_sharpened = augmentation_sharpen(X, -1, channel_last, inplace)
    y_sharpened = augmentation_sharpen(y, -1, channel_last, inplace)

    return x_sharpened, y_sharpened


class AugmentationSharpenXY:
    def __init__(self, *, p: float = 1.0, channel_last: bool = True, inplace: bool = False):
        self.p = p
        self.channel_last = channel_last
        self.inplace = inplace
        self.applies_to_features = True
        self.applies_to_labels = True
        self.requires_dataset = False

    def __call__(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() > self.p:
            return X, y

        return augmentation_sharpen_xy(
            X,
            y,
            channel_last=self.channel_last,
            inplace=self.inplace,
        )


@jit(nopython=True, nogil=True, cache=True, fastmath=True)
def augmentation_misalign(
    X: np.ndarray,
    max_offset: float = 0.5,
    per_channel: bool = False,
    channel_to_adjust: int = -1,
    channel_last: bool = True,
    inplace: bool = False,
) -> np.ndarray:
    """Misaligns channels in the image at random.
    input should be (height, width, channels) or (channels, height, width).

    Parameters
    ----------
    X : np.ndarray
        The image to misalign the channels of.

    max_offset : float, optional
        The maximum offset to misalign the channels by. Default: 0.5.
        Measured in percentage pixels.

    per_channel : bool, optional
        Whether to misalign each channel by a different amount. Default: False.

    channel_to_adjust: int, optional
        A specific channel to apply misalignment to. (-1 == all)

    channel_last : bool, optional
        Whether the image is (channels, height, width) or (height, width, channels). Default: True.

    Returns
    -------
    np.ndarray
        The misaligned image.
    """
    if not inplace:
        X = X.copy()

    if per_channel:
        offsets, weights = kernel_shift(
            min(np.random.rand(), max_offset),
            min(np.random.rand(), max_offset),
        )

        if channel_last:
            if channel_to_adjust == -1:
                for c in prange(X.shape[2]):
                    X[:, :, c] = convolve_array_simple(
                        X[:, :, c], offsets, weights,
                    )
            else:
                X[:, :, channel_to_adjust] = convolve_array_simple(
                    X[:, :, channel_to_adjust], offsets, weights,
                )  
        else:
            if channel_to_adjust == -1:
                for c in prange(X.shape[0]):
                    X[c, :, :] = convolve_array_simple(
                        X[c, :, :], offsets, weights,
                    )
            else:
                    X[channel_to_adjust, :, :] = convolve_array_simple(
                        X[channel_to_adjust, :, :], offsets, weights,
                    )

    else:
        offsets, weights = kernel_shift(
            min(np.random.rand(), max_offset),
            min(np.random.rand(), max_offset),
        )

        channels = X.shape[2] if channel_last else X.shape[0]
        channel_to_adjust = np.random.randint(0, channels) if channel_to_adjust == -1 else channel_to_adjust

        if channel_last:
            X[:, :, channel_to_adjust] = convolve_array_simple(
                X[:, :, channel_to_adjust], offsets, weights,
            )
        else:
            X[channel_to_adjust, :, :] = convolve_array_simple(
                X[channel_to_adjust, :, :], offsets, weights,
            )

    return X


class AugmentationMisalign:
    def __init__(self, *, p: float = 1.0, max_offset: float = 0.5, per_channel: bool = False, channel_to_adjust: int = -1, channel_last: bool = True, inplace: bool = False):
        self.p = p
        self.max_offset = max_offset
        self.per_channel = per_channel
        self.channel_to_adjust = channel_to_adjust
        self.channel_last = channel_last
        self.inplace = inplace
        self.applies_to_features = True
        self.applies_to_labels = False
        self.requires_dataset = False

    def __call__(self, X: np.ndarray) -> np.ndarray:
        if random.random() > self.p:
            return X

        return augmentation_misalign(
            X,
            max_offset=self.max_offset,
            per_channel=self.per_channel,
            channel_to_adjust=self.channel_to_adjust,
            channel_last=self.channel_last,
            inplace=self.inplace,
        )


class AugmentationMisalignLabel:
    def __init__(self, *, p: float = 1.0, max_offset: float = 0.5, per_channel: bool = False, channel_last: bool = True, inplace: bool = False):
        self.p = p
        self.max_offset = max_offset
        self.per_channel = per_channel
        self.channel_last = channel_last
        self.inplace = inplace
        self.applies_to_features = False
        self.applies_to_labels = True
        self.requires_dataset = False

    def __call__(self, y: np.ndarray) -> np.ndarray:
        if random.random() > self.p:
            return y

        return augmentation_misalign(
            y,
            max_offset=self.max_offset,
            per_channel=self.per_channel,
            channel_last=self.channel_last,
            inplace=self.inplace,
        )
