""" Noise augmentation functions. """

# Standard library
import random
from typing import Optional, Tuple

# External
import numpy as np
from numba import jit, prange


@jit(nopython=True, nogil=True, cache=True, fastmath=True)
def augmentation_noise_uniform(
    X: np.ndarray,
    max_amount: float = 0.1,
    additive: bool = False,
    per_channel: bool = True,
    channel_last: bool = True,
    inplace: bool = False,
) -> np.ndarray:
    """Adds random noise seperately to each pixel of the image. The noise works
    for both channel first and last images. Follows a uniform distribution.
    Input should be (height, width, channels) or (channels, height, width).

    NOTE: Beware of datatypes. Consider casting to float32 before adding noise.

    Parameters
    ----------
    X : np.ndarray
        The image to add noise to.

    max_amount : float, optional
        The maximum amount of noise to add, sampled uniformly, default: 0.1.

    additive : bool, optional
        Whether to add or multiply the noise, default: False.

    per_channel : bool, optional
        Whether to add the same noise to each channel or different noise to each channel (per_channel), default: True.

    channel_last : bool, optional
        Whether the image is (channels, height, width) or (height, width, channels), ignored for this function.
        Kept to keep the same function signature as other augmentations, default: True.

    inplace : bool, optional
        Whether to perform the rotation in-place, default: False.

    Returns
    -------
    np.ndarray
        The image with uniform noise.
    """
    if not inplace:
        X = X.copy()

    amount = np.random.rand() * max_amount

    if per_channel:
        if additive:
            random_noise = np.random.uniform(-amount, amount, size=X.shape).astype(X.dtype)
            X[:] += random_noise
        else:
            random_noise = np.random.uniform(1 - amount, 1 + amount, size=X.shape).astype(X.dtype)
            X[:] *= random_noise

    else:
        if channel_last: # hwc
            if additive:
                random_noise = np.random.uniform(-amount, amount, size=X.shape[:2]).astype(X.dtype)
                for i in range(X.shape[2]):
                    X[:, :, i] += random_noise
            else:
                random_noise = np.random.uniform(1 - amount, 1 + amount, size=X.shape[:2]).astype(X.dtype)
                for i in range(X.shape[2]):
                    X[:, :, i] *= random_noise
        else:
            if additive:
                random_noise = np.random.uniform(-amount, amount, size=X.shape[1:]).astype(X.dtype)
                for i in range(X.shape[0]):
                    X[i, :, :] += random_noise
            else:
                random_noise = np.random.uniform(1 - amount, 1 + amount, size=X.shape[1:]).astype(X.dtype)
                for i in range(X.shape[0]):
                    X[i, :, :] *= random_noise

    return X


class AugmentationNoiseUniform:
    def __init__(self, *, p: float = 1.0, max_amount: float = 0.1, additive: bool = False, per_channel: bool = True, channel_last: bool = True, inplace: bool = False):
        self.p = p
        self.max_amount = max_amount
        self.additive = additive
        self.per_channel = per_channel
        self.channel_last = channel_last
        self.inplace = inplace
        self.applies_to_features = True
        self.applies_to_labels = False
        self.requires_dataset = False

    def __call__(self, X: np.ndarray) -> np.ndarray:
        if random.random() > self.p:
            return X

        return augmentation_noise_uniform(
            X,
            max_amount=self.max_amount,
            additive=self.additive,
            per_channel=self.per_channel,
            channel_last=self.channel_last,
            inplace=self.inplace,
        )


@jit(nopython=True, nogil=True, cache=True, fastmath=True)
def augmentation_noise_normal(
    X: np.ndarray,
    max_amount: float = 0.1,
    additive: bool = False,
    per_channel: bool = True,
    channel_last: bool = True,
    inplace: bool = False,
) -> np.ndarray:
    """Adds random noise seperately to each pixel of the image. The noise works
    for both channel first and last images. Follows a normal distribution.
    max_amount is the standard deviation of the normal distribution.
    Input should be (height, width, channels) or (channels, height, width).

    NOTE: Beware of datatypes. Consider casting to float32 before adding noise.

    Parameters
    ----------
    X : np.ndarray
        The image to add noise to.

    max_amount : float, optional
        The maximum amount of noise to add, sampled uniformly, default: 0.1.

    additive : bool, optional
        Whether to add or multiply the noise, default: False.

    per_channel : bool, optional
        Whether to add the same noise to each channel or different noise to each channel (per_channel), default: True.

    channel_last : bool, optional
        Whether the image is (channels, height, width) or (height, width, channels), ignored for this function.
        Kept to keep the same function signature as other augmentations, default: True.

    inplace : bool, optional
        Whether to perform the rotation in-place, default: False.

    Returns
    -------
    np.ndarray
        The image with normal noise.
    """
    if not inplace:
        X = X.copy()

    amount = np.random.rand() * max_amount

    if per_channel:
        if additive:
            random_noise = np.random.normal(0, amount, size=X.shape).astype(X.dtype)
            X[:] += random_noise
        else:
            random_noise = np.random.normal(1, amount, size=X.shape).astype(X.dtype)
            X[:] *= random_noise

    else:
        if channel_last: # hwc
            if additive:
                random_noise = np.random.normal(0, amount, size=X.shape[:2]).astype(X.dtype)
                for i in range(X.shape[2]):
                    X[:, :, i] += random_noise
            else:
                random_noise = np.random.normal(1, amount, size=X.shape[:2]).astype(X.dtype)
                for i in range(X.shape[2]):
                    X[:, :, i] *= random_noise
        else:
            if additive:
                random_noise = np.random.normal(0, amount, size=X.shape[1:]).astype(X.dtype)
                for i in range(X.shape[0]):
                    X[i, :, :] += random_noise
            else:
                random_noise = np.random.normal(1, amount, size=X.shape[1:]).astype(X.dtype)
                for i in range(X.shape[0]):
                    X[i, :, :] *= random_noise

    return X


class AugmentationNoiseNormal:
    def __init__(self, *, p: float = 1.0, max_amount: float = 0.1, additive: bool = False, per_channel: bool = True, channel_last: bool = True, inplace: bool = False):
        self.p = p
        self.max_amount = max_amount
        self.additive = additive
        self.per_channel = per_channel
        self.channel_last = channel_last
        self.inplace = inplace
        self.applies_to_features = True
        self.applies_to_labels = False
        self.requires_dataset = False

    def __call__(self, X: np.ndarray) -> np.ndarray:
        if random.random() > self.p:
            return X

        return augmentation_noise_normal(
            X,
            max_amount=self.max_amount,
            additive=self.additive,
            per_channel=self.per_channel,
            channel_last=self.channel_last,
            inplace=self.inplace,
        )
