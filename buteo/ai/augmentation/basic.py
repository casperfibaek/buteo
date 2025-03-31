""" Basic augmentation functions like rotation and mirroring. """

# Standard library
import random
from typing import Optional, Tuple

# External
import numpy as np

# Internal - utilities for array operations
from buteo.array.convolution import convolve_array_simple


def _rotate_arr(arr: np.ndarray, k: int, channel_last: bool = True) -> np.ndarray:
    """Rotates an array by 90 degrees k times.

    Parameters
    ----------
    arr : np.ndarray
        The array to rotate.
    k : int
        The number of 90 degree rotations to apply (1=90°, 2=180°, 3=270°).
    channel_last : bool, optional
        Whether the channels are in the last dimension, by default True.

    Returns
    -------
    np.ndarray
        The rotated array.
    """
    if channel_last:
        axes = (0, 1)
    else:
        axes = (1, 2)

    # Using np.rot90 with the appropriate axes
    return np.rot90(arr, k=k, axes=axes)


def _mirror_arr(arr: np.ndarray, k: int, channel_last: bool = True) -> np.ndarray:
    """Mirrors an array along specified axes.

    Parameters
    ----------
    arr : np.ndarray
        The array to mirror.
    k : int
        1: flip horizontally, 2: flip vertically, 3: flip both.
    channel_last : bool, optional
        Whether the channels are in the last dimension, by default True.

    Returns
    -------
    np.ndarray
        The mirrored array.
    """
    if channel_last:
        if k == 1:  # horizontal flip
            return np.flip(arr, axis=1)
        if k == 2:  # vertical flip
            return np.flip(arr, axis=0)
        if k == 3:  # both
            return np.flip(np.flip(arr, axis=0), axis=1)
    else:
        if k == 1:  # horizontal flip 
            return np.flip(arr, axis=2)
        if k == 2:  # vertical flip
            return np.flip(arr, axis=1)
        if k == 3:  # both
            return np.flip(np.flip(arr, axis=1), axis=2)
    
    return arr


def augmentation_rotation(
    X: np.ndarray,
    k: int = -1,
    channel_last: bool = True,
    inplace = False,
) -> np.ndarray:
    """Randomly rotate the image by 90 degrees intervals. Images
    can be (channels, height, width) or (height, width, channels).

    Parameters
    ----------
    X : np.ndarray
        The image to rotate.

    k : int, optional
        The number of 90 degree intervals to rotate by, default: -1 (random).

    channel_last : bool, optional
        Whether the image is (channels, height, width) or (height, width, channels), default: True.

    inplace : bool, optional
        Whether to perform the rotation in-place, default: False.

    Returns
    -------
    np.ndarray
        The rotated image.
    """
    if not inplace:
        X = X.copy()

    if k == -1:
        random_k = random.choice([1, 2, 3])
    elif k in [1, 2, 3]:
        random_k = k
    else:
        raise ValueError("k must be -1 or 1, 2, 3")

    X[:] = _rotate_arr(X, random_k, channel_last)

    return X


class AugmentationRotation:
    def __init__(self, *, p: float = 1.0, k: int = -1, channel_last: bool = True, inplace: bool = False):
        self.p = p
        self.k = k
        self.channel_last = channel_last
        self.inplace = inplace
        self.applies_to_features = True
        self.applies_to_labels = False
        self.requires_dataset = False

    def __call__(self, X: np.ndarray) -> np.ndarray:
        if random.random() > self.p:
            return X

        return augmentation_rotation(
            X,
            k=self.k,
            channel_last=self.channel_last,
            inplace=self.inplace,
        )


def augmentation_rotation_xy(
    X: np.ndarray,
    y: np.ndarray,
    k: int = -1,
    channel_last: bool = True,
    inplace: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Randomly rotate the image and label by 90 degrees intervals. Images
    can be (channels, height, width) or (height, width, channels).

    Parameters
    ----------
    X : np.ndarray
        The image to rotate.

    y : np.ndarray
        The label to rotate.

    k : int, optional
        The number of 90 degree intervals to rotate by, default: -1 (random).

    channel_last : bool, optional
        Whether the image is (channels, height, width) or (height, width, channels), default: True.

    inplace : bool, optional
        Whether to perform the rotation in-place, default: False.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The rotated image and optionally the label.
    """
    if not inplace:
        X = X.copy()
        y = y.copy()

    if k == -1:
        random_k = random.choice([1, 2, 3])
    else:
        random_k = k

    X[:] = _rotate_arr(X, random_k, channel_last)
    y[:] = _rotate_arr(y, random_k, channel_last)

    return X, y


class AugmentationRotationXY:
    def __init__(self, *, p: float = 1.0, k: int = -1, channel_last: bool = True, inplace: bool = False):
        self.p = p
        self.k = k
        self.channel_last = channel_last
        self.inplace = inplace
        self.applies_to_features = True
        self.applies_to_labels = True
        self.requires_dataset = False

    def __call__(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() > self.p:
            return X, y

        return augmentation_rotation_xy(
            X,
            y,
            k=self.k,
            channel_last=self.channel_last,
            inplace=self.inplace,
        )


def augmentation_mirror(
    X: np.ndarray,
    k: int = -1,
    channel_last: bool = True,
    inplace: bool = False,
) -> np.ndarray:
    """Randomly mirrors the image.
    Images can be (channels, height, width) or (height, width, channels).

    Parameters
    ----------
    X : np.ndarray
        The image to mirror.

    k : int, optional
        If -1, randomly mirrors the image along the horizontal or vertical axis.
        1. mirrors the image along the horizontal axis.
        2. mirrors the image along the vertical axis.
        3. mirrors the image along both the horizontal and vertical axis, default: None.

    channel_last : bool, optional
        Whether the image is (channels, height, width) or (height, width, channels), default: True.

    inplace : bool, optional
        Whether to perform the rotation in-place, default: False.

    Returns
    -------
    np.ndarray
        The mirrored image.
    """
    if not inplace:
        X = X.copy()

    if k == -1:
        random_k = random.choice([1, 2, 3])
    else:
        random_k = k

    X[:] = _mirror_arr(X, random_k, channel_last)

    return X


class AugmentationMirror:
    def __init__(self, *, p: float = 1.0, k: int = -1, channel_last: bool = True, inplace: bool = False):
        self.p = p
        self.k = k
        self.channel_last = channel_last
        self.inplace = inplace
        self.applies_to_features = True
        self.applies_to_labels = False
        self.requires_dataset = False

    def __call__(self, X: np.ndarray) -> np.ndarray:
        if random.random() > self.p:
            return X

        return augmentation_mirror(
            X,
            k=self.k,
            channel_last=self.channel_last,
            inplace=self.inplace,
        )


def augmentation_mirror_xy(
    X: np.ndarray,
    y: np.ndarray,
    k: int = -1,
    channel_last: bool = True,
    inplace: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Randomly mirrors the image. Images can be (channels, height, width) or (height, width, channels).

    Parameters
    ----------
    X : np.ndarray
        The image to mirror.

    y : np.ndarray
        The label to mirror.

    k : int, optional
        If -1, randomly mirrors the image along the horizontal or vertical axis.
        1. mirrors the image along the horizontal axis.
        2. mirrors the image along the vertical axis.
        3. mirrors the image along both the horizontal and vertical axis, default: None.

    channel_last : bool, optional
        Whether the image is (channels, height, width) or (height, width, channels), default: True.

    inplace : bool, optional
        Whether to perform the rotation in-place, default: False.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The mirrored image and optionally the label.
    """
    if not inplace:
        X = X.copy()
        y = y.copy()

    if k == -1:
        random_k = random.choice([1, 2, 3])
    else:
        random_k = k

    X[:] = _mirror_arr(X, random_k, channel_last)
    y[:] = _mirror_arr(y, random_k, channel_last)

    return X, y


class AugmentationMirrorXY:
    def __init__(self, *, p: float = 1.0, k: int = -1, channel_last: bool = True, inplace: bool = False):
        self.p = p
        self.k = k
        self.channel_last = channel_last
        self.inplace = inplace
        self.applies_to_features = True
        self.applies_to_labels = True
        self.requires_dataset = False

    def __call__(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() > self.p:
            return X, y

        return augmentation_mirror_xy(
            X,
            y,
            k=self.k,
            channel_last=self.channel_last,
            inplace=self.inplace,
        )
