"""
This module contains functions for augmenting images that are
suited to remote sensing imagery.

Random rotations at 90 degrees intervals are applied to the image.
Random noise (gaussian) is added to the image.
    Channel-wise
"""
# Standard library
from typing import Optional, Tuple

# External
import numpy as np


def augmentation_rotation(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    channel_last: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Randomly rotate the image by 90 degrees intervals. Images
    can be (channels, height, width) or (height, width, channels).

    Args:
        X (np.ndarray): The image to rotate.
        y (Optional[np.ndarray], optional): The label to rotate. Defaults to None.

    Keyword Args:
        channel_last (bool=True): Whether the image is (channels, height, width) or (height, width, channels).

    Returns:
        Tuple[np.ndarray, Optional[np.ndarray]]: The rotated image and optionally the label.
    """
    random_k = np.random.randint(1, 5)

    if channel_last:
        X_rot = np.rot90(X, k=random_k, axes=(0, 1))
    else:
        X_rot = np.rot90(X, k=random_k, axes=(1, 2))

    if y is None:
        return X_rot

    if channel_last:
        y_rot = np.rot90(y, k=random_k, axes=(0, 1))
    else:
        y_rot = np.rot90(y, k=random_k, axes=(1, 2))

    return X_rot, y_rot


def augmentation_rotation_batch(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    channel_last: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Randomly rotates images in a batch by 90 degrees intervals. Images
    can be (batch, channels, height, width) or (batch, height, width, channels).

    Args:
        X (np.ndarray): The batch of images to rotate.
        y (Optional[np.ndarray], optional): The batch of labels to rotate. Defaults to None.
    
    Keyword Args:
        channel_last (bool=True): Whether the image is (channels, height, width) or (height, width, channels).
    
    Returns:
        Tuple[np.ndarray, Optional[np.ndarray]]: The rotated images and optionally labels.
    """
    X_rot = np.zeros_like(X, dtype=X.dtype)

    if y is not None:
        y_rot = np.zeros_like(y, dtype=y.dtype)
    else:
        y_rot = None

    for i in range(X.shape[0]):
        if y is None:
            X_rot[i] = augmentation_rotation(X[i], channel_last=channel_last)
        else:
            X_rot[i], y_rot[i] = augmentation_rotation(X[i], y[i], channel_last)

    if y_rot is None:
        return X_rot

    return X_rot, y_rot


def augmentation_mirror(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    channel_last: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Randomly mirrors the image. Images can be (channels, height, width) or (height, width, channels).

    Args:
        X (np.ndarray): The image to mirror.
        y (Optional[np.ndarray], optional): The label to mirror. Defaults to None.

    Keyword Args:
        channel_last (bool=True): Whether the image is (channels, height, width) or (height, width, channels).

    Returns:
        Tuple[np.ndarray, Optional[np.ndarray]]: The mirrored image and optionally the label.
    """
    random_k = np.random.randint(0, 3)

    if random_k == 0:

        if y is not None:
            return X, y
        return X

    flipped_x = X.copy()
    flipped_y = None

    axis = 1 if channel_last else 2
    axis = axis - 1 if random_k == 2 else axis

    flipped_x = np.flip(X, axis=axis)

    if y is not None:
        flipped_y = np.flip(y, axis=axis)
        return flipped_x, flipped_y

    return flipped_x


def augmentation_mirror_batch(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    channel_last: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Randomly mirrors images in a batch. Images can be (batch, channels, height, width) or (batch, height, width, channels).

    Args:
        X (np.ndarray): The batch of images to mirror.
        y (Optional[np.ndarray], optional): The batch of labels to mirror. Defaults to None.
    
    Keyword Args:
        channel_last (bool=True): Whether the image is (channels, height, width) or (height, width, channels).
    
    Returns:
        Tuple[np.ndarray, Optional[np.ndarray]]: The mirrored images and optionally labels.
    """
    X_mirror = np.zeros_like(X, dtype=X.dtype)

    if y is not None:
        y_mirror = np.zeros_like(y, dtype=y.dtype)
    else:
        y_mirror = None

    for i in range(X.shape[0]):
        if y is None:
            X_mirror[i] = augmentation_mirror(X[i], channel_last=channel_last)
        else:
            X_mirror[i], y_mirror[i] = augmentation_mirror(X[i], y[i], channel_last)

    if y_mirror is None:
        return X_mirror

    return X_mirror, y_mirror


def augmentation_pixel_noise(
    X: np.ndarray,
    amount: float = 0.025,
    additive: bool = True,
) -> np.ndarray:
    """
    Adds random noise seperately to each channel of the image. The noise works
    for both channel first and last images.
    input should be (height, width, channels) or (channels, height, width).

    Args:
        X (np.ndarray): The image to add noise to.
    
    Keyword Args:
        amount (float=0.01): The amount of noise to add.
        additive (bool=True): Whether to add or multiply the noise.

    Returns:
        np.ndarray: The noisy image.
    """
    if additive:
        noise = X + np.random.normal(0, amount, X.shape)
    else:
        noise = X * np.random.normal(1, amount, X.shape)

    return noise


def augmentation_pixel_noise_batch(
    X: np.ndarray,
    amount: float = 0.025,
    additive: bool = True,
) -> np.ndarray:
    """
    Adds random noise seperately to each channel of the batch of images. The noise works
    for both channel first and last images.
    input should be (batch, height, width, channels) or (batch, channels, height, width).

    Args:
        X (np.ndarray): The batch of images to add noise to.
    
    Keyword Args:
        amount (float=0.01): The amount of noise to add.
        additive (bool=True): Whether to add or multiply the noise.

    Returns:
        np.ndarray: The noisy batch of images.
    """
    return augmentation_pixel_noise(X, amount, additive)


def augmentation_channel_scale(
    X: np.ndarray,
    amount: float = 0.025,
    additive: bool = False,
    channel_last: bool = True,
) -> np.ndarray:
    """
    Scales the channels of the image seperately by a fixed amount.
    input should be (height, width, channels) or (channels, height, width).

    Args:
        X (np.ndarray): The image to scale the channels of.
    
    Keyword Args:
        amount (float=0.01): The amount to scale the channels by.
        additive (bool=False): Whether to add or multiply the scaling.
        channel_last (bool=True): Whether the image is (channels, height, width) or (height, width, channels).

    Returns:
        np.ndarray: The scaled image.
    """
    x = X.copy()

    if channel_last:
        for i in range(X.shape[2]):

            random_amount = np.random.uniform(-amount, amount)
            if additive:
                x[:, :, i] += random_amount
            else:
                x[:, :, i] *= random_amount
    else:
        for i in range(X.shape[0]):
            random_amount = np.random.uniform(-amount, amount)
            if additive:
                x[i, :, :] += random_amount
            else:
                x[i, :, :] *= random_amount

    return x


def augmentation_channel_scale_batch(
    X: np.ndarray,
    amount: float = 0.025,
    additive: bool = False,
    channel_last: bool = True,
) -> np.ndarray:
    """
    Scales the channels of the batch of images seperately by a fixed amount.
    input should be (batch, height, width, channels) or (batch, channels, height, width).

    Args:
        X (np.ndarray): The batch of images to scale the channels of.
    
    Keyword Args:
        amount (float=0.01): The amount to scale the channels by.
        additive (bool=False): Whether to add or multiply the scaling.
        channel_last (bool=True): Whether the image is (channels, height, width) or (height, width, channels).

    Returns:
        np.ndarray: The scaled batch of images.
    """
    X_scaled = np.zeros_like(X, dtype=X.dtype)

    for i in range(X.shape[0]):
        X_scaled[i] = augmentation_channel_scale(X[i], amount, additive, channel_last)

    return X_scaled


def augmentation_contrast(
    X: np.ndarray,
    contrast_factor: float = 0.025,
    channel_last: bool = True,
) -> np.ndarray:
    """
    Changes the contrast of an image by a random amount, seperately for each channel.
    input should be (height, width, channels) or (channels, height, width).

    Args:
        X (np.ndarray): The image to change the contrast of.
    
    Keyword Args:
        contrast_factor (float=0.01): The amount to change the contrast by.
        channel_last (bool=True): Whether the image is (channels, height, width) or (height, width, channels).

    Returns:
        np.ndarray: The contrast changed image.
    """
    x = X.copy()

    if channel_last:
        mean_pixel = np.mean(X, axis=(0, 1))
    else:
        mean_pixel = np.mean(X, axis=(1, 2))

    if channel_last:
        for i in range(X.shape[2]):
            x[:, :, i] = (x[:, :, i] - mean_pixel[i]) * (1 + contrast_factor) + mean_pixel[i]
    else:
        for i in range(X.shape[0]):
            x[i, :, :] = (x[i, :, :] - mean_pixel[i]) * (1 + contrast_factor) + mean_pixel[i]

    return x


def augmentation_contrast_batch(
    X: np.ndarray,
    contrast_factor: float = 0.025,
    channel_last: bool = True,
) -> np.ndarray:
    """
    Changes the contrast of a batch of images by a random amount, seperately for each channel.
    input should be (batch, height, width, channels) or (batch, channels, height, width).

    Args:
        X (np.ndarray): The batch of images to change the contrast of.
    
    Keyword Args:
        contrast_factor (float=0.01): The amount to change the contrast by.
        channel_last (bool=True): Whether the image is (channels, height, width) or (height, width, channels).

    Returns:
        np.ndarray: The contrast changed batch of images.
    """
    X_contrast = np.zeros_like(X, dtype=X.dtype)

    for i in range(X.shape[0]):
        X_contrast[i] = augmentation_contrast(X[i], contrast_factor, channel_last)

    return X_contrast
