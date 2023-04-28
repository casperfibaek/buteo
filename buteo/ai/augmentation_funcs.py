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
    _rotate_arr,
    _mirror_arr,
)
from buteo.array.convolution import (
    convolve_array_simple_chw,
    convolve_array_simple_hwc,
    convolve_array_simple_2D,
    _simple_blur_kernel_2d_3x3,
    _simple_unsharp_kernel_2d_3x3,
    _simple_shift_kernel_2d,
)



@jit(nopython=True, nogil=True, cache=True, fastmath=True)
def augmentation_rotation(
    X: np.ndarray,
    k: Optional[int] = None,
    channel_last: bool = True,
    inplace=False,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Randomly rotate the image by 90 degrees intervals. Images
    can be (channels, height, width) or (height, width, channels).
    
    Parameters
    ----------
    X : np.ndarray
        The image to rotate.

    k : int, optional
        The number of 90 degree intervals to rotate by, default: None.

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

    random_k = np.random.randint(1, 4) if k is None else k

    X[:] = _rotate_arr(X, random_k, channel_last)

    return X


@jit(nopython=True, nogil=True, cache=True, fastmath=True)
def augmentation_rotation_xy(
    X: np.ndarray,
    y: np.ndarray,
    k: Optional[int] = None,
    channel_last: bool = True,
    inplace: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Randomly rotate the image and label by 90 degrees intervals. Images
    can be (channels, height, width) or (height, width, channels).
    
    Parameters
    ----------
    X : np.ndarray
        The image to rotate.

    y : np.ndarray
        The label to rotate.

    k : int, optional
        The number of 90 degree intervals to rotate by, default: None.

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

    random_k = np.random.randint(1, 4) if k is None else k

    X[:] = _rotate_arr(X, random_k, channel_last)
    y[:] = _rotate_arr(y, random_k, channel_last)

    return X, y


@jit(nopython=True, nogil=True, cache=True, fastmath=True)
def augmentation_mirror(
    X: np.ndarray,
    k: Optional[int] = None,
    channel_last: bool = True,
    inplace: bool = False,
) -> np.ndarray:
    """
    Randomly mirrors the image.
    Images can be (channels, height, width) or (height, width, channels).
    
    Parameters
    ----------
    X : np.ndarray
        The image to mirror.

    k : int, optional
        If None, randomly mirrors the image along the horizontal or vertical axis.
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

    random_k = np.random.randint(1, 4) if k is None else k

    X[:] = _mirror_arr(X, random_k, channel_last)

    return X


@jit(nopython=True, nogil=True, cache=True, fastmath=True)
def augmentation_mirror_xy(
    X: np.ndarray,
    y: np.ndarray,
    k: Optional[int] = None,
    channel_last: bool = True,
    inplace: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Randomly mirrors the image. Images can be (channels, height, width) or (height, width, channels).
    
    Parameters
    ----------
    X : np.ndarray
        The image to mirror.

    y : np.ndarray
        The label to mirror.

    k : int, optional
        If None, randomly mirrors the image along the horizontal or vertical axis.
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

    random_k = np.random.randint(1, 4) if k is None else k

    X[:] = _mirror_arr(X, random_k, channel_last)
    y[:] = _mirror_arr(y, random_k, channel_last)

    return X, y


@jit(nopython=True, nogil=True, cache=True, fastmath=True)
def augmentation_noise_uniform(
    X: np.ndarray,
    max_amount: float = 0.1,
    additive: bool = False,
    channel_last: Any = None, # pylint: disable=unused-argument
    inplace: bool = False,
) -> np.ndarray:
    """
    Adds random noise seperately to each pixel of the image. The noise works
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

    if additive:
        random_noise = np.random.uniform(-amount, amount, X.shape).astype(X.dtype)
        X[:] += random_noise
    else:
        random_noise = np.random.uniform(1 - amount, 1 + amount, X.shape).astype(X.dtype)
        X[:] *= random_noise

    return X


@jit(nopython=True, nogil=True, cache=True, fastmath=True)
def augmentation_noise_normal(
    X: np.ndarray,
    max_amount: float = 0.1,
    additive: bool = False,
    channel_last: Any = None, # pylint: disable=unused-argument
    inplace: bool = False,
) -> np.ndarray:
    """
    Adds random noise seperately to each pixel of the image. The noise works
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

    if additive:
        random_noise = np.random.normal(0, amount, X.shape).astype(X.dtype)
        X[:] += random_noise
    else:
        random_noise = np.random.normal(1, amount, X.shape).astype(X.dtype)
        X[:] *= random_noise

    return X


@jit(nopython=True, nogil=True, cache=True, fastmath=True, parallel=True)
def augmentation_channel_scale(
    X: np.ndarray,
    max_amount: float = 0.1,
    additive: bool = False,
    channel_last: bool = True,
    inplace: bool = False,
) -> np.ndarray:
    """
    Scales the channels of the image seperately by a fixed amount.
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


@jit(nopython=True, nogil=True, cache=True, fastmath=True, parallel=True)
def augmentation_contrast(
    X: np.ndarray,
    max_amount: float = 0.1,
    channel_last: bool = True,
    inplace: bool = False,
) -> np.ndarray:
    """
    Changes the contrast of an image by a random amount, seperately for each channel.
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


@jit(nopython=True, nogil=True, cache=True, fastmath=True, parallel=True)
def augmentation_drop_pixel(
    X: np.ndarray,
    drop_probability: float = 0.01,
    drop_value: float = 0.0,
    channel_last: bool = True,
    inplace: bool = False,
) -> np.ndarray:
    """
    Drops random pixels from an image. Input should be (height, width, channels) or (channels, height, width).
    Only drops pixels from features, not labels.
    
    Parameters
    ----------
    X : np.ndarray
        The image to drop a pixel from.

    drop_probability : float, optional
        The probability of dropping a pixel, default: 0.05.

    drop_value : float, optional
        The value to drop the pixel to, default: 0.0.

    channel_last : bool, optional
        Whether the image is (channels, height, width) or (height, width, channels), default: True.
    
    inplace : bool, optional
        Whether to perform the rotation in-place, default: False.

    Returns
    -------
    np.ndarray
        The image with dropped pixels.
    """
    if not inplace:
        X = X.copy()

    # Create a mask of random numbers
    mask = np.random.uniform(0.0, 1.0, size=X.shape).astype(np.float32)

    # Agreed. This looks terrible. But it's the only way to get numba to parallelize this.
    if channel_last:
        for col in prange(X.shape[0]):
            for row in prange(X.shape[1]):
                for channel in prange(X.shape[2]):
                    if mask[col, row, channel] <= drop_probability:
                        X[col, row, channel] = drop_value

    else:
        for channel in prange(X.shape[0]):
            for col in prange(X.shape[1]):
                for row in prange(X.shape[2]):
                    if mask[channel, col, row] <= drop_probability:
                        X[channel, col, row] = drop_value

    return X


@jit(nopython=True, nogil=True, cache=True, fastmath=True)
def augmentation_drop_channel(
    X: np.ndarray,
    drop_value: float = 0.0,
    channel_last: bool = True,
    inplace: bool = False,
) -> np.ndarray:
    """
    Drops a random channel from an image. Input should be (height, width, channels) or (channels, height, width).
    A maximum of one channel will be dropped.
    
    Parameters
    ----------
    X : np.ndarray
        The image to drop a channel from.

    drop_value : float, optional
        The value to drop the channel to, default: 0.0.

    channel_last : bool, optional
        Whether the image is (channels, height, width) or (height, width, channels), default: True.
    
    inplace : bool, optional
        Whether to perform the rotation in-place, default: False.

    Returns
    -------
    np.ndarray
        The image with dropped channels.
    """
    if not inplace:
        X = X.copy()

    channels = X.shape[2] if channel_last else X.shape[0]
    channel_to_drop = np.random.randint(0, channels)

    if channel_last:
        X[:, :, channel_to_drop] = drop_value
    else:
        X[channel_to_drop, :, :] = drop_value

    return X


@jit(nopython=True, nogil=True, cache=True, fastmath=True)
def augmentation_blur(
    X: np.ndarray,
    channel_last: bool = True,
    inplace: bool = False,
) -> np.ndarray:
    """
    Blurs an image at random. Input should be (height, width, channels) or (channels, height, width).
    
    NOTE: Beware of datatypes. Consider casting to float32 before adding noise.

    Parameters
    ----------
    X : np.ndarray
        The image to blur.

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

    offsets, weights = _simple_blur_kernel_2d_3x3()

    if channel_last:
        X[:] = convolve_array_simple_hwc(X, offsets, weights)
    else:
        X[:] = convolve_array_simple_chw(X, offsets, weights)

    return X


@jit(nopython=True, nogil=True, cache=True, fastmath=True)
def augmentation_blur_xy(
    X: np.ndarray,
    y: np.ndarray,
    channel_last: bool = True,
    inplace: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Blurs an image at random. Input should be (height, width, channels) or (channels, height, width).
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
    x_blurred, _ = augmentation_blur(X, channel_last, inplace)
    y_blurred, _ = augmentation_blur(y, channel_last, inplace)

    return x_blurred, y_blurred


@jit(nopython=True, nogil=True, cache=True, fastmath=True, parallel=True)
def augmentation_sharpen(
    X: np.ndarray,
    channel_last: bool = True,
    inplace: bool = False,
) -> np.ndarray:
    """
    Sharpens an image at random. Input should be (height, width, channels) or (channels, height, width).

    NOTE: Beware of datatypes. Consider casting to float32 before adding noise.

    Parameters
    ----------
    X : np.ndarray
        The image to sharpen.

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

    offsets, weights = _simple_unsharp_kernel_2d_3x3()

    if channel_last:
        X[:] = convolve_array_simple_hwc(X, offsets, weights)
    else:
        X[:] = convolve_array_simple_chw(X, offsets, weights)

    return X


@jit(nopython=True, nogil=True, cache=True, fastmath=True)
def augmentation_sharpen_xy(
    X: np.ndarray,
    y: np.ndarray,
    channel_last: bool = True,
    inplace: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Sharpens an image at random. Input should be (height, width, channels) or (channels, height, width).
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
    x_sharpened, _ = augmentation_sharpen(X, channel_last, inplace)
    y_sharpened, _ = augmentation_sharpen(y, channel_last, inplace)

    return x_sharpened, y_sharpened


@jit(nopython=True, nogil=True, cache=True, fastmath=True)
def augmentation_misalign(
    X: np.ndarray,
    max_offset: float = 0.5,
    channel_last: bool = True,
    inplace: bool = False,
) -> np.ndarray:
    """
    Misaligns one channel in the image at random.
    input should be (height, width, channels) or (channels, height, width).

    Parameters
    ----------
    X : np.ndarray
        The image to misalign the channels of.

    max_offset : float, optional
        The maximum offset to misalign the channels by. Default: 0.5.

    channel_last : bool, optional
        Whether the image is (channels, height, width) or (height, width, channels). Default: True.

    Returns
    -------
    np.ndarray
        The misaligned image.
    """
    if not inplace:
        X = X.copy()

    offsets, weights = _simple_shift_kernel_2d(
        min(np.random.rand(), max_offset),
        min(np.random.rand(), max_offset),
    )

    channels = X.shape[2] if channel_last else X.shape[0]
    channel_to_adjust = np.random.randint(0, channels)

    if channel_last:
        X[:, :, channel_to_adjust] = convolve_array_simple_2D(
            X[:, :, channel_to_adjust], offsets, weights,
        )
    else:
        X[channel_to_adjust, :, :] = convolve_array_simple_2D(
            X[channel_to_adjust, :, :], offsets, weights,
        )

    return X


@jit(nopython=True, nogil=True, cache=True, fastmath=True, parallel=True)
def augmentation_cutmix(
    X_target: np.ndarray,
    y_target: np.ndarray,
    X_source: np.ndarray,
    y_source: np.ndarray,
    min_size: float = 0.333,
    max_size: float = 0.666,
    channel_last: bool = True,
    inplace: bool = False,
) -> Tuple[np.ndarray, np.ndarray,]:
    """
    Cutmixes two images.
    Input should be (height, width, channels) or (channels, height, width).

    Parameters
    ----------
    X_target : np.ndarray
        The image to transfer the cutmix to.

    y_target : np.ndarray
        The label to transfer the cutmix to.

    X_source : np.ndarray
        The image to cutmix from.

    y_source : np.ndarray
        The label to cutmix from.

    min_size : float, optional
        The minimum size of the patch to cutmix. In percentage of the image width, default: 0.333.

    max_size : float, optional
        The maximum size of the patch to cutmix. In percentage of the image width, default: 0.666.

    channel_last : bool, optional
        Whether the image is (channels, height, width) or (height, width, channels), default: True.
    
    inplace : bool, optional
        Whether to perform the rotation in-place, default: False.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The cutmixed image and label.
    """
    if not inplace:
        X_target = X_target.copy()
        y_target = y_target.copy()

    if channel_last:
        height, width, _channels_x = X_target.shape
    else:
        _channels_x, height, width = X_target.shape

    # Create random size patch
    patch_height = np.random.randint(int(height * min_size), int(height * max_size))
    patch_width = np.random.randint(int(width * min_size), int(width * max_size))

    # Determine patch location
    x0 = np.random.randint(0, width - patch_width)
    y0 = np.random.randint(0, height - patch_height)
    x1 = x0 + patch_width
    y1 = y0 + patch_height

    # Cut and paste
    if channel_last:
        X_target[y0:y1, x0:x1, :] = X_source[y0:y1, x0:x1, :]
        y_target[y0:y1, x0:x1, :] = y_source[y0:y1, x0:x1, :]

    else:
        X_target[:, y0:y1, x0:x1] = X_source[:, y0:y1, x0:x1]
        y_target[:, y0:y1, x0:x1] = y_source[:, y0:y1, x0:x1]

    return X_target, y_target


@jit(nopython=True, nogil=True, cache=True, fastmath=True, parallel=True)
def augmentation_mixup(
    X_target: np.ndarray,
    y_target: np.ndarray,
    X_source: np.ndarray,
    y_source: np.ndarray,
    min_size: float = 0.333,
    max_size: float = 0.666,
    label_mix: int = 0,
    channel_last: bool = True, # pylint: disable=unused-argument
    inplace: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Mixups two images at random. This works by doing a linear intepolation between
    two images and then adding a random weight to each image.

    Mixup involves taking two images and blending them together by randomly interpolating
    their pixel values. More specifically, suppose we have two images x and x' with their
    corresponding labels y and y'. To generate a new training example, mixup takes a
    weighted sum of x and x', such that the resulting image x^* = λx + (1-λ)x',
    where λ is a randomly chosen interpolation coefficient. The label for the new image
    is also a weighted sum of y and y' based on the same interpolation coefficient.

    input should be (height, width, channels) or (channels, height, width).

    Parameters
    ----------
    X_target : np.ndarray
        The image to transfer to.

    y_target : np.ndarray
        The label to transfer to.

    X_source : np.ndarray
        The image to transfer from.

    y_source : np.ndarray
        The label to transfer from.

    min_size : float, optional
        The minimum mixup coefficient, default: 0.333.

    max_size : float, optional
        The maximum mixup coefficient, default: 0.666.

    label_mix : int, optional
        If 0, the labels will be mixed by the weights. If 1, the target label will be used. If 2, 
        the source label will be used. If 3, the max of the labels will be used. If 4, the min 
        of the labels will be used. If 5, the max of the image with the highest weight will be used. 
        If 6, the min of the image with the highest weight will be used. If 7, the sum of the labels 
        will be used, default: 0.

    channel_last : bool, optional
        Whether the image is (channels, height, width) or (height, width, channels), default: True.
    
    inplace : bool, optional
        Whether to perform the rotation in-place, default: False.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The mixed up image and label.
    """
    if not inplace:
        X_target = X_target.copy()
        y_target = y_target.copy()

    mixup_coeff = np.float32(
        min(np.random.uniform(min_size, max_size + 0.001), 1.0),
    )

    X_target[:] = (X_target * mixup_coeff) + (X_source * (np.float32(1.0) - mixup_coeff))

    if label_mix == 0:
        y_target[:] = (y_target * mixup_coeff) + (y_source * (np.float32(1.0) - mixup_coeff))
    elif label_mix == 1:
        y_target[:] = y_target # pylint: disable=self-assigning-variable
    elif label_mix == 2:
        y_target[:] = y_source
    elif label_mix == 3:
        y_target[:] = np.maximum(y_target, y_source)
    elif label_mix == 4:
        y_target[:] = np.minimum(y_target, y_source)
    elif label_mix == 5:
        y_target[:] = y_target if mixup_coeff >= 0.5 else y_source
    elif label_mix == 6:
        y_target[:] = y_target if mixup_coeff >= 0.5 else y_source
    elif label_mix == 7:
        y_target[:] = y_target + y_source

    return X_target, y_target
