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
from buteo.ai.augmentation_utils import feather_box_2d
from buteo.ai.augmentation_funcs import (
    augmentation_rotation,
    augmentation_mirror,
    augmentation_noise,
    augmentation_channel_scale,
    augmentation_contrast,
    augmentation_drop_pixel,
    augmentation_drop_channel,
    augmentation_blur,
    augmentation_sharpen,
    augmentation_misalign_pixels,
)


@jit(nopython=True, nogil=True, cache=True)
def _augmentation_batch_default(
    X: np.ndarray,
    y: Optional[np.ndarray],
    chance: float = 0.5,
    max_images: float = 1.0,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """ INTERNAL FUNCTION. DO NOT USE. """

    X_aug = X.copy()
    y_aug = y.copy() if y is not None else None

    batch_size = X.shape[0]

    n_mixes = min(
        (np.random.rand(batch_size) <= chance).sum(),
        int(batch_size * max_images),
    )

    if n_mixes == 0:
        aug_flags = np.array([False] * batch_size, dtype=np.int64)
        return X, y, aug_flags

    aug_flags = np.array([False] * batch_size, dtype=np.int64)

    idx_targets = np.random.choice(batch_size, n_mixes, replace=False)
    for i in aug_flags:
        if i in idx_targets:
            aug_flags[i] = True

    return X_aug, y_aug, aug_flags


@jit(nopython=True, nogil=True, cache=True, fastmath=True, parallel=True)
def augmentation_batch_rotation(
    X: np.ndarray,
    y: Optional[np.ndarray],
    chance: float = 0.5,
    max_images: float = 1.0,
    channel_last: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Randomly rotates images in a batch by 90 degrees intervals. Images
    can be (batch, channels, height, width) or (batch, height, width, channels).

    Args:
        X (np.ndarray): The batch of images to rotate.
        y (np.ndarray/None): The label of images to rotate. If None, no label is returned.
    
    Keyword Args:
        y (np.ndarray/none=None): The batch of labels to rotate.
        chance (float=0.5): The chance of rotating the image.
        max_images (float=None): The maximum proportion of the images in the batch to possibly rotate.
        channel_last (bool=True): Whether the image is (channels, height, width) or (height, width, channels).
    
    Returns:
        Tuple[np.ndarray, Optional[np.ndarray]]: The rotated images and optionally labels.
    """
    x_rotated, y_rotated, selection = _augmentation_batch_default(X, y, chance, max_images)

    for idx, selected in enumerate(prange(selection.shape[0])):
        if not selected:
            continue

        if y is None:
            x_rotated[idx], _ = augmentation_rotation(
                X[idx], None,
                chance=1.0,
                k=None,
                channel_last=channel_last,
            )

        else:
            x_rotated[idx], y_rotated[idx] = augmentation_rotation(
                X[idx], y[idx],
                chance=1.0,
                k=None,
                channel_last=channel_last,
            )

    return x_rotated, y_rotated


@jit(nopython=True, nogil=True, cache=True, fastmath=True, parallel=True)
def augmentation_batch_mirror(
    X: np.ndarray,
    y: Optional[np.ndarray],
    chance: float = 0.5,
    max_images: float = 1.0,
    channel_last: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Randomly mirrors images in a batch. Images can be (batch, channels, height, width) or (batch, height, width, channels).

    Args:
        X (np.ndarray): The batch of images to mirror.
        y (np.ndarray/None): The labels of images to mirror. If None, no label is returned.

    Keyword Args:
        chance (float=0.5): The chance of mirroring the image.
        max_images (float=1.0): The maximum proportion  of images in the batch to possibly mirror.
        k (int=None): If None, randomly mirrors the image along the horizontal or vertical axis.
            If 1, mirrors the image along the horizontal axis.
            If 2, mirrors the image along the vertical axis.
        channel_last (bool=True): Whether the image is (channels, height, width) or (height, width, channels).

    Returns:
        Tuple[np.ndarray, Optional[np.ndarray]]: The mirrored images and optionally labels.
    """
    x_mirrored, y_mirrored, selection = _augmentation_batch_default(X, y, chance, max_images)

    for idx, selected in enumerate(prange(selection.shape[0])):
        if not selected:
            continue

        if y is None:
            x_mirrored[idx], _ = augmentation_mirror(
                X[idx], None,
                chance=1.0,
                k=None,
                channel_last=channel_last,
            )

        else:
            x_mirrored[idx], y_mirrored[idx] = augmentation_mirror(
                X[idx], y[idx],
                chance=1.0,
                k=None,
                channel_last=channel_last,
            )

    return x_mirrored, y_mirrored


@jit(nopython=True, nogil=True, cache=True, fastmath=True, parallel=True)
def augmentation_batch_noise(
    X: np.ndarray,
    y: Optional[np.ndarray],
    chance: float = 0.5,
    max_amount: float = 0.1,
    max_images: float = 1.0,
    additive: bool = False,
    channel_last: Any = None, # pylint: disable=unused-argument
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Adds random noise seperately to each channel of the batch of images. The noise works
    for both channel first and last images.
    input should be (batch, height, width, channels) or (batch, channels, height, width).

    Args:
        X (np.ndarray): The batch of images to add noise to.
        y (np.ndarray/None): The labels of images to add noise to. If None, no label is returned.
    
    Keyword Args:
        chance (float=0.5): The chance of adding noise.
        amount (float=0.01): The amount of noise to add.
        max_amount (float=0.1): The maximum amount of noise to add, sampled uniformly.
        max_images (float=1.0): The maximum proportion of images in the batch to add noise to.
        additive (bool=False): Whether to add or multiply the noise.
        channel_last (any=None): Whether the image is (channels, height, width) or (height, width, channels).
            ignored for this function. Kept to keep the same function signature as other augmentations.

    Returns:
        Tuple[np.ndarray, Optional[np.ndarray]]: The batch of images with noise and optionally the unmodified label.
    """
    x_noised, y_noised, selection = _augmentation_batch_default(X, y, chance, max_images)

    for idx, selected in enumerate(prange(selection.shape[0])):
        if not selected:
            continue

        x_noised[idx], _ = augmentation_noise(
            X[idx], None,
            chance=1.0,
            max_amount=max_amount,
            additive=additive,
            channel_last=channel_last,
        )

    return x_noised, y_noised


@jit(nopython=True, nogil=True, cache=True, fastmath=True, parallel=True)
def augmentation_batch_channel_scale(
    X: np.ndarray,
    y: Optional[np.ndarray],
    chance: float = 0.5,
    max_amount: float = 0.1,
    max_images: Optional[float] = 1.0,
    additive: bool = False,
    channel_last: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Scales the channels of the batch of images seperately by a fixed amount.
    input should be (batch, height, width, channels) or (batch, channels, height, width).

    Args:
        X (np.ndarray): The batch of images to scale the channels of.
        y (np.ndarray/None): The labels of images to scale the channels of. If None, no label is returned.
    
    Keyword Args:
        chance (float=0.5): The chance of scaling the channels.
        max_amount (float=0.1): The amount to possible scale the channels by. Sampled uniformly.
        max_images (float=1.0): The maximum number of images to add noise to.
        additive (bool=False): Whether to add or multiply the scaling.
        channel_last (bool=True): Whether the image is (channels, height, width) or (height, width, channels).

    Returns:
        Tuble[np.ndarray, Optional[np.ndarray]]: The batch of images with scaled channels and optionally the unmodified label.
    """
    x_scaled, y_scaled, selection = _augmentation_batch_default(X, y, chance, max_images)

    for idx, selected in enumerate(prange(selection.shape[0])):
        if not selected:
            continue

        x_scaled[idx], _ = augmentation_channel_scale(
            X[idx], None,
            chance=1.0,
            max_amount=max_amount,
            additive=additive,
            channel_last=channel_last,
        )

    return x_scaled, y_scaled


@jit(nopython=True, nogil=True, cache=True, fastmath=True, parallel=True)
def augmentation_batch_contrast(
    X: np.ndarray,
    y: Optional[np.ndarray],
    chance: float = 0.5,
    max_amount: float = 0.1,
    max_images: float = 1.0,
    channel_last: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Changes the contrast of a batch of images by a random amount, seperately for each channel.
    input should be (batch, height, width, channels) or (batch, channels, height, width).

    Args:
        X (np.ndarray): The batch of images to change the contrast of.
        y (np.ndarray/None): The labels of images to change the contrast of. If None, no label is returned.
    
    Keyword Args:
        chance (float=0.5): The chance of changing the contrast.
        max_amount (float=0.01): The amount to change the contrast by.
        max_images (float=1.0): The maximum proportion of images in the batch to change the contrast of.
        channel_last (bool=True): Whether the image is (channels, height, width) or (height, width, channels).

    Returns:
        Tuble[np.ndarray, Optional[np.ndarray]]: The batch of images with changed contrast and optionally the unmodified label.
    """
    x_constrast, y_contrast, selection = _augmentation_batch_default(X, y, chance, max_images)

    for idx, selected in enumerate(prange(selection.shape[0])):
        if not selected:
            continue

        x_constrast[idx], _ = augmentation_contrast(
            X[idx], None,
            chance=1.0,
            max_amount=max_amount,
            channel_last=channel_last,
        )

    return x_constrast, y_contrast


@jit(nopython=True, nogil=True, cache=True, fastmath=True, parallel=True)
def augmentation_batch_drop_pixel(
    X: np.ndarray,
    y: Optional[np.ndarray],
    chance: float = 0.5,
    drop_probability: float = 0.1,
    drop_value: float = 0.0,
    max_images: float = 1.0,
    channel_last: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Drops a random pixels from a batch of images.
    input should be (batch, height, width, channels) or (batch, channels, height, width).

    Args:
        X (np.ndarray): The batch of images to drop a pixel from.
        y (np.ndarray/None): The labels of images to drop pixels from. If None, no label is returned.
    
    Keyword Args:
        chance (float=0.5): The chance of dropping a pixel.
        drop_probability (float=0.05): The probability of dropping a pixel.
        drop_value (float=0.0): The value to drop the pixel to.
        max_images (float=1.0): The maximum proportion of images in the batch to drop a pixel from.
        channel_last (bool=True): Whether the image is (channels, height, width) or (height, width, channels).

    Returns:
        Tuple[np.ndarray, Optional[np.ndarray]]: The batch of images with the dropped pixels and optionally the unmodified label.
    """
    x_drop, y_drop, selection = _augmentation_batch_default(X, y, chance, max_images)

    for idx, selected in enumerate(prange(selection.shape[0])):
        if not selected:
            continue

        x_drop[idx], _ = augmentation_drop_pixel(
            X[idx], None,
            chance=1.0,
            drop_probability=drop_probability,
            drop_value=drop_value,
            channel_last=channel_last,
        )

    return x_drop, y_drop


@jit(nopython=True, nogil=True, cache=True, fastmath=True, parallel=True)
def augmentation_batch_drop_channel(
    X: np.ndarray,
    y: Optional[np.ndarray],
    chance: float = 0.5,
    drop_probability: float = 0.1,
    drop_value: float = 0.0,
    max_images: float = 1.0,
    channel_last: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Drops a random channel from a batch of images.
    input should be (batch, height, width, channels) or (batch, channels, height, width).
    A maximum of one channel will be dropped.

    Args:
        X (np.ndarray): The batch of images to drop a channel from.
        y (np.ndarray/None): The labels of images to drop channels from. If None, no label is returned.

    Keyword Args:
        chance (float=0.5): The chance of dropping a channel.
        max_images (float=1.0): The maximum proportion of images in the batch to drop a channel from.
        drop_probability (float=0.1): The probability of dropping a channel.
        drop_value (float=0.0): The value to drop the channel to.
        channel_last (bool=True): Whether the image is (channels, height, width) or (height, width, channels).
    """
    x_drop, y_drop, selection = _augmentation_batch_default(X, y, chance, max_images)

    for idx, selected in enumerate(prange(selection.shape[0])):
        if not selected:
            continue

        x_drop[idx], _ = augmentation_drop_channel(
            X[idx], None,
            chance=1.0,
            drop_probability=drop_probability,
            drop_value=drop_value,
            channel_last=channel_last,
        )

    return x_drop, y_drop


@jit(nopython=True, nogil=True, cache=True, fastmath=True, parallel=True)
def augmentation_batch_blur(
    X: np.ndarray,
    y: Optional[np.ndarray],
    chance: float = 0.5,
    max_images: float = 1.0,
    intensity: float = 1.0,
    apply_to_y: bool = False,
    channel_last: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Blurs a batch of images at random.
    input should be (batch, height, width, channels) or (batch, channels, height, width).

    Args:
        X (np.ndarray): The batch of images to potentially blur.

    Keyword Args:
        y (np.ndarray/none=None): The label to blur a pixel in. If None, no label is returned.
        chance (float=0.5): The chance of blurring a pixel.
        max_images (float=1.0): The maximum proportion of images in the batch to blur.
        intensity (float=1.0): The intensity of the blur. from 0.0 to 1.0.
        apply_to_y (bool=False): Whether to blur the label as well.
        channel_last (bool=True): Whether the image is (channels, height, width) or (height, width, channels).
    """
    x_blurred, y_blurred, selection = _augmentation_batch_default(X, y, chance, max_images)

    for idx, selected in enumerate(prange(selection.shape[0])):
        if not selected:
            continue

        x_blurred[idx], _ = augmentation_blur(
            X[idx], None,
            chance=1.0,
            intensity=intensity,
            apply_to_y=apply_to_y,
            channel_last=channel_last,
        )

    return x_blurred, y_blurred


@jit(nopython=True, nogil=True, cache=True, fastmath=True, parallel=True)
def augmentation_batch_sharpen(
    X: np.ndarray,
    y: Optional[np.ndarray],
    chance: float = 0.5,
    max_images: float = 1.0,
    intensity: float = 1.0,
    apply_to_y: bool = False,
    channel_last: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Sharpens a batch of images at random.
    input should be (batch, height, width, channels) or (batch, channels, height, width).

    Args:
        X (np.ndarray): The batch of images to potentially sharpen.

    Keyword Args:
        y (np.ndarray/none=None): The label to sharpen a pixel in. If None, no label is returned.
        chance (float=0.5): The chance of sharpening a pixel.
        max_images (float=1.0): The maximum proportion of images in the batch to sharpen.
        intensity (float=1.0): The intensity of the sharpening. from 0.0 to 1.0.
        apply_to_y (bool=False): Whether to sharpen the label as well.
        channel_last (bool=True): Whether the image is (channels, height, width) or (height, width, channels).
    """
    x_sharpened, y_sharpened, selection = _augmentation_batch_default(X, y, chance, max_images)

    for idx, selected in enumerate(prange(selection.shape[0])):
        if not selected:
            continue

        x_sharpened[idx], _ = augmentation_sharpen(
            X[idx], None,
            chance=1.0,
            intensity=intensity,
            apply_to_y=apply_to_y,
            channel_last=channel_last,
        )

    return x_sharpened, y_sharpened


@jit(nopython=True, nogil=True, cache=True, fastmath=True, parallel=True)
def augmentation_batch_misalign(
    X: np.ndarray,
    y: Optional[np.ndarray],
    chance: float = 0.5,
    max_offset: float = 0.5,
    max_images: float = 0.2,
    channel_last: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Misaligns the channels of a batch of images at random.
    input should be (batch, height, width, channels) or (batch, channels, height, width).

    Args:
        X (np.ndarray): The batch of images to potentially misalign the channels of.

    Keyword Args:
        y (np.ndarray/none=None): The label to misalign the channels of a pixel in. If None, no label is returned.
        chance (float=0.5): The chance of misaligning the channels of a pixel.
        max_offset (float=0.5): The maximum offset to misalign the channels by.
        max_images (float=0.2): The maximum number of images to misalign the channels of.
        max_channels (int=1): The maximum number of channels to misalign.
        channel_last (bool=True): Whether the image is (channels, height, width) or (height, width, channels).
    """
    x_misaligned, y_misaligned, selection = _augmentation_batch_default(X, y, chance, max_images)

    for idx, selected in enumerate(prange(selection.shape[0])):
        if not selected:
            continue

        x_misaligned[idx], _ = augmentation_misalign_pixels(
            X[idx], None,
            chance=1.0,
            max_offset=max_offset,
            channel_last=channel_last,
        )

    return x_misaligned, y_misaligned


@jit(nopython=True, nogil=True, cache=True, fastmath=True, parallel=True)
def augmentation_batch_cutmix(
    X: np.ndarray,
    y: Optional[np.ndarray],
    chance: float = 0.5,
    max_size: float = 0.5,
    max_images: float = 0.2,
    feather: bool = True,
    feather_dist: int = 3,
    channel_last: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Cutmixes a batch of images at random.
    input should be (batch, height, width, channels) or (batch, channels, height, width).

    Args:
        X (np.ndarray): The batch of images to potentially cutmix.

    Keyword Args:
        y (np.ndarray/none=None): The label to cutmix a pixel in. If None, no label is returned.
        chance (float=0.5): The chance of cutmixing a pixel.
        max_size (float=0.5): The maximum size of the patch to cutmix. In percentage of the image width.
        max_images (float=0.2): The maximum percentage of images in a batch to mixup.
        feather (bool=True): Whether to feather the edges of the cutmix.
        feather_dist (int=3): The distance to feather the edges of the cutmix in pixels
        channel_last (bool=True): Whether the image is (channels, height, width) or (height, width, channels).
    """
    x = X.copy().astype(np.float32, copy=False)

    if channel_last:
        batch_size, height, width, channels = x.shape
    else:
        batch_size, channels, height, width = x.shape

    n_mixes = min(
        (np.random.rand(batch_size) <= chance).sum(),
        int(batch_size * max_images),
    )

    if n_mixes == 0:
        return x, y

    idx_targets = np.random.choice(batch_size, n_mixes, replace=False)

    for idx_target in idx_targets:
        patch_height = np.random.randint(1, int(height * max_size))
        patch_width = np.random.randint(1, int(width * max_size))

        if feather:
            patch_height += feather_dist * 2
            patch_width += feather_dist * 2

            patch_height = min(patch_height, height)
            patch_width = min(patch_width, width)

        x0 = np.random.randint(0, width - patch_width)
        y0 = np.random.randint(0, height - patch_height)
        x1 = x0 + patch_width
        y1 = y0 + patch_height

        source_img = np.random.choice(np.where(np.arange(batch_size) != idx_target)[0])

        if feather:
            bbox = np.array([x0, x1, y0, y1])
            feather_weight_source, feather_weight_target = feather_box_2d(x[idx_target, :, :, 0], bbox, feather_dist)

            feather_weight_source = np.repeat(feather_weight_source[:, :, np.newaxis], channels, axis=2)
            feather_weight_target = np.repeat(feather_weight_target[:, :, np.newaxis], channels, axis=2)

            if channel_last:

                x[idx_target, y0:y1, x0:x1, :] = (
                    x[idx_target, y0:y1, x0:x1, :] * feather_weight_target[y0:y1, x0:x1, :]
                    + x[source_img, y0:y1, x0:x1, :] * feather_weight_source[y0:y1, x0:x1, :]
                )
                if y is not None:
                    y[idx_target, y0:y1, x0:x1, :] = (
                        y[idx_target, y0:y1, x0:x1, :] * feather_weight_target[y0:y1, x0:x1, :]
                        + y[source_img, y0:y1, x0:x1, :] * feather_weight_source[y0:y1, x0:x1, :]
                    )
        else:
            if channel_last:
                x[idx_target, y0:y1, x0:x1, :] = x[source_img, y0:y1, x0:x1, :]
                if y is not None:
                    y[idx_target, y0:y1, x0:x1, :] = y[source_img, y0:y1, x0:x1, :]
            else:
                x[idx_target, :, y0:y1, x0:x1] = x[source_img, :, y0:y1, x0:x1]
                if y is not None:
                    y[idx_target, :, y0:y1, x0:x1] = y[source_img, :, y0:y1, x0:x1]

    return x, y


@jit(nopython=True, nogil=True, cache=True, fastmath=True, parallel=True)
def augmentation_batch_mixup(
    X: np.ndarray,
    y: Optional[np.ndarray],
    chance: float = 0.5,
    max_mixes: float = 0.2,
    channel_last: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Mixups a batch of images at random. This works by doing a linear intepolation between
    two images in the batch and then adding a random weight to each image.

    Mixup involves taking two images and blending them together by randomly interpolating
    their pixel values. More specifically, suppose we have two images x and x' with their
    corresponding labels y and y'. To generate a new training example, mixup takes a
    weighted sum of x and x', such that the resulting image x^* = λx + (1-λ)x',
    where λ is a randomly chosen interpolation coefficient. The label for the new image
    is also a weighted sum of y and y' based on the same interpolation coefficient.

    input should be (batch, height, width, channels) or (batch, channels, height, width).

    Args:
        X (np.ndarray): The batch of images to potentially mixup.

    Keyword Args:
        y (np.ndarray/none=None): The label to mixup a pixel in. If None, no label is returned.
        chance (float=0.5): The chance of mixuping a pixel.
        max_mixes (float=0.2): The maximum percentage of images in a batch to mixup.
        channel_last (bool=True): Whether the image is (channels, height, width) or (height, width, channels).
    """
    x = X.copy().astype(np.float32, copy=False)

    batch_size = x.shape[0]

    n_mixes = min(
        (np.random.rand(batch_size) <= chance).sum(),
        int(batch_size * max_mixes),
    )

    if n_mixes == 0:
        return x, y

    idx_targets = np.random.choice(batch_size, n_mixes, replace=False)

    for idx_target in idx_targets:
        source_img = np.random.choice(np.where(np.arange(batch_size) != idx_target)[0])
        mixup_coeff = np.random.rand()

        if channel_last:
            x[idx_target, :, :, :] = x[idx_target, :, :, :] * mixup_coeff + x[source_img, :, :, :] * (1 - mixup_coeff)
            if y is not None:
                y[idx_target, :, :, :] = y[idx_target, :, :, :] * mixup_coeff + y[source_img, :, :, :] * (1 - mixup_coeff)
        else:
            x[idx_target, :, :, :] = x[idx_target, :, :, :] * mixup_coeff + x[source_img, :, :, :] * (1 - mixup_coeff)
            if y is not None:
                y[idx_target, :, :, :] = y[idx_target, :, :, :] * mixup_coeff + y[source_img, :, :, :] * (1 - mixup_coeff)

    return x, y
