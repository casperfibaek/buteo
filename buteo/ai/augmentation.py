"""
This module contains functions for augmenting images that are
suited to remote sensing imagery.
"""
# Standard library
from typing import Optional, Tuple, List, Union, Any

# External
import numpy as np
from numba import jit, prange


@jit(nopython=True, nogil=True, cache=True, fastmath=True, parallel=True)
def feather_box(
    array: np.ndarray,
    bbox: Union[List[int], Tuple[int]],
    feather_dist: int = 3,
) -> np.ndarray:
    """
    Feather a box into an array (2D). Box should be the original box
        buffered by feather_dist.

    Args:
        array_whole (np.ndarray): The array containing the box.
        bbox (Union[List[int], Tuple[int]]): The box.
            the bbox should be in the form [x_min, x_max, y_min, y_max].
    
    Keyword Args:
        feather_dist (int=3): The distance to feather the box.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The featherweights for the array and the bbox.
    """
    # Get the bbox
    x_min, x_max, y_min, y_max = bbox

    kernel_size = int((feather_dist * 2) + 1)
    n_offsets = kernel_size ** 2
    max_dist = (np.sqrt(2) / 2) + feather_dist

    feather_offsets = np.zeros((n_offsets, 2), dtype=np.int64)

    within_circle = 0
    for i in prange(-feather_dist, feather_dist + 1):
        for j in range(-feather_dist, feather_dist + 1):
            dist = np.sqrt(i ** 2 + j ** 2)
            if dist <= max_dist:
                feather_offsets[within_circle][0] = i
                feather_offsets[within_circle][1] = j
                within_circle += 1

    feather_offsets = feather_offsets[:within_circle]

    masking_array = np.zeros_like(array, dtype=np.uint8)
    x0 = x_min + feather_dist
    x1 = x_max + 1 - feather_dist
    y0 = y_min + feather_dist
    y1 = y_max + 1 - feather_dist

    masking_array[y0:y1, x0:x1] = 1

    feather_weights_box = np.zeros_like(array, dtype=np.float32)
    feather_weights_array = np.zeros_like(array, dtype=np.float32)

    # Feather the box
    for y in prange(masking_array.shape[0]):
        for x in prange(masking_array.shape[1]):
            within_bbox = 0
            for offset in feather_offsets:
                x_offset, y_offset = offset
                x_new = x + x_offset
                y_new = y + y_offset

                if y_new < 0 or y_new >= array.shape[0]:
                    continue

                if x_new < 0 or x_new >= array.shape[1]:
                    continue

                if x_new >= x_min and x_new <= x_max and y_new >= y_min and y_new <= y_max:
                    within_bbox += 1

            weights_box = within_bbox / n_offsets
            feather_weights_box[y, x] = weights_box
            feather_weights_array[y, x] = 1 - weights_box

    return feather_weights_array, feather_weights_box


@jit(nopython=True, nogil=True, cache=True, fastmath=True, inline='always')
def blur_kernel() -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a 2D blur kernel.

    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1],

    Returns:
        Tuple[np.ndarray, np.ndarray]: The offsets and weights.
    """

    offsets = np.array([
        [0 , 0], [0 , -1], [0 , 1],
        [-1, 0], [-1, -1], [-1, 1],
        [1 , 0], [1 , -1], [1 , 1],
    ])

    weights = np.array([
        1.0, 2.0, 1.0, 2.0, 4.0, 2.0, 1.0, 2.0, 1.0,
    ], dtype=np.float32)

    return offsets, weights


@jit(nopython=True, nogil=True, cache=True, fastmath=True, inline='always')
def unsharp_kernel() -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a 2D unsharp kernel.

    [ 0, -1,  0],
    [-1,  5, -1],
    [ 0, -1,  0],

    Returns:
        Tuple[np.ndarray, np.ndarray]: The offsets and weights.
    """
    offsets = np.array([
        [0 , 0], [0 , -1], [0 , 1],
        [-1, 0], [-1, -1], [-1, 1],
        [1 , 0], [1 , -1], [1 , 1],
    ])

    weights = np.array([
        0, -1.0, 0.0, -1.0, 5.0, -1.0, 0.0, -1.0, 0.0,
    ], dtype=np.float32)

    return offsets, weights


@jit(nopython=True, nogil=True, cache=True, fastmath=True, parallel=True)
def convolution_simple(
    array: np.ndarray,
    offsets: np.ndarray,
    weights: np.ndarray,
    intensity: float = 1.0,
):
    """
    Convolve a kernel with an array using a simple method.

    Args:
        array (np.ndarray): The array to convolve.
        offsets (np.ndarray): The offsets of the kernel.
        weights (np.ndarray): The weights of the kernel.
    
    Keyword Args:
        intensity (float=1.0): The intensity of the convolution. If
            1.0, the convolution is applied as is. If 0.5, the
            convolution is applied at half intensity.

    Returns:
        np.ndarray: The convolved array.
    """
    result = np.empty_like(array, dtype=np.float32)

    if intensity <= 0.0:
        return array.astype(np.float32)

    for col in prange(array.shape[0]):
        for row in prange(array.shape[1]):
            weights_sum = 0.0
            result_value = 0.0
            for i in prange(offsets.shape[0]):
                new_col = col + offsets[i, 0]
                new_row = row + offsets[i, 1]

                if 0 <= new_col < array.shape[0] and 0 <= new_row < array.shape[1]:
                    result_value += array[new_col, new_row] * weights[i]
                    weights_sum += weights[i]

            result[col, row] = result_value / weights_sum

    if intensity < 1.0:
        result *= intensity
        array *= (1.0 - intensity)
        result += array

    return result


@jit(nopython=True, nogil=True, cache=True, fastmath=True, inline='always')
def rotate_90(
    arr: np.ndarray,
    channel_last: bool = True
) -> np.ndarray:
    """ Rotate a 3D array 90 degrees clockwise.
    
    Args:
        arr (np.ndarray): The array to rotate.
    
    Keyword Args:
        channel_last (bool=True): Whether the last axis is the channel axis.

    Returns:
        np.ndarray: The rotated array.
    """
    if channel_last:
        return arr[::-1, :, :].transpose(1, 0, 2) # (H, W, C)

    return arr[:, ::-1, :].transpose(0, 2, 1) # (C, H, W)


@jit(nopython=True, nogil=True, cache=True, fastmath=True, inline='always')
def rotate_180(
    arr: np.ndarray,
    channel_last: bool = True
) -> np.ndarray:
    """ Rotate a 3D array 180 degrees clockwise.

    Args:
        arr (np.ndarray): The array to rotate.
    
    Keyword Args:
        channel_last (bool=True): Whether the last axis is the channel axis.

    Returns:
        np.ndarray: The rotated array.
    """
    if channel_last:
        return arr[::-1, ::-1, :]  # (H, W, C)

    return arr[:, ::-1, ::-1] # (C, H, W)


@jit(nopython=True, nogil=True, cache=True, fastmath=True, inline='always')
def rotate_270(
    arr: np.ndarray,
    channel_last: bool = True
) -> np.ndarray:
    """ 
    Rotate a 3D image array 270 degrees clockwise.

    Args:
        arr (np.ndarray): The array to rotate.

    Keyword Args:
        channel_last (bool=True): Whether the last axis is the channel axis.

    Returns:
        np.ndarray: The rotated array.
    """
    if channel_last:
        return arr[:, ::-1, :].transpose(1, 0, 2) # (H, W, C)

    return arr[:, :, ::-1].transpose(0, 2, 1) # (C, H, W)

@jit(nopython=True, nogil=True, cache=True, fastmath=True, inline='always')
def rotate_arr(
    arr: np.ndarray,
    k: int,
    channel_last: bool = True,
) -> np.ndarray:
    """ Rotate an array by 90 degrees intervals clockwise.
    
    Args:
        arr (np.ndarray): The array to rotate.
        k (int): The number of 90 degree intervals to rotate by.

    Keyword Args:
        channel_last (bool=True): Whether the last axis is the channel axis.

    Returns:
        np.ndarray: The rotated array.
    """
    if k == 0:
        return arr
    if k == 1:
        return rotate_90(arr, channel_last=channel_last)
    elif k == 2:
        return rotate_180(arr, channel_last=channel_last)
    elif k == 3:
        return rotate_270(arr, channel_last=channel_last)
    else:
        raise ValueError("k should be 1, 2, or 3")


def augmentation_rotation(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    *,
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
    if np.random.rand() > chance:
        return X, y

    random_k = np.random.randint(1, 4) if k is None else k
    X_rot = rotate_arr(X, random_k, channel_last)

    if y is None:
        return X_rot, y

    y_rot = rotate_arr(y, random_k, channel_last)

    return X_rot, y_rot


def augmentation_rotation_batch(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    *,
    chance: float = 0.5,
    channel_last: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Randomly rotates images in a batch by 90 degrees intervals. Images
    can be (batch, channels, height, width) or (batch, height, width, channels).

    Args:
        X (np.ndarray): The batch of images to rotate.
    
    Keyword Args:
        y (np.ndarray/none=None): The batch of labels to rotate.
        chance (float=0.5): The chance of rotating the image.
        channel_last (bool=True): Whether the image is (channels, height, width) or (height, width, channels).
    
    Returns:
        Tuple[np.ndarray, Optional[np.ndarray]]: The rotated images and optionally labels.
    """
    X_rot = np.zeros_like(X, dtype=X.dtype)
    y_rot = np.zeros_like(y, dtype=y.dtype) if y is not None else None

    for i in range(X.shape[0]):
        img_x = X[i, :, :, :]
        img_y = y[i, :, :, :] if y is not None else None

        if y is None:
            X_rot[i, :, :, :], _ = augmentation_rotation(
                img_x,
                y=None,
                chance=chance,
                channel_last=channel_last,
            )

        else:
            X_rot[i, :, :, :], y_rot[i, :, :, :] = augmentation_rotation(
                img_x,
                y=img_y,
                chance=chance,
                channel_last=channel_last,
            )

    return X_rot, y_rot


def augmentation_mirror(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    *,
    chance: float = 0.5,
    channel_last: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Randomly mirrors the image. Images can be (channels, height, width) or (height, width, channels).

    Args:
        X (np.ndarray): The image to mirror.

    Keyword Args:
        y (np.ndarray/none=None): The label to mirror.
        chance (float=0.5): The chance of mirroring the image.
        channel_last (bool=True): Whether the image is (channels, height, width) or (height, width, channels).

    Returns:
        Tuple[np.ndarray, Optional[np.ndarray]]: The mirrored image and optionally the label.
    """
    if np.random.rand() > chance:
        return X, y

    random_k = np.random.randint(1, 3)

    flipped_x = X.copy()
    flipped_y = None if y is None else y.copy()

    axis = 1 if channel_last else 2
    axis = axis - 1 if random_k == 2 else axis

    flipped_x = np.flip(flipped_x, axis=axis)
    flipped_y = np.flip(flipped_y, axis=axis) if y is not None else None

    return flipped_x, flipped_y


def augmentation_mirror_batch(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    *,
    chance: float = 0.5,
    channel_last: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Randomly mirrors images in a batch. Images can be (batch, channels, height, width) or (batch, height, width, channels).

    Args:
        X (np.ndarray): The batch of images to mirror.
    
    Keyword Args:
        y (np.ndarray/none=None): The label to mirror.
        chance (float=0.5): The chance of mirroring the image.
        channel_last (bool=True): Whether the image is (channels, height, width) or (height, width, channels).
    
    Returns:
        Tuple[np.ndarray, Optional[np.ndarray]]: The mirrored images and optionally labels.
    """
    X_mirror = np.zeros_like(X, dtype=X.dtype)
    y_mirror = np.zeros_like(y, dtype=y.dtype) if y is not None else None

    for i in range(X.shape[0]):
        if y is None:
            X_mirror[i], _ = augmentation_mirror(X[i], y=None, chance=chance, channel_last=channel_last)
        else:
            X_mirror[i], y_mirror[i] = augmentation_mirror(X[i], y=y[i], chance=chance, channel_last=channel_last)

    return X_mirror, y_mirror


def augmentation_noise(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    *,
    chance: float = 0.5,
    amount: float = 0.025,
    additive: bool = True,
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
        amount (float=0.01): The amount of noise to add.
        additive (bool=True): Whether to add or multiply the noise.
        channel_last (bool=True): Whether the image is (channels, height, width) or (height, width, channels).
            ignored for this function. Kept to keep the same function signature as other augmentations.

    Returns:
        Tuple[np.ndarray, Optional[np.ndarray]]: The image with noise and optionally the unmodified label.
    """
    if np.random.rand() > chance:
        return X, y

    if additive:
        noise = X + np.random.normal(0, amount, X.shape)
    else:
        noise = X * np.random.normal(1, amount, X.shape)

    return noise, y


def augmentation_noise_batch(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    *,
    chance: float = 0.5,
    amount: float = 0.025,
    additive: bool = True,
    channel_last: Any = None, # pylint: disable=unused-argument
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Adds random noise seperately to each channel of the batch of images. The noise works
    for both channel first and last images.
    input should be (batch, height, width, channels) or (batch, channels, height, width).

    Args:
        X (np.ndarray): The batch of images to add noise to.
    
    Keyword Args:
        y (np.ndarray/none=None): The label to add noise to. If None, no label is returned.
        chance (float=0.5): The chance of adding noise.
        amount (float=0.01): The amount of noise to add.
        additive (bool=True): Whether to add or multiply the noise.
        channel_last (any=None): Whether the image is (channels, height, width) or (height, width, channels).
            ignored for this function. Kept to keep the same function signature as other augmentations.


    Returns:
        Tuple[np.ndarray, Optional[np.ndarray]]: The batch of images with noise and optionally the unmodified label.
    """
    return augmentation_noise(
        X,
        y=y,
        chance=chance,
        amount=amount,
        additive=additive,
    )


def augmentation_channel_scale(
    X: np.ndarray,
    *,
    y: Optional[np.ndarray] = None,
    chance: float = 0.5,
    amount: float = 0.025,
    additive: bool = False,
    channel_last: bool = True, # pylint: disable=unused-argument
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Scales the channels of the image seperately by a fixed amount.
    input should be (height, width, channels) or (channels, height, width).

    Args:
        X (np.ndarray): The image to scale the channels of.
    
    Keyword Args:
        y (np.ndarray/none=None): The label to scale the channels of. If None, no label is returned.
        chance (float=0.5): The chance of scaling the channels.
        amount (float=0.01): The amount to scale the channels by.
        additive (bool=False): Whether to add or multiply the scaling.
        channel_last (bool=True): Whether the image is (channels, height, width) or (height, width, channels).

    Returns:
        Tuple[np.ndarray, Optional[np.ndarray]]: The image with scaled channels and optionally the unmodified label.
    """
    if np.random.rand() > chance:
        return X, y

    x = X.copy()

    if channel_last:
        for i in range(X.shape[2]):
            if additive:
                random_amount = np.random.uniform(-amount, amount)
                x[:, :, i] += random_amount
            else:
                random_amount = np.random.uniform(1 - amount, 1 + amount)
                x[:, :, i] *= random_amount
    else:
        for i in range(X.shape[0]):
            if additive:
                random_amount = np.random.uniform(-amount, amount)
                x[i, :, :] += random_amount
            else:
                random_amount = np.random.uniform(1 - amount, 1 + amount)
                x[i, :, :] *= random_amount

    return x, y


def augmentation_channel_scale_batch(
    X: np.ndarray,
    *,
    y: Optional[np.ndarray] = None,
    chance: float = 0.5,
    amount: float = 0.025,
    additive: bool = False,
    channel_last: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Scales the channels of the batch of images seperately by a fixed amount.
    input should be (batch, height, width, channels) or (batch, channels, height, width).

    Args:
        X (np.ndarray): The batch of images to scale the channels of.
    
    Keyword Args:
        y (np.ndarray/none=None): The label to scale the channels of. If None, no label is returned.
        chance (float=0.5): The chance of scaling the channels.
        amount (float=0.01): The amount to scale the channels by.
        additive (bool=False): Whether to add or multiply the scaling.
        channel_last (bool=True): Whether the image is (channels, height, width) or (height, width, channels).

    Returns:
        Tuble[np.ndarray, Optional[np.ndarray]]: The batch of images with scaled channels and optionally the unmodified label.
    """
    X_scaled = np.zeros_like(X, dtype=X.dtype)

    for i in range(X.shape[0]):
        X_scaled[i], _ = augmentation_channel_scale(
            X[i],
            y=y[i] if y is not None else None,
            chance=chance,
            amount=amount,
            additive=additive,
            channel_last=channel_last,
        )

    return X_scaled, y


def augmentation_contrast(
    X: np.ndarray,
    *,
    y: Optional[np.ndarray] = None,
    chance: float = 0.5,
    contrast_factor: float = 0.025,
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
        contrast_factor (float=0.01): The amount to change the contrast by.
        channel_last (bool=True): Whether the image is (channels, height, width) or (height, width, channels).

    Returns:
        Tuple[np.ndarray, Optional[np.ndarray]]: The image with changed contrast and optionally the unmodified label.
    """
    if np.random.rand() > chance:
        return X, y

    x = X.copy()

    channels = X.shape[2] if channel_last else X.shape[0]
    mean_pixel = np.mean(X, axis=(0, 1)) if channel_last else np.mean(X, axis=(1, 2))

    for i in range(channels):
        x[:, :, i] = (x[:, :, i] - mean_pixel[i]) * (1 + contrast_factor) + mean_pixel[i]

    return x, y


def augmentation_contrast_batch(
    X: np.ndarray,
    *,
    y: Optional[np.ndarray] = None,
    chance: float = 0.5,
    contrast_factor: float = 0.025,
    channel_last: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Changes the contrast of a batch of images by a random amount, seperately for each channel.
    input should be (batch, height, width, channels) or (batch, channels, height, width).

    Args:
        X (np.ndarray): The batch of images to change the contrast of.
    
    Keyword Args:
        y (np.ndarray/none=None): The label to change the contrast of. If None, no label is returned.
        chance (float=0.5): The chance of changing the contrast.
        contrast_factor (float=0.01): The amount to change the contrast by.
        channel_last (bool=True): Whether the image is (channels, height, width) or (height, width, channels).

    Returns:
        Tuble[np.ndarray, Optional[np.ndarray]]: The batch of images with changed contrast and optionally the unmodified label.
    """
    X_contrast = np.zeros_like(X, dtype=X.dtype)

    for i in range(X.shape[0]):
        X_contrast[i], _ = augmentation_contrast(
            X[i],
            y=y,
            chance=chance,
            contrast_factor=contrast_factor,
            channel_last=channel_last,
        )

    return X_contrast, y


def augmentation_drop_pixel(
    X: np.ndarray,
    *,
    y: Optional[np.ndarray] = None,
    chance: float = 0.5,
    drop_probability: float = 0.05,
    drop_value: float = 0.0,
    channel_last: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Drops a random pixels from an image.
    input should be (height, width, channels) or (channels, height, width).

    Args:
        X (np.ndarray): The image to drop a pixel from.
    
    Keyword Args:
        y (np.ndarray/none=None): The label to drop a pixel from. If None, no label is returned.
        chance (float=0.5): The chance of dropping a pixel.
        drop_value (float=0.0): The value to drop the pixel to.
        channel_last (bool=True): Whether the image is (channels, height, width) or (height, width, channels).

    Returns:
        Tuple[np.ndarray, Optional[np.ndarray]]: The image with the dropped pixels and optionally the unmodified label.
    """
    if np.random.rand() > chance:
        return X, y

    x = X.copy()

    if channel_last:
        height, width, channels = x.shape
    else:
        channels, height, width = x.shape

    mask = np.random.random(size=(height, width, channels))

    x[mask < drop_probability] = drop_value

    return x, y


def augmentation_drop_pixel_batch(
    X: np.ndarray,
    *,
    y: Optional[np.ndarray] = None,
    chance: float = 0.5,
    drop_probability: float = 0.05,
    drop_value: float = 0.0,
    channel_last: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Drops a random pixels from a batch of images.
    input should be (batch, height, width, channels) or (batch, channels, height, width).

    Args:
        X (np.ndarray): The batch of images to drop a pixel from.
    
    Keyword Args:
        y (np.ndarray/none=None): The label to drop a pixel from. If None, no label is returned.
        chance (float=0.5): The chance of dropping a pixel.
        drop_probability (float=0.05): The probability of dropping a pixel.
        drop_value (float=0.0): The value to drop the pixel to.
        channel_last (bool=True): Whether the image is (channels, height, width) or (height, width, channels).

    Returns:
        Tuple[np.ndarray, Optional[np.ndarray]]: The batch of images with the dropped pixels and optionally the unmodified label.
    """
    X_dropped = np.zeros_like(X, dtype=X.dtype)

    for i in range(X.shape[0]):
        X_dropped[i], _ = augmentation_drop_pixel(
            X[i],
            y=y,
            chance=chance,
            drop_probability=drop_probability,
            drop_value=drop_value,
            channel_last=channel_last,
        )

    return X_dropped, y


def augmentation_drop_channel(
    X: np.ndarray,
    *,
    y: Optional[np.ndarray] = None,
    chance: float = 0.5,
    drop_probability: 0.1,
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

    x = X.copy()

    channels = X.shape[2] if channel_last else X.shape[0]

    drop_a_channel = False
    for _ in range(channels):
        if np.random.rand() < drop_probability:
            drop_a_channel = True
            break

    if not drop_a_channel:
        return x, y

    channel_to_drop = np.random.randint(0, channels)

    if channel_last:
        x[:, :, channel_to_drop] = drop_value
    else:
        x[channel_to_drop, :, :] = drop_value

    return x, y


def augmentation_drop_channel_batch(
    X: np.ndarray,
    *,
    y: Optional[np.ndarray] = None,
    chance: float = 0.5,
    drop_probability: float = 0.1,
    drop_value: float = 0.0,
    channel_last: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Drops a random channel from a batch of images.
    input should be (batch, height, width, channels) or (batch, channels, height, width).
    A maximum of one channel will be dropped.

    Args:
        X (np.ndarray): The batch of images to drop a channel from.

    Keyword Args:
        y (np.ndarray/none=None): The label to drop a channel from. If None, no label is returned.
        chance (float=0.5): The chance of dropping a channel.
        drop_probability (float=0.1): The probability of dropping a channel.
        drop_value (float=0.0): The value to drop the channel to.
        channel_last (bool=True): Whether the image is (channels, height, width) or (height, width, channels).
    """
    X_dropped = np.zeros_like(X, dtype=X.dtype)

    for i in range(X.shape[0]):
        X_dropped[i], _ = augmentation_drop_channel(
            X[i],
            y=y,
            chance=chance,
            drop_probability=drop_probability,
            drop_value=drop_value,
            channel_last=channel_last,
        )

    return X_dropped, y


def augmentation_blur(
    X: np.ndarray,
    *,
    y: Optional[np.ndarray] = None,
    chance: float = 0.5,
    intensity: float = 1.0,
    channel_last: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Blurs an image at random.
    input should be (height, width, channels) or (channels, height, width).

    Args:
        X (np.ndarray): The image to potentially blur.

    Keyword Args:
        y (np.ndarray/none=None): The label to blur a pixel in. If None, no label is returned.
        chance (float=0.5): The chance of blurring a pixel.
        intensity (float=1.0): The intensity of the blur. from 0.0 to 1.0.
        channel_last (bool=True): Whether the image is (channels, height, width) or (height, width, channels).
    """
    if np.random.rand() > chance:
        return X, y

    x = X.copy().astype(np.float32, copy=False)

    offsets, weights = blur_kernel()

    if channel_last:
        for channel in range(x.shape[2]):
            x[:, :, channel] = convolution_simple(x[:, :, channel], offsets, weights, intensity)
    else:
        for channel in range(x.shape[0]):
            x[channel, :, :] = convolution_simple(x[channel, :, :], offsets, weights, intensity)

    return x, y


def augmentation_blur_batch(
    X: np.ndarray,
    *,
    y: Optional[np.ndarray] = None,
    chance: float = 0.5,
    intensity: float = 1.0,
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
        intensity (float=1.0): The intensity of the blur. from 0.0 to 1.0.
        channel_last (bool=True): Whether the image is (channels, height, width) or (height, width, channels).
    """
    X_blurred = np.zeros_like(X, dtype=X.dtype)

    for i in range(X.shape[0]):
        X_blurred[i], _ = augmentation_blur(
            X[i],
            y=y,
            chance=chance,
            intensity=intensity,
            channel_last=channel_last,
        )

    return X_blurred, y


def augmentation_sharpen(
    X: np.ndarray,
    *,
    y: Optional[np.ndarray] = None,
    chance: float = 0.5,
    intensity: float = 1.0,
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
        channel_last (bool=True): Whether the image is (channels, height, width) or (height, width, channels).
    """
    if np.random.rand() > chance:
        return X, y

    x = X.copy().astype(np.float32, copy=False)

    offsets, weights = unsharp_kernel()

    if channel_last:
        for channel in range(x.shape[2]):
            x[:, :, channel] = convolution_simple(x[:, :, channel], offsets, weights, intensity)
    else:
        for channel in range(x.shape[0]):
            x[channel, :, :] = convolution_simple(x[channel, :, :], offsets, weights, intensity)

    return x, y


def augmentation_sharpen_batch(
    X: np.ndarray,
    *,
    y: Optional[np.ndarray] = None,
    chance: float = 0.5,
    intensity: float = 1.0,
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
        intensity (float=1.0): The intensity of the sharpening. from 0.0 to 1.0.
        channel_last (bool=True): Whether the image is (channels, height, width) or (height, width, channels).
    """
    X_sharpened = np.zeros_like(X, dtype=X.dtype)

    for i in range(X.shape[0]):
        X_sharpened[i], _ = augmentation_sharpen(
            X[i],
            y=y,
            chance=chance,
            intensity=intensity,
            channel_last=channel_last,
        )

    return X_sharpened, y


def augmentation_cutmix_batch(
    X: np.ndarray,
    *,
    y: Optional[np.ndarray] = None,
    chance: float = 0.5,
    max_size: float = 0.5,
    max_mixes: float = 0.2,
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
        max_mixes (float=0.2): The maximum percentage of images in a batch to mixup.
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
        int(batch_size * max_mixes),
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
            feather_weight_source, feather_weight_target = feather_box(x[idx_target, :, :, 0], bbox, feather_dist)

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


def augmentation_mixup_batch(
    X: np.ndarray,
    *,
    y: Optional[np.ndarray] = None,
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


def apply_augmentations(
    batch_x: np.ndarray,
    batch_y: Optional[np.ndarray] = None,
    *,
    augmentations: Optional[List[dict]] = None,
    channel_last: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Apply a list of augmentations to a batch of images.
    
    Args:
        batch_x (np.ndarray): The batch of images to augment.
        batch_y (np.ndarray/None=None): The batch of labels to augment.

    Keyword Args:
        augmentations (list/dict=None): The list of augmentations to apply.
        channel_last (bool=True): Whether the image is (channels, height, width) or (height, width, channels).

    Returns:
        Tuple(np.ndarray, np.ndarray): The augmented batch of images and labels (if provided).
    """

    if augmentations is None:
        return batch_x, batch_y

    batch_x, batch_y = batch_x.copy(), batch_y.copy()

    for aug in augmentations:
        if aug["name"] == "rotation":
            kwargs = {}
            if "chance" in aug:
                kwargs["chance"] = aug["chance"]
            if "channel_last" in aug:
                kwargs["channel_last"] = aug["channel_last"]
            else:
                kwargs["channel_last"] = channel_last

            batch_x, batch_y = augmentation_rotation_batch(
                batch_x,
                y=batch_y,
                **kwargs,
            )

        elif aug["name"] == "mirror":
            kwargs = {}
            if "chance" in aug:
                kwargs["chance"] = aug["chance"]
            if "channel_last" in aug:
                kwargs["channel_last"] = aug["channel_last"]
            else:
                kwargs["channel_last"] = channel_last

            batch_x, batch_y = augmentation_mirror_batch(
                batch_x,
                y=batch_y,
                **kwargs,
            )

        elif aug["name"] == "channel_scale":
            kwargs = {}
            if "chance" in aug:
                kwargs["chance"] = aug["chance"]
            if "amount" in aug:
                kwargs["amount"] = aug["amount"]
            if "additive" in aug:
                kwargs["additive"] = aug["additive"]
            if "channel_last" in aug:
                kwargs["channel_last"] = aug["channel_last"]
            else:
                kwargs["channel_last"] = channel_last

            batch_x, batch_y = augmentation_channel_scale_batch(
                batch_x,
                y=batch_y,
                **kwargs,
            )

        elif aug["name"] == "noise":
            kwargs = {}
            if "chance" in aug:
                kwargs["chance"] = aug["chance"]
            if "amount" in aug:
                kwargs["amount"] = aug["amount"]
            if "additive" in aug:
                kwargs["additive"] = aug["additive"]
            if "channel_last" in aug:
                kwargs["channel_last"] = aug["channel_last"]
            else:
                kwargs["channel_last"] = channel_last

            batch_x, batch_y = augmentation_noise_batch(
                batch_x,
                y=batch_y,
                **kwargs,
            )

        elif aug["name"] == "contrast":
            kwargs = {}
            if "chance" in aug:
                kwargs["chance"] = aug["chance"]
            if "contrast_factor" in aug:
                kwargs["contrast_factor"] = aug["contrast_factor"]
            if "channel_last" in aug:
                kwargs["channel_last"] = aug["channel_last"]
            else:
                kwargs["channel_last"] = channel_last

            batch_x, batch_y = augmentation_contrast_batch(
                batch_x,
                y=batch_y,
                **kwargs,
            )

        elif aug["name"] == "drop_pixel":
            kwargs = {}
            if "chance" in aug:
                kwargs["chance"] = aug["chance"]
            if "drop_probability" in aug:
                kwargs["drop_probability"] = aug["drop_probability"]
            if "drop_value" in aug:
                kwargs["drop_value"] = aug["drop_value"]
            if "channel_last" in aug:
                kwargs["channel_last"] = aug["channel_last"]
            else:
                kwargs["channel_last"] = channel_last

            batch_x, batch_y = augmentation_drop_pixel_batch(
                batch_x,
                y=batch_y,
                **kwargs,
            )

        elif aug["name"] == "drop_channel":
            kwargs = {}
            if "chance" in aug:
                kwargs["chance"] = aug["chance"]
            if "drop_probability" in aug:
                kwargs["drop_probability"] = aug["drop_probability"]
            if "drop_value" in aug:
                kwargs["drop_value"] = aug["drop_value"]
            if "channel_last" in aug:
                kwargs["channel_last"] = aug["channel_last"]
            else:
                kwargs["channel_last"] = channel_last

            batch_x, batch_y = augmentation_drop_channel_batch(
                batch_x,
                y=batch_y,
                **kwargs,
            )

        elif aug["name"] == "blur":
            kwargs = {}
            if "chance" in aug:
                kwargs["chance"] = aug["chance"]
            if "intensity" in aug:
                kwargs["intensity"] = aug["intensity"]
            if "channel_last" in aug:
                kwargs["channel_last"] = aug["channel_last"]
            else:
                kwargs["channel_last"] = channel_last

            batch_x, batch_y = augmentation_blur_batch(
                batch_x,
                y=batch_y,
                **kwargs,
            )

        elif aug["name"] == "sharpen":
            kwargs = {}
            if "chance" in aug:
                kwargs["chance"] = aug["chance"]
            if "intensity" in aug:
                kwargs["intensity"] = aug["intensity"]
            if "channel_last" in aug:
                kwargs["channel_last"] = aug["channel_last"]
            else:
                kwargs["channel_last"] = channel_last

            batch_x, batch_y = augmentation_sharpen_batch(
                batch_x,
                y=batch_y,
                **kwargs,
            )

        elif aug["name"] == "cutmix":
            kwargs = {}
            if "chance" in aug:
                kwargs["chance"] = aug["chance"]
            if "max_size" in aug:
                kwargs["max_size"] = aug["max_size"]
            if "max_mixes" in aug:
                kwargs["max_mixes"] = aug["max_mixes"]
            if "feather" in aug:
                kwargs["feather"] = aug["feather"]
            if "feather_dist" in aug:
                kwargs["feather_dist"] = aug["feather_dist"]
            if "channel_last" in aug:
                kwargs["channel_last"] = aug["channel_last"]
            else:
                kwargs["channel_last"] = channel_last

            batch_x, batch_y = augmentation_cutmix_batch(
                batch_x,
                y=batch_y,
                **kwargs,
            )

        elif aug["name"] == "mixup":
            kwargs = {}
            if "chance" in aug:
                kwargs["chance"] = aug["chance"]
            if "max_mixes" in aug:
                kwargs["max_mixes"] = aug["max_mixes"]
            if "channel_last" in aug:
                kwargs["channel_last"] = aug["channel_last"]
            else:
                kwargs["channel_last"] = channel_last

            batch_x, batch_y = augmentation_mixup_batch(
                batch_x,
                y=batch_y,
                **kwargs,
            )

    return batch_x, batch_y


def augmentation_generator(
    X: np.ndarray,
    *,
    y: Optional[np.ndarray] = None,
    batch_size: int = 64,
    augmentations: Optional[List[dict]] = None,
    shuffle: bool = True,
    seed: Optional[int] = None,
    channel_last: bool = True,
) -> np.ndarray:
    """
    Generate batches of augmented data.
    
    Args:
        X (np.ndarray): The data to augment.

    Keyword Args:
        y (np.ndarray): The labels for the data.
        batch_size (int): The size of the batches to generate.
        augmentations (list): The augmentations to apply.
        shuffle (bool): Whether to shuffle the data before generating batches.
        seed (int): The seed to use for shuffling.
        channel_last (bool): Whether the data is in channel last format.

    Returns:
        A generator yielding batches of augmented data.
    """
    if seed is not None:
        np.random.seed(seed)

    if shuffle:
        idx = np.random.permutation(X.shape[0])
        X = X[idx]
        y = y[idx]

    num_batches = (X.shape[0] + batch_size - 1) // batch_size

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, X.shape[0])

        batch_x = X[start_idx:end_idx]
        batch_y = y[start_idx:end_idx]

        batch_augmented_x, batch_augmented_y = apply_augmentations(
            batch_x,
            batch_y,
            augmentations=augmentations,
            channel_last=channel_last,
        )

        yield batch_augmented_x, batch_augmented_y


class AugmentationDataset:
    """
    A dataset that applies augmentations to the data.

    Args:
        X (np.ndarray): The data to augment.

    Keyword Args:
        y (np.ndarray): The labels for the data.
        batch_size (int): The size of the batches to generate.
        augmentations (list): The augmentations to apply.
        shuffle (bool): Whether to shuffle the data before generating batches.
            and whenever the dataset is iterated over.
        seed (int): The seed to use for shuffling.
        channel_last (bool): Whether the data is in channel last format.
    
    Returns:
        A dataset yielding batches of augmented data. For Pytorch,
            convert the batches to tensors before ingestion.
    """
    def __init__(
        self,
        X: np.ndarray,
        *,
        y: Optional[np.ndarray] = None,
        batch_size: int = 64,
        augmentations: Optional[List[dict]] = None,
        shuffle: bool = True,
        seed: Optional[int] = None,
        channel_last: bool = True,
    ):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.augmentations = augmentations
        self.shuffle = shuffle
        self.seed = seed
        self.channel_last = channel_last
        self.generator = None

    def __iter__(self):
        self.generator = augmentation_generator(
            self.X,
            y=self.y,
            batch_size=self.batch_size,
            augmentations=self.augmentations,
            shuffle=self.shuffle,
            seed=self.seed,
            channel_last=self.channel_last,
        )
        return self

    def __next__(self):
        try:
            return next(self.generator)
        except StopIteration as e:
            raise StopIteration from e

    def __len__(self):
        num_batches = (self.X.shape[0] + self.batch_size - 1) // self.batch_size

        return num_batches
