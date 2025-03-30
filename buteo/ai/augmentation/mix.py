""" Mix augmentation functions. """

# Standard library
import random
from typing import Optional, Tuple

# External
import numpy as np
from numba import jit, prange


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
    """Cutmixes two images.
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


class AugmentationCutmix:
    def __init__(self, *, p: float = 1.0, min_size: float = 0.333, max_size: float = 0.666, channel_last: bool = True, inplace: bool = False):
        self.p = p
        self.min_size = min_size
        self.max_size = max_size
        self.channel_last = channel_last
        self.inplace = inplace
        self.applies_to_features = True
        self.applies_to_labels = True
        self.requires_dataset = True

    def __call__(self, X_t: np.ndarray, y_t: np.ndarray, X_s: np.ndarray, y_s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() > self.p:
            return X_t, y_t

        return augmentation_cutmix(
            X_t,
            y_t,
            X_s,
            y_s,
            min_size=self.min_size,
            max_size=self.max_size,
            channel_last=self.channel_last,
            inplace=self.inplace,
        )


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
) -> Tuple[np.ndarray, np.ndarray]:
    """Mixups two images at random. This works by doing a linear intepolation between
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


class AugmentationMixup:
    def __init__(self, *, p: float = 1.0, min_size: float = 0.333, max_size: float = 0.666, label_mix: int = 0, channel_last: bool = True, inplace: bool = False):
        self.p = p
        self.min_size = min_size
        self.max_size = max_size
        self.label_mix = label_mix
        self.channel_last = channel_last
        self.inplace = inplace
        self.applies_to_features = True
        self.applies_to_labels = True
        self.requires_dataset = True

    def __call__(self, X_t: np.ndarray, y_t: np.ndarray, X_s: np.ndarray, y_s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() > self.p:
            return X_t, y_t

        return augmentation_mixup(
            X_t,
            y_t,
            X_s,
            y_s,
            min_size=self.min_size,
            max_size=self.max_size,
            label_mix=self.label_mix,
            channel_last=self.channel_last,
            inplace=self.inplace,
        )
