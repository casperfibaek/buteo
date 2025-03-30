""" Label augmentation functions. """

# Standard library
import random
from typing import Optional, Tuple

# External
import numpy as np
from numba import jit


@jit(nopython=True, nogil=True, cache=True, fastmath=True)
def augmentation_label_smoothing(
    y: np.ndarray,
    flat_array: bool = False,
    max_amount: float = 0.1,
    fixed_amount: bool = False,
    channel_last: bool = True,
    inplace: bool = False,
) -> np.ndarray:
    """Smooths the labels of an image at random. Input should be (height, width, channels) or (channels, height, width).
    The label is blurred by a random amount.

    NOTE: Beware of datatypes. Consider casting to float32 before adding noise.

    Parameters
    ----------
    y : np.ndarray
        The label to smooth.

    flat_array : bool, optional
        Whether the array is flat (1D) or not, default: False.

    max_amount : float, optional
        The maximum amount to smooth the label by, default: 0.1.

    fixed_amount : bool, optional
        If True, uses max_amount directly instead of random amount, default: False.

    channel_last : bool, optional
        Whether the image is (channels, height, width) or (height, width, channels), ignored for this function.
        Kept to keep the same function signature as other augmentations, default: True.

    inplace : bool, optional
        Whether to perform the rotation in-place, default: False.

    Returns
    -------
    np.ndarray
        The smoothed label.
    """
    if not inplace:
        y = y.copy()

    if fixed_amount:
        amount = max_amount
    else:
        amount = np.random.rand() * max_amount
    
    if flat_array:
        mean = np.mean(y)
        y = ((1 - amount) * y) + (amount * mean)
    else:
        if channel_last:
            mean = np.mean(y, axis=2, keepdims=True)
            y = ((1 - amount) * y) + (amount * mean)
        else:
            mean = np.mean(y, axis=0, keepdims=True)
            y = ((1 - amount) * y) + (amount * mean)

    return y


class AugmentationLabelSmoothing:
    def __init__(self, *, p: float = 1.0, flat_array: bool = False, max_amount: float = 0.1, fixed_amount: bool = False, channel_last: bool = True, inplace: bool = False):
        self.p = p
        self.flat_array = flat_array
        self.max_amount = max_amount
        self.fixed_amount = fixed_amount
        self.channel_last = channel_last
        self.inplace = inplace
        self.applies_to_features = False
        self.applies_to_labels = True
        self.requires_dataset = False

    def __call__(self, y: np.ndarray) -> np.ndarray:
        if random.random() > self.p:
            return y

        return augmentation_label_smoothing(
            y,
            flat_array=self.flat_array,
            max_amount=self.max_amount,
            fixed_amount=self.fixed_amount,
            channel_last=self.channel_last,
            inplace=self.inplace,
        )
