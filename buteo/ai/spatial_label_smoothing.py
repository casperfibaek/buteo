"""
This module contains functions for augmenting images that are
suited to remote sensing imagery.
"""
# Standard library
import sys; sys.path.append("../../")
from typing import Optional

# External
import numpy as np

# Internal
from buteo.array.filters import filter_operation
from buteo.array.convolution_kernels import kernel_base
from numba import jit, prange


@jit(nopython=True, nogil=True, parallel=True)
def _reweight(
    arr: np.array,
    feathered: np.array,
    classes: np.array,
    strength: float,
    method: int,
    argmax: Optional[np.array] = None,
    channel_last: bool = True,
) -> np.ndarray:
    height, width, _channels = arr.shape
    if not channel_last:
        _channels, height, width = arr.shape

    class_holder = np.arange(0, classes.max() + 1) * 0
    
    for i, c in enumerate(classes):
        class_holder[c] = i

    if channel_last:
        feathered = feathered / np.expand_dims(np.sum(feathered, axis=2), axis=2)
    else:
        feathered = feathered / np.expand_dims(np.sum(feathered, axis=0), axis=0)

    for col in prange(height):
        for row in prange(width):

            if channel_last:
                cls_idx = class_holder[arr[col, row, 0]]
            else:
                cls_idx = class_holder[arr[0, col, row]]

            if method == 0:
                if channel_last:
                    feathered[col, row, cls_idx] = (feathered[col, row, cls_idx] + 1.0) * strength
                else:
                    feathered[cls_idx, col, row] = (feathered[cls_idx, col, row] + 1.0) * strength
            else:
                if channel_last:
                    max_val = feathered[col, row, argmax[col, row]]
                    feathered[col, row, cls_idx] = max_val * strength
                else:
                    max_val = feathered[argmax[col, row], col, row]
                    feathered[cls_idx, col, row] = max_val * strength

    return feathered


def spatial_label_smoothing(
    arr: np.array,
    radius: int = 2, *,
    classes: Optional[np.array] = None,
    strength: float = 1.01,
    flip_protection: bool = True,
    method: int = 1,
    channel_last: bool = True,
) -> np.ndarray:
    """
    Smoothes the labels in a landcover classification by counting the weighted
    occurances of classes in a given radius.

    Parameters
    ----------
    arr : np.ndarray
        The array to smooth. Should only contain integer values and 1 channel.

    radius : int, optional
        The radius of the smoothing kernel, default: 2.

    classes : np.array, optional
        The classes in the classification. Numpy integer list of unique values in the array if None, default: None.

    strength : float, optional
        The strength of the smoothing, default: 1.001.
        Calculated as the power of the sum of the kernel without the central pixel.
        If the value is above 1, the class, if using argmax downstream, will never be replaced by another class.

    flip_protection : bool, optional
        Whether to protect against flipping of classes, default: True.
        If True, the class will never be replaced by another class.

    method : int, optional
        Determines the flip protection method, default: 1.
        0: The class is protected by ensuring that it is always at least 50% + strength of the weight of the classes in the neighbourhood.
        1: The class is protected by ensuring that it is always at at least the max + strength of the strongest class in the neighbourhood.

    channel_last : bool, optional
        Whether the image is (channels, height, width) or (height, width, channels), default: True.
    
    Returns
    -------
    np.ndarray
        The smoothed array. One channel for each class in ascending order. Float32.
    """
    if classes is None:
        classes = np.unique(arr)

    if channel_last:
        dst = np.zeros((arr.shape[0], arr.shape[1], len(classes)), dtype=np.float32)
    else:
        dst = np.zeros((len(classes), arr.shape[0], arr.shape[1]), dtype=np.float32)

    kernel = kernel_base(
        radius=radius,
        circular=True,
        distance_weighted=True,
        normalised=False,
        hole=True if flip_protection else False,
        method=3,
        decay=0.5,
        sigma=2,
    )

    for i, c in enumerate(classes):
        feathered = filter_operation(
            arr,
            15, # count weighted occurances
            radius=radius,
            func_value=int(c),
            normalised=False,
            kernel=kernel,
            channel_last=channel_last,
        )

        if channel_last:
            dst[:, :, i] = feathered[:, :, 0]
        else:
            dst[i, :, :] = feathered[0, :, :]

    if flip_protection:
        if method == 1 and channel_last:
            argmax = np.argmax(dst, axis=2)
        elif method == 1:
            argmax = np.argmax(dst, axis=0)
        else:
            argmax = None

        dst = _reweight(arr, dst, classes, strength, method, argmax, channel_last=channel_last)

    dst = dst / np.sum(dst, axis=2, keepdims=True)

    return dst
