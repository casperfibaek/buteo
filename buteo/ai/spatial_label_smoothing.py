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


def spatial_label_smoothing(
    arr: np.array,
    radius: int = 2,
    classes: Optional[np.array] = None,
    strength: float = 1.001,
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
        hole=True,
        method=3,
        decay=0.5,
        sigma=2,
    )

    kernel[kernel.shape[0] // 2, kernel.shape[1] // 2] = kernel.sum() ** strength

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

    if channel_last:
        dst = dst / np.sum(dst, axis=2, keepdims=True)
    else:
        dst = dst / np.sum(dst, axis=0, keepdims=True)

    return dst
