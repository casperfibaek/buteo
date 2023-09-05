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
    radius: int = 2, *,
    classes: Optional[np.array] = None,
    method: Optional[str] = "half",
    variance: Optional[np.ndarray] = None,
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

    method : str, optional
        Determines the flip protection method, default: 'half'.
        kernel: The class is protected by ensuring that the weight of the center class is always at least the sum of the surrounding weights.
        half: The class is protected by ensuring that it is always at least 50% + strength of the weight of the classes in the neighbourhood.
        max: The class is protected by ensuring that it is always at at least the max + strength of the strongest class in the neighbourhood.
        None: No flip protection.

    channel_last : bool, optional
        Whether the image is (channels, height, width) or (height, width, channels), default: True.
    
    Returns
    np.ndarray
        The smoothed array. One channel for each class in ascending order. Float32.
    """
    if classes is None:
        classes = np.unique(arr)

    if channel_last:
        classes_shape = np.array(classes).reshape(1, 1, -1)
    else:
        classes_shape = np.array(classes).reshape(-1, 1, 1)

    classes_hot = (arr == classes_shape).astype(np.uint8)

    if variance is not None:
        classes_arr = classes_hot * variance
    else:
        classes_arr = classes_hot

    kernel = kernel_base(
        radius=radius,
        circular=True,
        distance_weighted=True,
        normalised=False,
        hole=True if method is not None else False,
        method=3,
        sigma=2,
    )

    strength = np.float32(kernel.size / (kernel.size - 1.0))

    if method == "kernel":
        kernel[kernel.shape[0] // 2, kernel.shape[1] // 2] = kernel.sum() * strength

    feathered = filter_operation(classes_arr, 1, radius=radius, normalised=False, kernel=kernel, channel_last=channel_last)

    if method == "max":
        if channel_last:
            maxval = np.max(feathered, axis=2, keepdims=True)
            argmax = np.argmax(feathered, axis=2, keepdims=True)
        else:
            maxval = np.max(feathered, axis=0, keepdims=True)
            argmax = np.argmax(feathered, axis=0, keepdims=True)

        argmax_hot = np.argmax(classes_hot, axis=2, keepdims=True)

        weight = np.where(argmax == argmax_hot, feathered, maxval * strength)
        dst = np.where(classes_hot == 1, weight, feathered)
    
    elif method == "half":
        weight = kernel.sum() * strength 
        dst = np.where(classes_hot == 1, weight, feathered)

    elif method is None or method == "kernel":
        dst = feathered

    else:
        raise ValueError("Invalid method.")
    
    dst = dst / np.maximum(np.sum(dst, axis=2, keepdims=True), 1e-7)

    return dst
