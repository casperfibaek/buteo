"""
### Perform morphological operations on arrays and rasters.  ###
"""

# Standard library
import sys; sys.path.append("../../")
from typing import Union

# External
import numpy as np

# Internal
from buteo.array.convolution import convolve_array, pad_array
from buteo.array.convolution_kernels import get_kernel_sobel, get_offsets_and_weights


def edge_detection(
    arr: np.ndarray,
    radius: Union[int, float] = 1,
    scale: Union[int, float] = 1,
    gradient: bool = False,
    channel_last: bool = True,
) -> np.ndarray:
    """
    Perform edge detection on an array using a Sobel-style operator.

    Parameters
    ----------
    arr : np.ndarray
        Input array.

    radius : Union[int, float], optional
        Radius of the kernel. Default: 1 (3x3 kernel)
        The radius can be fractional.

    scale : Union[int, float], optional
        scale of the edge detection. Increase to increase the
        sensitivity of the edge detection. This can also be fractional.
        Default: 1.0

    gradient : bool, optional
        Return the gradient as well as the magnitude, default: False.

    Returns
    -------
    np.ndarray
        Edge detection result.
    """
    arr = arr.astype(np.float32, copy=False)
    mask = None
    if np.ma.isMaskedArray(arr):
        nodata = True
        nodata_value = arr.fill_value
        arr = np.ma.getdata(arr)
        mask = np.ma.getmask(arr)
    else:
        nodata = False
        nodata_value = 0.0

    kernel_gx, kernel_gy = get_kernel_sobel(radius, scale)

    offsets_gx, weights_gx = get_offsets_and_weights(kernel_gx)
    offsets_gy, weights_gy = get_offsets_and_weights(kernel_gy)

    pad_size = kernel_gx.shape[0] // 2
    padded = pad_array(arr, pad_size)

    arr_gx = convolve_array(
        padded,
        offsets_gx,
        weights_gx,
        nodata=nodata,
        nodata_value=nodata_value,
        channel_last=channel_last,
    )

    arr_gy = convolve_array(
        padded,
        offsets_gy,
        weights_gy,
        nodata=nodata,
        nodata_value=nodata_value,
        channel_last=channel_last,
    )

    magnitude = np.sqrt(arr_gx ** 2 + arr_gy ** 2)
    magnitude = magnitude[pad_size:-pad_size, pad_size:-pad_size]

    if nodata:
        magnitude = np.ma.array(magnitude, mask=mask, fill_value=nodata_value)

    if gradient:
        gradient = np.arctan2(arr_gy, arr_gx)
        gradient = gradient[pad_size:-pad_size, pad_size:-pad_size]

        return magnitude, gradient

    return magnitude
