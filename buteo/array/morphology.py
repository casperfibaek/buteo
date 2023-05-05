"""
### Perform morphological operations on arrays and rasters.  ###
"""

# Standard library
import sys; sys.path.append("../../")
from typing import Union


# External
import numpy as np

# Internal
from buteo.array.convolution import convolve_array
from buteo.array.convolution_kernels import kernel_base, kernel_get_offsets_and_weights
from buteo.utils.utils_base import _type_check


def _morphology_operation(
    arr: np.ndarray,
    method: int,
    radius: Union[int, float] = 1,
    spherical: bool = True,
    channel_last: bool = True,
) -> np.ndarray:
    _type_check(arr, [np.ndarray], "arr")
    _type_check(method, [int], "method")
    _type_check(radius, [int, float], "radius")
    _type_check(spherical, [bool], "spherical")
    _type_check(channel_last, [bool], "channel_last")

    kernel = kernel_base(radius, spherical=spherical, normalise=False)
    offsets, weights = kernel_get_offsets_and_weights(kernel)

    mask = None
    if np.ma.isMaskedArray(arr):
        nodata = True
        nodata_value = arr.fill_value
        arr = np.ma.getdata(arr)
        mask = np.ma.getmask(arr)
    else:
        nodata = False
        nodata_value = 0.0

    arr_convolved = convolve_array(
        arr,
        offsets,
        weights,
        method=method,
        nodata=nodata,
        nodata_value=nodata_value,
        channel_last=channel_last,
    )

    if nodata:
        arr_convolved = np.ma.array(arr_convolved, mask=mask, fill_value=nodata_value)

    return arr_convolved


def morph_erode(
    arr: np.ndarray,
    radius: Union[int, float] = 1,
    spherical: bool = True,
    channel_last: bool = True,
) -> np.ndarray:
    """
    Erode an array by taking the local minimum.
    
    Parameters
    ----------
    arr : np.ndarray
        Input array.

    radius : Union[int, float], optional
        Radius of the kernel. Default: 1 (3x3 kernel)

    spherical : bool, optional
        Use a spherical kernel. Default: True

    channel_last : bool, optional
        If True, the channel axis is the last axis. Default: True

    Returns
    -------
    np.ndarray
        Eroded array.
    """
    return _morphology_operation(
        arr,
        method="min",
        radius=radius,
        spherical=spherical,
        channel_last=channel_last,
    )


def morph_dilate(
    arr: np.ndarray,
    radius: Union[int, float] = 1,
    spherical: bool = True,
    channel_last: bool = True,
) -> np.ndarray:
    """
    Dilate an array by taking the local maximum.
    
    Parameters
    ----------
    arr : np.ndarray
        Input array.

    radius : Union[int, float], optional
        Radius of the kernel. Default: 1 (3x3 kernel)

    spherical : bool, optional
        Use a spherical kernel. Default: True

    channel_last : bool, optional
        If True, the channel axis is the last axis. Default: True

    Returns
    -------
    np.ndarray
        Dilated array.
    """
    return _morphology_operation(
        arr,
        method="max",
        radius=radius,
        spherical=spherical,
        channel_last=channel_last,
    )


def morph_open(
    arr: np.ndarray,
    radius: Union[int, float] = 1,
    spherical: bool = True,
    channel_last: bool = True,
) -> np.ndarray:
    """
    Perform the open mortholigical operation on an array.
    
    Parameters
    ----------
    arr : np.ndarray
        Input array.

    radius : Union[int, float], optional
        Radius of the kernel. Default: 1 (3x3 kernel)

    spherical : bool, optional
        Use a spherical kernel. Default: True

    channel_last : bool, optional
        If True, the channel axis is the last axis. Default: True

    Returns
    -------
    np.ndarray
        Opened array.
    """
    erode = morph_erode(
        arr,
        radius=radius,
        spherical=spherical,
        channel_last=channel_last,
    )

    return morph_dilate(
        erode,
        radius=radius,
        spherical=spherical,
        channel_last=channel_last,
    )


def morph_close(
    arr: np.ndarray,
    radius: Union[int, float] = 1,
    spherical: bool = True,
    channel_last: bool = True,
) -> np.ndarray:
    """
    Perform the close morphological operation on an array.

    Parameters
    ----------
    arr : np.ndarray
        Input array.

    radius : Union[int, float], optional
        Radius of the kernel. Default: 1 (3x3 kernel)

    spherical : bool, optional
        Use a spherical kernel. Default: True

    channel_last : bool, optional
        If True, the channel axis is the last axis. Default: True

    Returns
    -------
    np.ndarray
        Closed array.
    """
    dilate = morph_dilate(
        arr,
        radius=radius,
        spherical=spherical,
        channel_last=channel_last,
    )

    return morph_erode(
        dilate,
        radius=radius,
        spherical=spherical,
        channel_last=channel_last,
    )


def morph_tophat(
    arr: np.ndarray,
    radius: Union[int, float] = 1,
    spherical: bool = True,
    channel_last: bool = True,
) -> np.ndarray:
    """
    Perform the top_hat morphological operation on an array.

    Same as: `array / opened(array)`

    Parameters
    ----------
    arr : np.ndarray
        Input array.

    radius : Union[int, float], optional
        Radius of the kernel. Default: 1 (3x3 kernel)

    spherical : bool, optional
        Use a spherical kernel. Default: True

    channel_last : bool, optional
        If True, the channel axis is the last axis. Default: True

    Returns
    -------
    np.ndarray
        TopHat array.
    """
    opened = morph_open(
        arr,
        radius=radius,
        spherical=spherical,
        channel_last=channel_last,
    )

    return arr / opened


def morph_bothat(
    arr: np.ndarray,
    radius: Union[int, float] = 1,
    spherical: bool = True,
    channel_last: bool = True,
) -> np.ndarray:
    """
    Perform the bottom_hat morphological operation on an array.
    
    Same as: `closed(array) / array`

    Parameters
    ----------
    arr : np.ndarray
        Input array.

    radius : Union[int, float], optional
        Radius of the kernel. Default: 1 (3x3 kernel)

    spherical : bool, optional
        Use a spherical kernel. Default: True

    channel_last : bool, optional
        If True, the channel axis is the last axis. Default: True

    Returns
    -------
    np.ndarray
        BotHat array.
    """
    closed = morph_close(
        arr,
        radius=radius,
        spherical=spherical,
        channel_last=channel_last,
    )

    return closed / arr


def morph_difference(
    arr: np.ndarray,
    radius: Union[int, float] = 1,
    spherical: bool = True,
    channel_last: bool = True,
) -> np.ndarray:
    """
    Perform the difference morphological operation on an array.
    
    Same as: `dilate(array) - erode(array)`

    Parameters
    ----------
    arr : np.ndarray
        Input array.

    radius : Union[int, float], optional
        Radius of the kernel. Default: 1 (3x3 kernel)

    spherical : bool, optional
        Use a spherical kernel. Default: True

    channel_last : bool, optional
        If True, the channel axis is the last axis. Default: True

    Returns
    -------
    np.ndarray
        Difference array.
    """
    erode = morph_erode(
        arr,
        radius=radius,
        spherical=spherical,
        channel_last=channel_last,
    )
    dilate = morph_dilate(
        arr,
        radius=radius,
        spherical=spherical,
        channel_last=channel_last,
    )

    return dilate - erode
