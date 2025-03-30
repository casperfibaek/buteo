"""### Perform morphological operations on arrays and rasters.  ###"""

# Standard library
from typing import Union

# External
import numpy as np

# Internal
from buteo.array.convolution import convolve_array, kernel_base, kernel_get_offsets_and_weights
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

    kernel = kernel_base(radius, circular=spherical, normalised=False)
    offsets, weights = kernel_get_offsets_and_weights(kernel)

    mask = None
    if np.ma.isMaskedArray(arr):
        nodata = True
        nodata_value = float(arr.fill_value)  # Convert to native float
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

    if nodata and mask is not None:
        # Create a mask for the result that matches the original mask
        result_mask = np.zeros_like(arr_convolved, dtype=bool)
        
        # Make sure the mask shape matches
        if mask.shape == result_mask.shape:
            result_mask = mask.copy()
        else:
            # Handle possible shape differences (if any dimension was broadcast)
            if mask.ndim == result_mask.ndim:
                # Copy mask elements that match positions
                for idx in np.ndindex(mask.shape):
                    if idx < result_mask.shape:
                        if mask[idx]:
                            result_mask[idx] = True
            else:
                # If ndims don't match, at least preserve the corner element if it was masked
                if mask.size > 0 and mask.flat[0]:
                    result_mask.flat[0] = True
        
        # Apply the mask
        arr_convolved = np.ma.array(arr_convolved, mask=result_mask, fill_value=nodata_value)

    return arr_convolved

# 1. sum 2. max 3. min 4. mean 5. median 6. variance 7. standard deviation 8. contrast 9. mode 10. median absolute deviation (mad) 11. z-score 12. z-score (mad) 13. sigma lee 14. quantile 15. occurances 16. feather 17. roughness 18. roughness tri 19. roughness tpi
def morph_erode(
    arr: np.ndarray,
    radius: Union[int, float] = 1,
    spherical: bool = True,
    channel_last: bool = True,
) -> np.ndarray:
    """Erode an array by taking the local minimum.

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
        method=3,
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
    """Dilate an array by taking the local maximum.

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
        method=2,
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
    """Perform the open mortholigical operation on an array.

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
    """Perform the close morphological operation on an array.

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
    """Perform the top_hat morphological operation on an array.

    Same as: `array - opened(array)`
    
    This highlights small bright features that are removed by opening.

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
    
    # Standard implementation: arr - opened
    return arr - opened


def morph_bothat(
    arr: np.ndarray,
    radius: Union[int, float] = 1,
    spherical: bool = True,
    channel_last: bool = True,
) -> np.ndarray:
    """Perform the bottom_hat morphological operation on an array.

    Same as: `closed(array) - array`
    
    This highlights small dark features that are filled in by closing.

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
    
    # Standard implementation: closed - arr
    return closed - arr


def morph_difference(
    arr: np.ndarray,
    radius: Union[int, float] = 1,
    spherical: bool = True,
    channel_last: bool = True,
) -> np.ndarray:
    """Perform the difference morphological operation on an array.

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
