"""
This module contains utility functions for augmenting images that are
suited to remote sensing imagery.
"""

# External
import numpy as np
from numba import jit



@jit(nopython=True, nogil=True, cache=True, fastmath=True, inline='always')
def _rotate_90(
    arr: np.ndarray,
    channel_last: bool = True
) -> np.ndarray:
    """
    Rotate a 3D array 90 degrees clockwise.

    Parameters
    ----------
    arr : np.ndarray
        The array to rotate.

    channel_last : bool, optional
        Whether the last axis is the channel axis, default: True.

    Returns
    -------
    np.ndarray
        The rotated array.
    """
    if channel_last:
        return arr[::-1, :, :].transpose(1, 0, 2) # (H, W, C)

    return arr[:, ::-1, :].transpose(0, 2, 1) # (C, H, W)


@jit(nopython=True, nogil=True, cache=True, fastmath=True, inline='always')
def _rotate_180(
    arr: np.ndarray,
    channel_last: bool = True
) -> np.ndarray:
    """
    Rotate a 3D array 180 degrees clockwise.

    Parameters
    ----------
    arr : np.ndarray
        The array to rotate.

    channel_last : bool, optional
        Whether the last axis is the channel axis, default: True.

    Returns
    -------
    np.ndarray
        The rotated array.
    """
    if channel_last:
        return arr[::-1, ::-1, :]  # (H, W, C)

    return arr[:, ::-1, ::-1] # (C, H, W)


@jit(nopython=True, nogil=True, cache=True, fastmath=True, inline='always')
def _rotate_270(
    arr: np.ndarray,
    channel_last: bool = True
) -> np.ndarray:
    """ 
    Rotate a 3D image array 270 degrees clockwise.

    Parameters
    ----------
    arr : np.ndarray
        The array to rotate.
        
    channel_last : bool, optional
        Whether the last axis is the channel axis, default: True.

    Returns
    -------
    np.ndarray
        The rotated array.
    """
    if channel_last:
        return arr[:, ::-1, :].transpose(1, 0, 2) # (H, W, C)

    return arr[:, :, ::-1].transpose(0, 2, 1) # (C, H, W)


@jit(nopython=True, nogil=True, cache=True, fastmath=True, inline='always')
def _mirror_horizontal(
    arr: np.ndarray,
    channel_last: bool = True
) -> np.ndarray:
    """
    Mirror a 3D array horizontally.

    Parameters
    ----------
    arr : np.ndarray
        The array to mirror.

    channel_last : bool, optional
        Whether the last axis is the channel axis. default: True

    Returns
    -------
    np.ndarray
        The mirrored array.
    """
    if channel_last:
        return arr[:, ::-1, :] # (H, W, C)

    return arr[:, :, ::-1] # (C, H, W)


@jit(nopython=True, nogil=True, cache=True, fastmath=True, inline='always')
def _mirror_vertical(
    arr: np.ndarray,
    channel_last: bool = True
) -> np.ndarray:
    """
    Mirror a 3D array vertically.

    Parameters
    ----------
    arr : np.ndarray
        The array to mirror.

    channel_last : bool, optional
        Whether the last axis is the channel axis. Default: True

    Returns
    -------
    np.ndarray
        The mirrored array.
    """
    if channel_last:
        return arr[::-1, :, :] # (H, W, C)

    return arr[:, ::-1, :] # (C, H, W)


@jit(nopython=True, nogil=True, cache=True, fastmath=True, inline='always')
def _mirror_horisontal_vertical(
    arr: np.ndarray,
    channel_last: bool = True
) -> np.ndarray:
    """
    Mirror a 3D array horizontally and vertically

    Parameters
    ----------
    arr : np.ndarray
        The array to mirror.

    channel_last : bool, optional
        Whether the last axis is the channel axis. Default: True

    Returns
    -------
    np.ndarray
        The mirrored array.
    """
    if channel_last:
        return arr[::-1, ::-1, :] # (H, W, C)

    return arr[:, ::-1, ::-1] # (C, H, W)


@jit(nopython=True, nogil=True, cache=True, fastmath=True, inline='always')
def _mirror_arr(
    arr: np.ndarray,
    k: int,
    channel_last: bool = True,
) -> np.ndarray:
    """
    Mirror an array horizontally and/or vertically.
    Returns a view to the mirrored array.

    Parameters
    ----------
    arr : np.ndarray
        The array to mirror.

    k : int
        1 for horizontal, 2 for vertical, 3 for both, 0 for no mirroring.

    channel_last : bool, optional
        Whether the last axis is the channel axis, default: True

    Returns
    -------
    np.ndarray
        The view to the mirrored array.
    """
    if k == 1:
        view = _mirror_horizontal(arr, channel_last=channel_last)
    elif k == 2:
        view = _mirror_vertical(arr, channel_last=channel_last)
    elif k == 3:
        view = _mirror_horisontal_vertical(arr, channel_last=channel_last)
    else:
        view = arr

    return view


@jit(nopython=True, nogil=True, cache=True, fastmath=True, inline='always')
def _rotate_arr(
    arr: np.ndarray,
    k: int,
    channel_last: bool = True,
) -> np.ndarray:
    """
    Rotate an array by 90 degrees intervals clockwise.
    Returns a view.

    Parameters
    ----------
    arr : numpy.ndarray
        The array to rotate.

    k : int
        The number of 90 degree intervals to rotate by. 1 for 90 degrees, 2 for 180 degrees, 3 for 270 degrees, and 0 for no rotation.

    channel_last : bool, optional
        Whether the last axis is the channel axis. Default is True.

    Returns
    -------
    numpy.ndarray
        The view to the rotated array.
    """
    if k == 1:
        view = _rotate_90(arr, channel_last=channel_last)
    elif k == 2:
        view = _rotate_180(arr, channel_last=channel_last)
    elif k == 3:
        view = _rotate_270(arr, channel_last=channel_last)
    else:
        view = arr

    return view
