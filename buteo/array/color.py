"""
### Generic utility functions ###

Functions that make interacting with the toolbox easier.
"""
# Internal
from typing import Tuple

# External
import numpy as np
from numba import jit, prange


@jit(nopython=True, fastmath=True, cache=True, nogil=True, inline='always')
def _single_hue_to_rgb(
    p: float,
    q: float,
    t: float,
) -> float:
    """
    Helper function to convert hue to RGB.

    Parameters
    ----------
    p : float
        Intermediate value used for hue to RGB conversion.

    q : float
        Intermediate value used for hue to RGB conversion.

    t : float
        Hue value.

    Returns
    -------
    float
        RGB value.
    """
    if t < 0:
        t += 1
    if t > 1:
        t -= 1
    if t < 1/6:
        return p + (q - p) * 6 * t
    if t < 1/2:
        return q
    if t < 2/3:
        return p + (q - p) * (2/3 - t) * 6
    return p


@jit(nopython=True, fastmath=True, cache=True, nogil=True, inline='always')
def _single_hsl_to_rgb(
    h: float,
    s: float,
    l: float,
) -> Tuple[float, float, float]:
    """
    Convert a single HSL color to RGB.

    Parameters
    ----------
    h : float
        Hue component.

    s : float
        Saturation component.

    l : float
        Lightness component.

    Returns
    -------
    Tuple[float, float, float]
        Tuple of RGB values (r, g, b).
    """
    if s == 0:
        return l, l, l

    q = l * (1 + s) if l < 0.5 else l + s - l * s
    p = 2 * l - q

    r = _single_hue_to_rgb(p, q, h + 1/3)
    g = _single_hue_to_rgb(p, q, h)
    b = _single_hue_to_rgb(p, q, h - 1/3)

    return r, g, b


@jit(nopython=True, parallel=True, fastmath=True, cache=True, nogil=True)
def color_hsl_to_rgb(hsl_array: np.ndarray) -> np.ndarray:
    """
    Convert an HSL array to an RGB array.

    Parameters
    ----------
    hsl_array : np.ndarray
        Input HSL array with shape (height, width, 3).

    Returns
    -------
    np.ndarray
        Output RGB array with shape (height, width, 3).
    """
    assert hsl_array.ndim == 3, "Input array must have 3 dimensions"
    assert hsl_array.shape[-1] == 3, "Input array must have 3 channels"
    assert hsl_array.min() >= 0 and hsl_array.max() <= 1, "Input array must be normalized"

    shape = hsl_array.shape

    rgb_array = np.empty(shape, dtype=np.float32)
    for i in prange(shape[0]):
        for j in prange(shape[1]):
            h, s, l = hsl_array[i, j]

            r, g, b = _single_hsl_to_rgb(h, s, l)

            if hsl_array.ndim == 3:
                rgb_array[i, j, 0] = r
                rgb_array[i, j, 1] = g
                rgb_array[i, j, 2] = b
            else:
                rgb_array[i, j] = [r, g, b]

    return rgb_array


@jit(nopython=True, parallel=True, fastmath=True, cache=True, nogil=True)
def color_rgb_to_hsl(rgb_array: np.ndarray) -> np.ndarray:
    """
    Convert an RGB array to an HSL array.
    
    Parameters
    ----------
    rgb_array : np.ndarray
        Input RGB array with shape (height, width, 3).
    
    Returns
    -------
    np.ndarray
        Output HSL array with shape (height, width, 3).
    """
    assert rgb_array.ndim == 3, "Input array must have 3 dimensions"
    assert rgb_array.shape[-1] == 3, "Input array must have 3 channels"
    assert rgb_array.min() >= 0 and rgb_array.max() <= 1, "Input array must be normalized"

    # Get the shape of the input array
    shape = rgb_array.shape

    # Initialize the minimum and maximum arrays
    cmin = np.zeros((shape[0], shape[1]))
    cmax = np.zeros((shape[0], shape[1]))

    # Calculate the minimum and maximum of the RGB values for each pixel
    for i in prange(shape[0]):
        for j in prange(shape[1]):
            cmin[i, j] = np.min(rgb_array[i, j, :])
            cmax[i, j] = np.max(rgb_array[i, j, :])

    # Calculate the difference of the RGB values
    delta = cmax - cmin

    # Initialize the HSL arrays
    hue = np.zeros((shape[0], shape[1]))
    saturation = np.zeros((shape[0], shape[1]))
    luminosity = (cmax + cmin) / 2

    # Initialize the HSL array
    hsl_array = np.zeros((shape[0], shape[1], 3))

    red, green, blue = rgb_array[..., 0], rgb_array[..., 1], rgb_array[..., 2]

    for i in prange(shape[0]):
        for j in prange(shape[1]):
            if delta[i, j] != 0:
                saturation[i, j] = delta[i, j] / (1 - np.abs(2 * luminosity[i, j] - 1))

                if cmax[i, j] == red[i, j]:
                    hue[i, j] = (green[i, j] - blue[i, j]) / delta[i, j] % 6
                elif cmax[i, j] == green[i, j]:
                    hue[i, j] = (blue[i, j] - red[i, j]) / delta[i, j] + 2
                elif cmax[i, j] == blue[i, j]:
                    hue[i, j] = (red[i, j] - green[i, j]) / delta[i, j] + 4

                hue[i, j] = (hue[i, j] * 60) % 360
                if hue[i, j] < 0:
                    hue[i, j] += 360

    # Normalize the hue value to [0, 1]
    hue /= 360

    # Assign the h, s, and l values to the HSL array
    hsl_array[..., 0] = hue
    hsl_array[..., 1] = saturation
    hsl_array[..., 2] = luminosity

    hsl_array = np.clip(hsl_array, 0.0, 1.0)

    return hsl_array
