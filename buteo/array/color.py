"""### Functions for interacting with colour. ### """

# Standard library
from typing import Tuple

# External
import numpy as np
from numba import jit, prange

# TOOD: Change order of channels

@jit(nopython=True, fastmath=True, cache=True, nogil=True, inline='always')
def _single_hue_to_rgb(
    p: float,
    q: float,
    t: float,
) -> float:
    """Helper function to convert hue to RGB.

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
    hue: float,
    saturation: float,
    lightness: float,
) -> Tuple[float, float, float]:
    """Convert a single HSL color to RGB.

    Parameters
    ----------
    hue : float
        Hue component.

    saturation : float
        Saturation component.

    lightness : float
        Lightness component.

    Returns
    -------
    Tuple[float, float, float]
        Tuple of RGB values (r, g, b).
    """
    if saturation == 0:
        return lightness, lightness, lightness

    q = lightness * (1 + saturation) if lightness < 0.5 else lightness + saturation - lightness * saturation
    p = 2 * lightness - q

    r = _single_hue_to_rgb(p, q, hue + 1/3)
    g = _single_hue_to_rgb(p, q, hue)
    b = _single_hue_to_rgb(p, q, hue - 1/3)

    return r, g, b


@jit(nopython=True, parallel=True, fastmath=True, cache=True, nogil=True)
def color_hsl_to_rgb(hsl_array: np.ndarray) -> np.ndarray:
    """
    Convert an HSL array to an RGB array with shape (3, height, width).

    Parameters
    ----------
    hsl_array : np.ndarray
        Input HSL array with shape (3, height, width).

    Returns
    -------
    np.ndarray
        Output RGB array with shape (3, height, width).
    """
    assert hsl_array.ndim == 3, "Input array must have 3 dimensions"
    assert hsl_array.shape[0] == 3, "First dimension must be channels"
    assert hsl_array.min() >= 0 and hsl_array.max() <= 1, "Input array must be normalized"

    c, height, width = hsl_array.shape
    rgb_array = np.empty_like(hsl_array, dtype=np.float32)

    for i in prange(height):
        for j in prange(width):
            h, s, l = hsl_array[0, i, j], hsl_array[1, i, j], hsl_array[2, i, j]
            r, g, b = _single_hsl_to_rgb(h, s, l)
            rgb_array[0, i, j] = r
            rgb_array[1, i, j] = g
            rgb_array[2, i, j] = b

    return rgb_array


@jit(nopython=True, parallel=True, fastmath=True, cache=True, nogil=True)
def color_rgb_to_hsl(rgb_array: np.ndarray) -> np.ndarray:
    """Convert an RGB array to an HSL array.

    Parameters
    ----------
    rgb_array : np.ndarray
        Input RGB array with shape (3, height, width).

    Returns
    -------
    np.ndarray
        Output HSL array with shape (3, height, width).
    """
    assert rgb_array.ndim == 3, "Input array must have 3 dimensions"
    c, height, width = rgb_array.shape
    assert c == 3, "Input array must have 3 channels in the first dimension"
    assert rgb_array.min() >= 0 and rgb_array.max() <= 1, "Input array must be normalized"

    cmin = np.zeros((height, width), dtype=rgb_array.dtype)
    cmax = np.zeros((height, width), dtype=rgb_array.dtype)

    for i in prange(height):
        for j in prange(width):
            cmin[i, j] = min(rgb_array[0, i, j], rgb_array[1, i, j], rgb_array[2, i, j])
            cmax[i, j] = max(rgb_array[0, i, j], rgb_array[1, i, j], rgb_array[2, i, j])

    delta = cmax - cmin
    luminosity = (cmax + cmin) / 2
    hue = np.zeros((height, width), dtype=rgb_array.dtype)
    saturation = np.zeros((height, width), dtype=rgb_array.dtype)

    red, green, blue = rgb_array[0], rgb_array[1], rgb_array[2]
    for i in prange(height):
        for j in prange(width):
            if delta[i, j] != 0:
                saturation[i, j] = delta[i, j] / (1 - abs(2 * luminosity[i, j] - 1))
                if cmax[i, j] == red[i, j]:
                    hue[i, j] = (green[i, j] - blue[i, j]) / delta[i, j] % 6
                elif cmax[i, j] == green[i, j]:
                    hue[i, j] = (blue[i, j] - red[i, j]) / delta[i, j] + 2
                else:
                    hue[i, j] = (red[i, j] - green[i, j]) / delta[i, j] + 4
                hue[i, j] = (hue[i, j] * 60) % 360
                if hue[i, j] < 0:
                    hue[i, j] += 360

    hue /= 360

    hsl_array = np.zeros_like(rgb_array)
    for i in prange(height):
        for j in prange(width):
            hsl_array[0, i, j] = hue[i, j]
            hsl_array[1, i, j] = saturation[i, j]
            hsl_array[2, i, j] = luminosity[i, j]

    np.clip(hsl_array, 0.0, 1.0, out=hsl_array)

    return hsl_array
