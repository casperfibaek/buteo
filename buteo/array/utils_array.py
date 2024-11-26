"""### Generic utility functions for arrays. ### """

# Standard Library
from typing import Tuple

# External
import numpy as np
from numba import jit, prange




@jit(nopython=True)
def _create_grid(range_rows, range_cols):
    """Create a grid of rows and columns.

    Parameters
    ----------
    range_rows : np.ndarray
        The rows to create the grid from.

    range_cols : np.ndarray
        The columns to create the grid from.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The rows and columns grid.
    """
    rows_grid = np.zeros((len(range_rows), len(range_cols)), dtype=np.int64)
    cols_grid = np.zeros((len(range_rows), len(range_cols)), dtype=np.int64)

    for i in range(len(range_rows)):
        for j in range(len(range_cols)):
            cols_grid[i, j] = range_rows[j]
            rows_grid[i, j] = range_cols[i]

    return rows_grid, cols_grid


@jit(nopython=True, parallel=True, nogil=True, fastmath=True, inline="always")
def encode_latitude(lat):
    """Latitude goes from -90 to 90"""
    lat_adj = lat + 90.0
    lat_max = 180

    encoded_sin = ((np.sin(2 * np.pi * (lat_adj / lat_max)) + 1)) / 2.0
    encoded_cos = ((np.cos(2 * np.pi * (lat_adj / lat_max)) + 1)) / 2.0

    return np.array([encoded_sin, encoded_cos], dtype=np.float32)

@jit(nopython=True, parallel=True, nogil=True, fastmath=True, inline="always")
def encode_longitude(lng):
    """Longitude goes from -180 to 180"""
    lng_adj = lng + 180.0
    lng_max = 360

    encoded_sin = ((np.sin(2 * np.pi * (lng_adj / lng_max)) + 1)) / 2.0
    encoded_cos = ((np.cos(2 * np.pi * (lng_adj / lng_max)) + 1)) / 2.0

    return np.array([encoded_sin, encoded_cos], dtype=np.float32)

@jit(nopython=True, parallel=True, nogil=True, fastmath=True, inline="always")
def encode_latlng(latlng):
    """Encode latitude and longitude values to be used as input to the model."""
    lat = latlng[0]
    lng = latlng[1]

    encoded_lat = encode_latitude(lat)
    encoded_lng = encode_longitude(lng)

    return np.concatenate((encoded_lat, encoded_lng)).astype(np.float32)

@jit(nopython=True, parallel=True, nogil=True, fastmath=True, inline="always")
def encode_latlngs(latlngs):
    """Encode multiple latitude and longitude values."""
    if latlngs.ndim == 1:
        encoded_latlngs = np.apply_along_axis(encode_latlng, 0, latlngs)
    elif latlngs.ndim == 2:
        encoded_latlngs = np.apply_along_axis(encode_latlng, 1, latlngs)
    elif latlngs.ndim == 3:
        rows = latlngs.shape[0]
        cols = latlngs.shape[1]

        output_shape = (rows, cols, 4)
        encoded_latlngs = np.zeros(output_shape, dtype=np.float32)

        for i in prange(rows):
            for j in range(cols):
                latlng = latlngs[i, j]
                encoded_latlngs[i, j] = encode_latlng(latlng)
    else:
        raise ValueError(
            f"The input array must have 1, 2 or 3 dimensions, not {latlngs.ndim}"
        )

    return encoded_latlngs


@jit(nopython=True, parallel=True, nogil=True, fastmath=True, inline="always")
def decode_latitude(encoded_sin, encoded_cos):
    """Decode encoded latitude values to the original latitude value."""
    lat_max = 180
    lat_max_half = lat_max / 2.0

    # Calculate the sin and cos values from the encoded values
    sin_val = (2 * encoded_sin) - 1
    cos_val = (2 * encoded_cos) - 1

    # Calculate the latitude adjustment
    lat_adj = np.arctan2(sin_val, cos_val)

    # Convert the adjusted latitude to the original latitude value
    sign = np.sign(lat_adj)
    sign_adj = np.where(sign == 0, 1, sign) * lat_max_half

    lat = ((lat_adj / (2 * np.pi)) * lat_max) - sign_adj
    lat = np.where(lat == -lat_max_half, lat_max_half, lat)

    return lat

@jit(nopython=True, parallel=True, nogil=True, fastmath=True, inline="always")
def decode_longitude(encoded_sin, encoded_cos):
    """Decode encoded longitude values to the original longitude value."""
    lng_max = 360
    lng_max_half = lng_max / 2.0

    # Calculate the sin and cos values from the encoded values
    sin_val = (2 * encoded_sin) - 1
    cos_val = (2 * encoded_cos) - 1

    # Calculate the longitude adjustment
    lng_adj = np.arctan2(sin_val, cos_val)

    # Convert the adjusted longitude to the original longitude value
    sign = np.sign(lng_adj)
    sign_adj = np.where(sign == 0, 1, sign) * lng_max_half

    lng = ((lng_adj / (2 * np.pi)) * lng_max) - sign_adj

    lng = np.where(lng == -lng_max_half, lng_max_half, lng)

    return lng


@jit(nopython=True, parallel=True, nogil=True, fastmath=True, inline="always")
def decode_latlng(encoded_latlng):
    """Decode encoded latitude and longitude values to the original values."""
    lat = decode_latitude(encoded_latlng[0], encoded_latlng[1])
    lng = decode_longitude(encoded_latlng[2], encoded_latlng[3])

    return np.array([lat, lng], dtype=np.float32)

@jit(nopython=True, parallel=True, nogil=True, fastmath=True, inline="always")
def decode_latlngs(encoded_latlngs):
    """Decode multiple latitude and longitude values."""
    latlngs = np.apply_along_axis(decode_latlng, 1, encoded_latlngs)
    return latlngs


@jit(nopython=True, parallel=True, nogil=True, fastmath=True, inline="always")
def encode_width(lng, lng_max):
    """Longitude goes from -180 to 180"""

    encoded_sin = ((np.sin(2 * np.pi * (lng / lng_max)) + 1)) / 2.0
    encoded_cos = ((np.cos(2 * np.pi * (lng / lng_max)) + 1)) / 2.0

    return np.array([encoded_sin, encoded_cos], dtype=np.float32)


@jit(nopython=True, fastmath=True, cache=True, nogil=True, inline='always')
def single_hue_to_rgb(
    p: float,
    q: float,
    t: float,
) -> float:
    """Helper function to convert hue to RGB.

    Args:
        p (float): Intermediate value used for hue to RGB conversion.
        q (float): Intermediate value used for hue to RGB conversion.
        t (float): Hue value.

    Returns:
        float: RGB value.
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
def single_hsl_to_rgb(
    h: float,
    s: float,
    l: float,
) -> Tuple[float, float, float]:
    """Convert a single HSL color to RGB.

    Args:
        h (float): Hue component.
        s (float): Saturation component.
        l (float): Lightness component.

    Returns:
        Tuple[float, float, float]: Tuple of RGB values (r, g, b).
    """
    if s == 0:
        return l, l, l

    q = l * (1 + s) if l < 0.5 else l + s - l * s
    p = 2 * l - q

    r = single_hue_to_rgb(p, q, h + 1/3)
    g = single_hue_to_rgb(p, q, h)
    b = single_hue_to_rgb(p, q, h - 1/3)

    return r, g, b


@jit(nopython=True, parallel=True, fastmath=True, cache=True, nogil=True)
def hsl_to_rgb(hsl_array: np.ndarray) -> np.ndarray:
    """Convert an HSL array to an RGB array.

    Args:
        hsl_array (np.ndarray): Input HSL array with shape (height, width, 3).

    Returns:
        np.ndarray: Output RGB array with shape (height, width, 3).
    """
    assert hsl_array.ndim == 3, "Input array must have 3 dimensions"
    assert hsl_array.shape[-1] == 3, "Input array must have 3 channels"
    assert hsl_array.min() >= 0 and hsl_array.max() <= 1, "Input array must be normalized"

    shape = hsl_array.shape

    rgb_array = np.empty(shape, dtype=np.float32)
    for i in prange(shape[0]):
        for j in range(shape[1]):
            h, s, l = hsl_array[i, j]

            r, g, b = single_hsl_to_rgb(h, s, l)

            if hsl_array.ndim == 3:
                rgb_array[i, j, 0] = r
                rgb_array[i, j, 1] = g
                rgb_array[i, j, 2] = b
            else:
                rgb_array[i, j] = [r, g, b]

    return rgb_array


@jit(nopython=True, parallel=True, fastmath=True, cache=True, nogil=True)
def rgb_to_hsl(rgb_array: np.ndarray) -> np.ndarray:
    """Convert an RGB array to an HSL array.

        Args:
       rgb_array (np.ndarray): Input RGB array with shape (height, width, 3).

        Returns:
       np.ndarray: Output HSL array with shape (height, width, 3).
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
