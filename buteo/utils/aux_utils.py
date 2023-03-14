"""
### Generic utility functions ###

Functions that make interacting with the toolbox easier.
"""

# External
import numpy as np
from numba import jit, prange



@jit(nopython=True, parallel=True, nogil=True, fastmath=True, inline="always")
def encode_latitude(lat):
    """ Latitude goes from -90 to 90 """
    lat_adj = lat + 90.0
    lat_max = 180

    encoded_sin = ((np.sin(2 * np.pi * (lat_adj / lat_max)) + 1)) / 2.0
    encoded_cos = ((np.cos(2 * np.pi * (lat_adj / lat_max)) + 1)) / 2.0

    return np.array([encoded_sin, encoded_cos], dtype=np.float32)

@jit(nopython=True, parallel=True, nogil=True, fastmath=True, inline="always")
def encode_longitude(lng):
    """ Longitude goes from -180 to 180 """
    lng_adj = lng + 180.0
    lng_max = 360

    encoded_sin = ((np.sin(2 * np.pi * (lng_adj / lng_max)) + 1)) / 2.0
    encoded_cos = ((np.cos(2 * np.pi * (lng_adj / lng_max)) + 1)) / 2.0

    return np.array([encoded_sin, encoded_cos], dtype=np.float32)

@jit(nopython=True, parallel=True, nogil=True, fastmath=True, inline="always")
def encode_latlng(latlng):
    """
    Encode latitude and longitude values to be used as input to the model.
    """
    lat = latlng[0]
    lng = latlng[1]

    encoded_lat = encode_latitude(lat)
    encoded_lng = encode_longitude(lng)

    return np.concatenate((encoded_lat, encoded_lng)).astype(np.float32)

@jit(nopython=True, parallel=True, nogil=True, fastmath=True, inline="always")
def encode_latlngs(latlngs):
    """ Encode multiple latitude and longitude values. """
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
    """
    Decode encoded latitude values to the original latitude value.
    """
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
    """
    Decode encoded longitude values to the original longitude value.
    """
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
    """
    Decode encoded latitude and longitude values to the original values.
    """
    lat = decode_latitude(encoded_latlng[0], encoded_latlng[1])
    lng = decode_longitude(encoded_latlng[2], encoded_latlng[3])

    return np.array([lat, lng], dtype=np.float32)

@jit(nopython=True, parallel=True, nogil=True, fastmath=True, inline="always")
def decode_latlngs(encoded_latlngs):
    """ Decode multiple latitude and longitude values. """
    latlngs = np.apply_along_axis(decode_latlng, 1, encoded_latlngs)
    return latlngs

def channel_first_to_last(arr):
    """ Converts a numpy array from channel first to channel last format. """
    if arr.ndim != 3:
        raise ValueError("Input array should be 3-dimensional with shape (channels, height, width)")

    # Swap the axes to change from channel first to channel last format
    arr = np.transpose(arr, (1, 2, 0))

    return arr

def channel_last_to_first(arr):
    """ Converts a numpy array from channel last to channel first format. """
    if arr.ndim != 3:
        raise ValueError("Input array should be 3-dimensional with shape (height, width, channels)")

    # Swap the axes to change from channel last to channel first format
    arr = np.transpose(arr, (2, 0, 1))

    return arr

def scale_to_range(arr, min_val, max_val):
    """ Scales the values in the input array to the specified range. """

    # Scale the values in the array to the specified range
    arr = (arr - arr.min()) / (arr.max() - arr.min())
    arr = (max_val - min_val) * arr + min_val

    return arr
