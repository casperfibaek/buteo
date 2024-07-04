"""### Encoding spatial values  ###"""
# Standard library
from typing import Tuple, Union, List
# External
import numpy as np
from numba import jit, prange



@jit(nopython=True, parallel=True, nogil=True, fastmath=True, inline="always")
def encode_latitude(lat: float) -> np.ndarray:
    """Encode a latitude value into a two-element numpy array of sine and cosine components.

    Parameters
    ----------
    lat : float
        The latitude value to encode. Must be in the range [-90, 90].

    Returns
    -------
    np.ndarray
        A two-element numpy array containing the sine and cosine components of the encoded latitude.
        The sine and cosine values are scaled to the range [0, 1].
    """
    lat_adj = lat + 90.0
    lat_max = 180

    encoded_sin = ((np.sin(2 * np.pi * (lat_adj / lat_max)) + 1)) / 2.0
    encoded_cos = ((np.cos(2 * np.pi * (lat_adj / lat_max)) + 1)) / 2.0

    return np.array([encoded_sin, encoded_cos], dtype=np.float32)


@jit(nopython=True, parallel=True, nogil=True, fastmath=True, inline="always")
def encode_longitude(lng: float) -> np.ndarray:
    """Encode a longitude value into a two-element numpy array of sine and cosine components.

    Parameters
    ----------
    lng : float
        The longitude value to encode. Must be in the range [-180, 180].

    Returns
    -------
    np.ndarray
        A two-element numpy array containing the sine and cosine components of the encoded longitude.
        The sine and cosine values are scaled to the range [0, 1].
    """
    lng_adj = lng + 180.0
    lng_max = 360

    encoded_sin = ((np.sin(2 * np.pi * (lng_adj / lng_max)) + 1)) / 2.0
    encoded_cos = ((np.cos(2 * np.pi * (lng_adj / lng_max)) + 1)) / 2.0

    return np.array([encoded_sin, encoded_cos], dtype=np.float32)


@jit(nopython=True, parallel=True, nogil=True, fastmath=True, inline="always")
def encode_latlng(latlng: Union[np.ndarray, Tuple[float, float], List[float]]) -> np.ndarray:
    """Encode a latitude-longitude coordinate into a four-element numpy array of sine and cosine components.

    Parameters
    ----------
    latlng : Union[np.ndarray, Tuple[float, float], List[float]]
        A tuple, list, or numpy array containing the latitude and longitude values to encode.
        The latitude value must be in the range [-90, 90] and the longitude value must be in the range [-180, 180].

    Returns
    -------
    np.ndarray
        A four-element numpy array containing the sine and cosine components of the encoded latitude and longitude.
        The sine and cosine values are scaled to the range [0, 1].
    """
    lat = latlng[0]
    lng = latlng[1]

    encoded_lat = encode_latitude(lat)
    encoded_lng = encode_longitude(lng)

    return np.concatenate((encoded_lat, encoded_lng)).astype(np.float32, copy=False)


@jit(nopython=True, parallel=True, nogil=True, fastmath=True, inline="always")
def encode_latlngs(latlngs: np.ndarray) -> np.ndarray:
    """Encode multiple latitude-longitude coordinates into a numpy array of sine and cosine components.

    Parameters
    ----------
    latlngs : np.ndarray
        A numpy array containing the latitude-longitude coordinates to encode.
        The shape of the array should be (n, 2), (m, n, 2), or (p, q, n, 2), where n=2 represents the latitude
        and longitude values. The latitude value must be in the range [-90, 90] and the longitude value must be
        in the range [-180, 180].

    Returns
    -------
    np.ndarray
        A numpy array containing the sine and cosine components of the encoded latitude and longitude values.
        The shape of the returned array will depend on the shape of the input array, with the last dimension
        being 4 (two sine and two cosine values).

    Raises
    ------
    ValueError
        If the input array has more than 3 dimensions or the shape of the array is not (n, 2), (m, n, 2),
        or (p, q, n, 2).
    """
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
def decode_latitude(
    encoded_sin: float,
    encoded_cos: float,
) -> float:
    """Decode an encoded sine and cosine value to the original latitude value.

    Parameters
    ----------
    encoded_sin : float
        The sine component of the encoded latitude value.
        Must be in the range [0, 1].

    encoded_cos : float
        The cosine component of the encoded latitude value.
        Must be in the range [0, 1].

    Returns
    -------
    float
        The original latitude value. The decoded latitude value is in the range [-90, 90].

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
def decode_longitude(
    encoded_sin: float,
    encoded_cos: float,
) -> float:
    """Decode an encoded sine and cosine value to the original longitude value.

    Parameters
    ----------
    encoded_sin : float
        The sine component of the encoded longitude value.
        Must be in the range [0, 1].

    encoded_cos : float
        The cosine component of the encoded longitude value.
        Must be in the range [0, 1].

    Returns
    -------
    float
        The original longitude value. The decoded longitude value is in the range [-180, 180].
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
def decode_latlng(encoded_latlng: Union[np.ndarray, List[Union[float, int]]]) -> np.ndarray:
    """Decode an encoded latitude-longitude coordinate to the original latitude-longitude values.

    Parameters
    ----------
    encoded_latlng : Union[np.ndarray, List[Union[float, int]]]
        A numpy array or list containing the encoded latitude-longitude coordinate to decode.
        The shape of the array should be (4,), with the first two values representing the encoded latitude
        and the last two values representing the encoded longitude. The encoded sine and cosine values
        must be in the range [0, 1].

    Returns
    -------
    np.ndarray
        A numpy array containing the original latitude and longitude values. The shape of the returned
        array will be (2,) with the latitude value at index 0 and the longitude value at index 1.
    """
    lat = decode_latitude(encoded_latlng[0], encoded_latlng[1])
    lng = decode_longitude(encoded_latlng[2], encoded_latlng[3])

    return np.array([lat, lng], dtype=np.float32)


@jit(nopython=True, parallel=True, nogil=True, fastmath=True, inline="always")
def decode_latlngs(encoded_latlngs: Union[np.ndarray, List[Union[float, int]]]) -> np.ndarray:
    """Decode multiple encoded latitude-longitude coordinates to the original latitude-longitude values.

    Parameters
    ----------
    encoded_latlngs : Union[np.ndarray, List[Union[float, int]]]
        A numpy array or list containing the encoded latitude-longitude coordinates to decode.
        The shape of the array should be (n, 4), where n represents the number of encoded coordinates.
        The first two values of each encoded coordinate represent the encoded latitude and the last two
        values represent the encoded longitude. The encoded sine and cosine values must be in the range [0, 1].

    Returns
    -------
    np.ndarray
        A numpy array containing the original latitude-longitude values. The shape of the returned
        array will be (n, 2), where n represents the number of encoded coordinates. The latitude value
        will be at index 0 and the longitude value will be at index 1.
    """
    latlngs = np.apply_along_axis(decode_latlng, 1, encoded_latlngs)
    return latlngs


@jit(nopython=True, parallel=True, nogil=True, fastmath=True, inline="always")
def encode_width(
    lng: float,
    lng_max: float,
) -> float:
    """Encode the width of a rectangle to be used as input to the model.

    Parameters
    ----------
    lng : float
        The longitude value representing the width of the rectangle.
        Must be in the range [-180, 180].

    lng_max : float
        The maximum longitude value. This is typically 360 for the full longitude range.

    Returns
    -------
    np.ndarray
        A numpy array containing the encoded width of the rectangle.
        The encoded width consists of two values: the encoded sine and cosine values.
        Both values will be in the range [0, 1].
    """
    encoded_sin = ((np.sin(2 * np.pi * (lng / lng_max)) + 1)) / 2.0
    encoded_cos = ((np.cos(2 * np.pi * (lng / lng_max)) + 1)) / 2.0

    return np.array([encoded_sin, encoded_cos], dtype=np.float32)


@jit(nopython=True, parallel=True, nogil=True, fastmath=True, inline="always")
def encode_arr_position(arr: np.ndarray) -> np.ndarray:
    """Fast encoding of a 2D numpy array of coordinates where the width is cyclical.
    Very useful for global maps.

    Parameters
    ----------
    arr : np.ndarray
        A 2D numpy array of coordinates to be encoded. The first dimension represents the columns
        and the second dimension represents the rows. The shape of the array should be (m, n), where
        m represents the number of columns and n represents the number of rows.

    Returns
    -------
    np.ndarray
        A numpy array containing the encoded coordinates. The shape of the returned array will be (m, n, 3),
        where m represents the number of columns, n represents the number of rows, and 3 represents the
        number of encoded values for each coordinate. The first two values of each encoded coordinate
        represent the encoded width, while the last value represents the encoded column.
    """
    result = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.float32)

    col_end = arr.shape[0] - 1
    row_end = arr.shape[1] - 1

    col_range = np.arange(0, arr.shape[0]).astype(np.float32)
    row_range = np.arange(0, arr.shape[1]).astype(np.float32)

    col_encoded = np.zeros((col_range.shape[0], 1), dtype=np.float32)
    row_encoded = np.zeros((row_range.shape[0], 2), dtype=np.float32)

    for col in prange(col_range.shape[0]):
        col_encoded[col, :] = col_range[col] / col_end

    for row in prange(row_range.shape[0]):
        row_encoded[row, :] = encode_width(row_range[row], row_end)

    for col in prange(arr.shape[0]):
        for row in range(arr.shape[1]):
            result[col, row, 0] = row_encoded[row, 0]
            result[col, row, 1] = row_encoded[row, 1]
            result[col, row, 2] = col_encoded[col, 0]

    return result
