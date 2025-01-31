"""### Generic utility functions for arrays. ### """

# External
import numpy as np
from numba import jit, prange



@jit(nopython=True, nogil=True, fastmath=True, inline="always")
def encode_latitude(lat: float) -> np.ndarray:
    """
    Encodes a latitude value into a 2D periodic representation using sine and cosine.
    This function transforms a latitude value from the range [-90, 90] into two periodic
    values using trigonometric encoding. This is useful for machine learning applications
    where periodic features need to be represented continuously.

    Parameters
    ----------
    lat : float
        The latitude value to encode, must be between -90 and 90 degrees.

    Returns
    -------
    np.ndarray:
        A 2D array of shape (2,) containing the encoded values:
            - [0]: Sine component normalized to [0,1]
            - [1]: Cosine component normalized to [0,1]
        The array is of type np.float32.

    Examples
    --------
    >>> encode_latitude(45.0)
    array([0.8535534, 0.8535534], dtype=float32)
    >>> encode_latitude(-90.0)
    array([0.5, 0.0], dtype=float32)
    """
    lat_adj = lat + 90.0
    lat_max = 180

    encoded_sin = ((np.sin(2 * np.pi * (lat_adj / lat_max)) + 1)) / 2.0
    encoded_cos = ((np.cos(2 * np.pi * (lat_adj / lat_max)) + 1)) / 2.0

    return np.array([encoded_sin, encoded_cos], dtype=np.float32)

@jit(nopython=True, nogil=True, fastmath=True, inline="always")
def encode_longitude(lng: float) -> np.ndarray:
    """
    Encodes a longitude value into a two-dimensional normalized array using sine and cosine.
    Takes a longitude value in degrees and transforms it into two values between 0 and 1
    using sine and cosine functions. This encoding preserves the circular nature of longitude
    coordinates and avoids discontinuity at the -180/180 boundary.

    Parameters
    ----------
    lng : float
        The longitude value to encode, in degrees (-180 to 180)

    Returns
    -------
    np.ndarray
        A 2D array containing the encoded values:
            - [0]: Sine component normalized to [0,1]
            - [1]: Cosine component normalized to [0,1]
        Array is of type np.float32
    """
    lng_adj = lng + 180.0
    lng_max = 360

    encoded_sin = ((np.sin(2 * np.pi * (lng_adj / lng_max)) + 1)) / 2.0
    encoded_cos = ((np.cos(2 * np.pi * (lng_adj / lng_max)) + 1)) / 2.0

    return np.array([encoded_sin, encoded_cos], dtype=np.float32)

@jit(nopython=True, nogil=True, fastmath=True, inline="always")
def encode_latlng(latlng: np.ndarray) -> np.ndarray:
    """
    Encode latitude and longitude values to be used as input to the model.
    
    Parameters
    ----------
    latlng : np.ndarray
        A numpy array containing latitude and longitude values.
    
    Returns
    -------
    np.ndarray
        A numpy array containing the encoded latitude and longitude values as float32.
    """
    lat = latlng[0]
    lng = latlng[1]

    encoded_lat = encode_latitude(lat)
    encoded_lng = encode_longitude(lng)

    return np.concatenate((encoded_lat, encoded_lng)).astype(np.float32)

@jit(nopython=True, nogil=True, fastmath=True, inline="always")
def encode_latlngs(latlngs: np.ndarray) -> np.ndarray:
    """
    Encode multiple latitude and longitude values.

    Parameters
    ----------
    latlngs : np.ndarray
        A numpy array of shape (n, 2) where n is the number of latitude and longitude pairs.

    Returns
    -------
    np.ndarray
        A numpy array of shape (n, 4) containing the encoded latitude and longitude values.
    """
    encoded_latlngs = np.zeros((latlngs.shape[0], 4), dtype=np.float32)
    for i in prange(latlngs.shape[0]):
        encoded_latlngs[i] = encode_latlng(latlngs[i])

    return encoded_latlngs

@jit(nopython=True, nogil=True, fastmath=True, inline="always")
def decode_latitude(encoded_sin: float, encoded_cos: float) -> np.ndarray:
    """
    Decode encoded latitude values to the original latitude value.

    Parameters
    ----------
    encoded_sin : float
        The sine component of the encoded latitude.
    encoded_cos : float
        The cosine component of the encoded latitude.

    Returns
    -------
    np.ndarray
        The decoded latitude values.
    """
    lat_max = 180
    lat_max_half = lat_max / 2.0

    sin_val = (2 * encoded_sin) - 1
    cos_val = (2 * encoded_cos) - 1

    lat_adj = np.arctan2(sin_val, cos_val)

    sign = np.sign(lat_adj)
    sign_adj = np.where(sign == 0, 1, sign) * lat_max_half

    lat = ((lat_adj / (2 * np.pi)) * lat_max) - sign_adj
    lat = np.where(lat == -lat_max_half, lat_max_half, lat)

    return lat

@jit(nopython=True, nogil=True, fastmath=True, inline="always")
def decode_longitude(encoded_sin: float, encoded_cos: float) -> np.ndarray:
    """
    Decode encoded longitude values to the original longitude value.

    Parameters
    ----------
    encoded_sin : float
        The sine component of the encoded longitude.
    encoded_cos : float
        The cosine component of the encoded longitude.

    Returns
    -------
    np.ndarray
        The decoded longitude value.
    """
    lng_max = 360
    lng_max_half = lng_max / 2.0

    sin_val = (2 * encoded_sin) - 1
    cos_val = (2 * encoded_cos) - 1

    lng_adj = np.arctan2(sin_val, cos_val)

    sign = np.sign(lng_adj)
    sign_adj = np.where(sign == 0, 1, sign) * lng_max_half

    lng = ((lng_adj / (2 * np.pi)) * lng_max) - sign_adj
    lng = np.where(lng == -lng_max_half, lng_max_half, lng)

    return lng

@jit(nopython=True, nogil=True, fastmath=True, inline="always")
def decode_latlng(encoded_latlng: np.ndarray) -> np.ndarray:
    """
    Decode encoded latitude and longitude values to the original values.

    Parameters
    ----------
    encoded_latlng : np.ndarray
        A numpy array containing the encoded latitude and longitude values.

    Returns
    -------
    np.ndarray
        A numpy array containing the decoded latitude and longitude values as float32.
    """
    lat = decode_latitude(encoded_latlng[0], encoded_latlng[1])
    lng = decode_longitude(encoded_latlng[2], encoded_latlng[3])

    return np.stack((lat, lng)).astype(np.float32)

@jit(nopython=True, nogil=True, fastmath=True, inline="always")
def decode_latlngs(encoded_latlngs: np.ndarray) -> np.ndarray:
    """
    Decode multiple latitude and longitude values.

    Parameters
    ----------
    encoded_latlngs : np.ndarray
        A numpy array of encoded latitude and longitude values.

    Returns
    -------
    np.ndarray
        A numpy array of decoded latitude and longitude values.
    """
    latlngs = np.zeros((encoded_latlngs.shape[0], 2), dtype=np.float32)
    for i in prange(encoded_latlngs.shape[0]):
        latlngs[i] = decode_latlng(encoded_latlngs[i])
    return latlngs

@jit(nopython=True, nogil=True, fastmath=True, inline="always")
def encode_width(lng: float, lng_max: float) -> np.ndarray:
    """
    Encodes a longitude value into a 2D array using sine and cosine functions.

    Parameters
    ----------
    lng : float
        The longitude value to encode.
    lng_max : float
        The maximum longitude value for normalization (typically 180).

    Returns
    -------
    np.ndarray
        A 2D array containing the sine and cosine encoded values of the longitude.
    """
    encoded_sin = ((np.sin(2 * np.pi * (lng / lng_max)) + 1)) / 2.0
    encoded_cos = ((np.cos(2 * np.pi * (lng / lng_max)) + 1)) / 2.0

    return np.array([encoded_sin, encoded_cos], dtype=np.float32)
