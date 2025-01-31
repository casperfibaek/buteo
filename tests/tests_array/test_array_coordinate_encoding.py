# pylint: skip-file
# type: ignore

import sys; sys.path.append("../../")

import pytest
import numpy as np
from buteo.array import coordinate_encoding


@pytest.fixture
def sample_latlng():
    return np.array([55.6761, 12.5683], dtype=np.float32)  # Copenhagen coordinates

@pytest.fixture
def sample_latlngs():
    return np.array([
        [55.6761, 12.5683],  # Copenhagen
        [40.7128, -74.0060],  # New York
        [34.0522, -118.2437]  # Los Angeles
    ], dtype=np.float32)

def test_encode_latitude(sample_latlng):
    encoded = coordinate_encoding.encode_latitude(sample_latlng[0])
    assert encoded.shape == (2,)
    assert np.all(encoded >= 0) and np.all(encoded <= 1)

def test_encode_longitude(sample_latlng):
    encoded = coordinate_encoding.encode_longitude(sample_latlng[1])
    assert encoded.shape == (2,)
    assert np.all(encoded >= 0) and np.all(encoded <= 1)

def test_encode_latlng(sample_latlng):
    encoded = coordinate_encoding.encode_latlng(sample_latlng)
    assert encoded.shape == (4,)
    assert np.all(encoded >= 0) and np.all(encoded <= 1)

def test_encode_latlngs(sample_latlngs):
    encoded = coordinate_encoding.encode_latlngs(sample_latlngs)
    assert encoded.shape == (3, 4)
    assert np.all(encoded >= 0) and np.all(encoded <= 1)

def test_decode_latitude(sample_latlng):
    encoded = coordinate_encoding.encode_latitude(sample_latlng[0])
    decoded = coordinate_encoding.decode_latitude(encoded[0], encoded[1])
    np.testing.assert_almost_equal(decoded, sample_latlng[0], decimal=4)

def test_decode_longitude(sample_latlng):
    encoded = coordinate_encoding.encode_longitude(sample_latlng[1])
    decoded = coordinate_encoding.decode_longitude(encoded[0], encoded[1])
    np.testing.assert_almost_equal(decoded, sample_latlng[1], decimal=4)

def test_decode_latlng(sample_latlng):
    encoded = coordinate_encoding.encode_latlng(sample_latlng)
    decoded = coordinate_encoding.decode_latlng(encoded)
    np.testing.assert_almost_equal(decoded, sample_latlng, decimal=4)

def test_decode_latlngs(sample_latlngs):
    encoded = coordinate_encoding.encode_latlngs(sample_latlngs)
    decoded = coordinate_encoding.decode_latlngs(encoded)
    np.testing.assert_almost_equal(decoded, sample_latlngs, decimal=4)
