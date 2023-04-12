""" Tests for ai/scalers.py """


# Standard library
import sys; sys.path.append("../")

# External
import numpy as np

# Internal
from buteo.ai.scalers import (
    scaler_minmax,
    scaler_standardise,
    scaler_standardise_mad,
    scaler_iqr,
    scaler_to_range,
    scaler_truncate,
)


# Test input array
arr = np.array([1, 2, 3, 4, 5], dtype=float)

def test_scaler_minmax():
    """ Test the min-max scaler. """
    result = scaler_minmax(arr)
    expected = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=float)
    assert np.allclose(result, expected)

def test_scaler_standardise():
    """ Test the standardise scaler. """
    result = scaler_standardise(arr)
    expected_mean = 0.0
    expected_std = 1.0
    assert np.isclose(result.mean(), expected_mean, atol=1e-6)
    assert np.isclose(result.std(), expected_std, atol=1e-6)

def test_scaler_standardise_mad():
    """ Test the standardise MAD scaler. """
    result = scaler_standardise_mad(arr)
    expected_median = 0.0
    assert np.isclose(np.median(result), expected_median, atol=1e-6)

def test_scaler_iqr():
    """ Test the IQR scaler. """
    result = scaler_iqr(arr)
    expected_median = 0.0
    assert np.isclose(np.median(result), expected_median, atol=1e-6)

def test_scaler_to_range():
    """ Test the to-range scaler. """
    result = scaler_to_range(arr, min_val=-1.0, max_val=1.0)
    expected = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=float)
    assert np.allclose(result, expected)

def test_scaler_truncate():
    """ Test the truncate scaler. """
    result = scaler_truncate(arr, trunc_min=2.0, trunc_max=4.0, target_min=0.0, target_max=1.0)
    expected = np.array([0.0, 0.0, 0.5, 1.0, 1.0], dtype=float)
    assert np.allclose(result, expected)

def test_scaler_minmax_another_input():
    """ Test the min-max scaler with another input. """
    result = scaler_minmax(arr * 2)
    expected = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=float)
    assert np.allclose(result, expected)

def test_scaler_standardise_another_input():
    """ Test the standardise scaler with another input. """
    result = scaler_standardise(arr * 2)
    expected_mean = 0.0
    expected_std = 1.0
    assert np.isclose(result.mean(), expected_mean, atol=1e-6)
    assert np.isclose(result.std(), expected_std, atol=1e-6)

def test_scaler_standardise_mad_another_input():
    """ Test the standardise MAD scaler with another input. """
    result = scaler_standardise_mad(arr * 2)
    expected_median = 0.0
    assert np.isclose(np.median(result), expected_median, atol=1e-6)

def test_scaler_iqr_another_input():
    """ Test the IQR scaler with another input. """
    result = scaler_iqr(arr * 2)
    expected_median = 0.0
    assert np.isclose(np.median(result), expected_median, atol=1e-6)

def test_scaler_to_range_another_input():
    """ Test the to-range scaler with another input. """
    result = scaler_to_range(arr * 2, min_val=-2.0, max_val=2.0)
    expected = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=float)
    assert np.allclose(result, expected)

def test_scaler_truncate_another_input():
    """ Test the truncate scaler with another input. """
    result = scaler_truncate(arr * 2, trunc_min=4.0, trunc_max=8.0, target_min=0.0, target_max=1.0)
    expected = np.array([0.0, 0.0, 0.5, 1.0, 1.0], dtype=float)
    assert np.allclose(result, expected)
