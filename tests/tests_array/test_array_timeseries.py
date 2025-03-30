# pylint: skip-file
# type: ignore

import pytest
import numpy as np
from buteo.array.timeseries import (
    timeseries_least_square_slope,
    timeseries_robust_least_squares_slope
)

@pytest.fixture
def linear_timeseries():
    """Create a simple 3D array containing linear time series data."""
    # Create a 5x5 array with 10 time steps
    arr = np.zeros((5, 5, 10), dtype=np.float32)
    
    # Add different slopes for each pixel
    for i in range(5):
        for j in range(5):
            # Slope is related to position (i, j)
            slope = 0.1 * (i + j)
            # Create a linear time series with the given slope
            arr[i, j, :] = np.arange(10) * slope
    
    return arr

@pytest.fixture
def noisy_linear_timeseries():
    """Create a 3D array containing linear time series data with noise."""
    # Create a 5x5 array with 10 time steps
    arr = np.zeros((5, 5, 10), dtype=np.float32)
    
    # Add different slopes for each pixel
    for i in range(5):
        for j in range(5):
            # Slope is related to position (i, j)
            slope = 0.1 * (i + j)
            # Create a linear time series with the given slope
            arr[i, j, :] = np.arange(10) * slope
    
    # Add some random noise
    np.random.seed(42)  # For reproducibility
    noise = np.random.normal(0, 0.1, size=(5, 5, 10)).astype(np.float32)
    
    return arr + noise

@pytest.fixture
def outlier_timeseries():
    """Create a 3D array containing linear time series data with outliers."""
    # Create a 5x5 array with 10 time steps
    arr = np.zeros((5, 5, 10), dtype=np.float32)
    
    # Add different slopes for each pixel
    for i in range(5):
        for j in range(5):
            # Slope is related to position (i, j)
            slope = 0.1 * (i + j)
            # Create a linear time series with the given slope
            arr[i, j, :] = np.arange(10) * slope
    
    # Add some extreme outliers
    # Modify the middle time step for all pixels
    arr[:, :, 5] *= 10.0
    
    return arr

def test_timeseries_least_square_slope_linear(linear_timeseries):
    """Test least square slope calculation on perfectly linear data."""
    result = timeseries_least_square_slope(linear_timeseries)
    
    # Check shape: should be (5, 5, 1)
    assert result.shape == (5, 5, 1)
    
    # Check values: For each pixel, the slope should match our input slopes
    for i in range(5):
        for j in range(5):
            expected_slope = 0.1 * (i + j)
            np.testing.assert_almost_equal(result[i, j, 0], expected_slope, decimal=5)

def test_timeseries_least_square_slope_noisy(noisy_linear_timeseries):
    """Test least square slope calculation on noisy linear data."""
    result = timeseries_least_square_slope(noisy_linear_timeseries)
    
    # Check shape: should be (5, 5, 1)
    assert result.shape == (5, 5, 1)
    
    # Check values: For each pixel, the slope should be close to our input slopes
    # but within some tolerance due to the noise
    for i in range(5):
        for j in range(5):
            expected_slope = 0.1 * (i + j)
            np.testing.assert_almost_equal(result[i, j, 0], expected_slope, decimal=1)

def test_timeseries_least_square_slope_empty():
    """Test least square slope calculation on empty array."""
    # Create an empty time series with the right shape
    arr = np.zeros((0, 0, 5), dtype=np.float32)
    
    # Should handle empty array with AssertionError
    # The function should validate dimensions
    with pytest.raises((AssertionError, ValueError, IndexError)):
        timeseries_least_square_slope(arr)

def test_timeseries_least_square_slope_single_point():
    """Test least square slope calculation with a single time point."""
    # Create a time series with only one time point
    arr = np.ones((5, 5, 1), dtype=np.float32)
    
    # When there's only one time point, division by zero should produce NaN or inf
    result = timeseries_least_square_slope(arr)
    
    # All slopes should be NaN or inf
    assert np.all(~np.isfinite(result))

def test_timeseries_robust_least_squares_slope_linear(linear_timeseries):
    """Test robust least squares slope calculation on perfectly linear data."""
    result = timeseries_robust_least_squares_slope(
        linear_timeseries, std_threshold=1.0, splits=1, report_progress=False
    )
    
    # Check shape: should be (5, 5, 1)
    assert result.shape == (5, 5, 1)
    
    # Check values: For each pixel, the slope should match our input slopes
    for i in range(5):
        for j in range(5):
            expected_slope = 0.1 * (i + j)
            np.testing.assert_almost_equal(result[i, j, 0], expected_slope, decimal=5)

def test_timeseries_robust_least_squares_slope_outliers(outlier_timeseries):
    """Test robust least squares slope calculation on data with outliers."""
    result = timeseries_robust_least_squares_slope(
        outlier_timeseries, std_threshold=1.0, splits=1, report_progress=False
    )
    
    # Check shape: should be (5, 5, 1)
    assert result.shape == (5, 5, 1)
    
    # Check values: For each pixel, the slope should be close to our input slopes
    # despite the outliers
    for i in range(5):
        for j in range(5):
            expected_slope = 0.1 * (i + j)
            np.testing.assert_almost_equal(result[i, j, 0], expected_slope, decimal=1)
    
    # The robust method should be less affected by outliers than the standard method
    standard_result = timeseries_least_square_slope(outlier_timeseries)
    
    # At least some pixels should differ significantly between the methods
    assert np.any(np.abs(result - standard_result) > 0.01)

def test_timeseries_robust_least_squares_slope_splits(linear_timeseries):
    """Test robust least squares slope calculation with multiple splits."""
    # Test with splits=5
    result = timeseries_robust_least_squares_slope(
        linear_timeseries, std_threshold=1.0, splits=5, report_progress=False
    )
    
    # Check shape: should be (5, 5, 1)
    assert result.shape == (5, 5, 1)
    
    # Check values: For each pixel, the slope should match our input slopes
    for i in range(5):
        for j in range(5):
            expected_slope = 0.1 * (i + j)
            np.testing.assert_almost_equal(result[i, j, 0], expected_slope, decimal=5)

def test_timeseries_robust_least_squares_slope_std_threshold(outlier_timeseries):
    """Test robust least squares slope calculation with different std thresholds."""
    # Compare with standard least squares (which is sensitive to outliers)
    standard_result = timeseries_least_square_slope(outlier_timeseries)
    
    # Test with a very permissive threshold (should include almost all points, similar to least squares)
    result_high_threshold = timeseries_robust_least_squares_slope(
        outlier_timeseries, std_threshold=10.0, splits=1, report_progress=False
    )
    
    # Test with a very restrictive threshold (should exclude outliers more aggressively)
    result_low_threshold = timeseries_robust_least_squares_slope(
        outlier_timeseries, std_threshold=0.1, splits=1, report_progress=False
    )
    
    # The permissive threshold should be closer to the standard least squares result
    # The restrictive threshold should be more different from standard least squares
    high_diff = np.sum(np.abs(result_high_threshold - standard_result))
    low_diff = np.sum(np.abs(result_low_threshold - standard_result))
    
    # When the differences are very close, they should be at least equal (because of floating point precision)
    # and we could consider this a valid case for lower threshold not performing worse
    assert low_diff >= high_diff, "Low threshold result should be at least as different from standard least squares"
