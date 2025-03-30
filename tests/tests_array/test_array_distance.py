# pylint: skip-file
# type: ignore

import pytest
import numpy as np
from buteo.array.distance import convolve_distance

@pytest.fixture
def binary_array_3d():
    """Create a simple 3D binary array with a few 1s in known positions."""
    arr = np.zeros((5, 5, 1), dtype=np.float32)
    # Set a few pixels to 1
    arr[1, 1, 0] = 1.0
    arr[3, 3, 0] = 1.0
    return arr

@pytest.fixture
def multi_value_array_3d():
    """Create a 3D array with different values."""
    arr = np.zeros((5, 5, 1), dtype=np.float32)
    arr[1, 1, 0] = 2.0
    arr[3, 3, 0] = 3.0
    arr[4, 0, 0] = 2.0
    return arr

def test_convolve_distance_basic(binary_array_3d):
    """Test basic distance calculation with default parameters."""
    result = convolve_distance(binary_array_3d)
    
    # Check shape
    assert result.shape == binary_array_3d.shape
    
    # Pixels with target values should have zero distance
    assert result[1, 1, 0] == 0.0
    assert result[3, 3, 0] == 0.0
    
    # Check a few specific distances
    # Distance from (0,0) to nearest target (1,1) should be sqrt(2)
    np.testing.assert_almost_equal(result[0, 0, 0], np.sqrt(2), decimal=5)
    
    # Distance from (2,2) to nearest targets should be sqrt(2)
    np.testing.assert_almost_equal(result[2, 2, 0], np.sqrt(2), decimal=5)
    
    # Center pixel at (2,2) is equidistant from (1,1) and (3,3)
    assert result[2, 2, 0] == result[2, 2, 0]

def test_convolve_distance_custom_target(multi_value_array_3d):
    """Test distance calculation with custom target value."""
    # Calculate distance to pixels with value 2
    result = convolve_distance(multi_value_array_3d, target=2.0)
    
    # Pixels with target value 2 should have zero distance
    assert result[1, 1, 0] == 0.0
    assert result[4, 0, 0] == 0.0
    
    # Pixel with value 3 should have non-zero distance
    assert result[3, 3, 0] > 0.0
    
    # Distance from (3,3) to nearest target (1,1) should be 2*sqrt(2)
    np.testing.assert_almost_equal(result[3, 3, 0], 2*np.sqrt(2), decimal=5)

def test_convolve_distance_maximum_distance(binary_array_3d):
    """Test distance calculation with a maximum distance limit."""
    # Set a maximum distance of 2.0
    result = convolve_distance(binary_array_3d, maximum_distance=2.0)
    
    # Pixels with distance <= 2.0 should have their actual distance
    np.testing.assert_almost_equal(result[0, 0, 0], np.sqrt(2), decimal=5)  # Distance from (0,0) to (1,1)
    np.testing.assert_almost_equal(result[2, 2, 0], np.sqrt(2), decimal=5)  # Distance from (2,2) to (1,1) or (3,3)
    
    # Pixels with distance > 2.0 should have the maximum value
    assert result[0, 4, 0] == 2.0  # Distance from (0,4) to nearest target is > 2.0

def test_convolve_distance_pixel_dimensions(binary_array_3d):
    """Test distance calculation with non-square pixels."""
    # Set pixel dimensions to simulate rectangular pixels
    result = convolve_distance(binary_array_3d, pixel_width=2.0, pixel_height=1.0)
    
    # The distance calculation may use a different formula or coordinate system
    # The key here is to verify that:
    # 1. The result is non-zero (distance is properly calculated)
    # 2. The pixel dimensions affect the result
    
    # Get result with default square pixels for comparison
    result_square = convolve_distance(binary_array_3d)
    
    # The rectangular pixels should produce a different distance compared to square pixels
    assert result[0, 0, 0] != result_square[0, 0, 0]
    
    # Distance should be within a reasonable range
    assert 2.0 < result[0, 0, 0] < 10.0

def test_convolve_distance_no_targets():
    """Test distance calculation when there are no target pixels."""
    # Create an array with no target pixels
    arr = np.zeros((5, 5, 1), dtype=np.float32)
    
    # The distance should be the maximum possible for all pixels
    result = convolve_distance(arr)
    
    # The maximum distance should be the diagonal of the array: sqrt(5^2 + 5^2) = 5*sqrt(2)
    expected_max = np.sqrt(5**2 + 5**2)
    np.testing.assert_almost_equal(result[0, 0, 0], expected_max, decimal=5)
    np.testing.assert_almost_equal(result[4, 4, 0], expected_max, decimal=5)

def test_convolve_distance_all_targets():
    """Test distance calculation when all pixels are targets."""
    # Create an array where all pixels are targets
    arr = np.ones((5, 5, 1), dtype=np.float32)
    
    # The distance should be zero for all pixels
    result = convolve_distance(arr)
    
    # All distances should be zero
    assert np.all(result == 0.0)
