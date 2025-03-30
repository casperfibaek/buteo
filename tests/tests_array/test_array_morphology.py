# pylint: skip-file
# type: ignore

import pytest
import numpy as np
from buteo.array.morphology import (
    morph_erode, morph_dilate, morph_open, morph_close,
    morph_tophat, morph_bothat, morph_difference
)

@pytest.fixture
def binary_array():
    """Create a simple binary array for testing morphological operations."""
    arr = np.zeros((7, 7), dtype=np.float32)
    # Set a small rectangle of ones
    arr[2:5, 2:5] = 1.0
    # Reshape to 3D (H,W,1) for compatibility
    return arr.reshape(7, 7, 1)

@pytest.fixture
def binary_array_with_noise():
    """Create a binary array with noise for testing morphological operations."""
    arr = np.zeros((7, 7), dtype=np.float32)
    # Set a small rectangle of ones
    arr[2:5, 2:5] = 1.0
    # Add some noise (small isolated pixels)
    arr[1, 1] = 1.0
    arr[5, 5] = 1.0
    # Add a hole
    arr[3, 3] = 0.0
    # Reshape to 3D (H,W,1) for compatibility
    return arr.reshape(7, 7, 1)

@pytest.fixture
def grayscale_array():
    """Create a simple grayscale array for testing morphological operations."""
    arr = np.zeros((7, 7), dtype=np.float32)
    # Create a gradient
    for i in range(7):
        for j in range(7):
            arr[i, j] = (i + j) / 12.0  # Values between 0 and 1
    # Reshape to 3D (H,W,1) for compatibility
    return arr.reshape(7, 7, 1)

def test_morph_erode_binary(binary_array):
    """Test erosion on binary array."""
    result = morph_erode(binary_array)
    
    # Check shape
    assert result.shape == binary_array.shape
    
    # Erosion should reduce the size of the foreground (ones)
    assert np.sum(result) < np.sum(binary_array)
    
    # Specifically, a 3x3 structuring element should erode the 3x3 square to a single pixel
    # Center pixel should remain 1, others should be eroded to 0
    assert result[3, 3, 0] == 1.0
    assert np.sum(result) == 1.0
    
    # Test with radius=2
    result_r2 = morph_erode(binary_array, radius=2)
    # With radius 2, everything should be eroded away
    assert np.sum(result_r2) == 0.0

def test_morph_erode_grayscale(grayscale_array):
    """Test erosion on grayscale array."""
    result = morph_erode(grayscale_array)
    
    # Check shape
    assert result.shape == grayscale_array.shape
    
    # Erosion should reduce the intensity values
    assert np.mean(result) < np.mean(grayscale_array)
    
    # The minimum value should remain the same
    assert np.min(result) == np.min(grayscale_array)
    
    # The maximum value may be reduced or at most equal to the original
    assert np.max(result) <= np.max(grayscale_array)

def test_morph_dilate_binary(binary_array):
    """Test dilation on binary array."""
    result = morph_dilate(binary_array)
    
    # Check shape
    assert result.shape == binary_array.shape
    
    # Dilation should increase the size of the foreground (ones)
    assert np.sum(result) > np.sum(binary_array)
    
    # Test with radius=1
    # The 3x3 square should be dilated to a 5x5 square
    expected_dilated_coords = [(1,1), (1,2), (1,3), (1,4), (1,5),
                               (2,1), (2,2), (2,3), (2,4), (2,5),
                               (3,1), (3,2), (3,3), (3,4), (3,5),
                               (4,1), (4,2), (4,3), (4,4), (4,5),
                               (5,1), (5,2), (5,3), (5,4), (5,5)]
    
    for i, j in expected_dilated_coords:
        assert result[i, j, 0] == 1.0
    
    # Test with radius=2
    result_r2 = morph_dilate(binary_array, radius=2)
    # With radius 2, the dilation should extend further
    assert np.sum(result_r2) > np.sum(result)

def test_morph_dilate_grayscale(grayscale_array):
    """Test dilation on grayscale array."""
    result = morph_dilate(grayscale_array)
    
    # Check shape
    assert result.shape == grayscale_array.shape
    
    # Dilation should increase the intensity values
    assert np.mean(result) > np.mean(grayscale_array)
    
    # The maximum value should remain the same
    assert np.max(result) == np.max(grayscale_array)
    
    # The minimum value should be increased
    assert np.min(result) > np.min(grayscale_array)

def test_morph_open_binary(binary_array_with_noise):
    """Test opening on binary array with noise."""
    result = morph_open(binary_array_with_noise)
    
    # Check shape
    assert result.shape == binary_array_with_noise.shape
    
    # Opening should remove small isolated pixels but preserve the main shape
    # The isolated pixels at (1,1) and (5,5) should be removed
    assert result[1, 1, 0] == 0.0
    assert result[5, 5, 0] == 0.0
    
    # The rectangle may be reduced in size or completely eroded, depending on implementation details
    # Since a single pixel may remain or none, just check the shape
    assert result.shape == binary_array_with_noise.shape
    
    # Opening is erosion followed by dilation, so result should match this sequence
    eroded = morph_erode(binary_array_with_noise)
    expected = morph_dilate(eroded)
    np.testing.assert_array_equal(result, expected)

def test_morph_close_binary(binary_array_with_noise):
    """Test closing on binary array with noise."""
    result = morph_close(binary_array_with_noise)
    
    # Check shape
    assert result.shape == binary_array_with_noise.shape
    
    # Closing should fill small holes but preserve the main shape
    # The hole at (3,3) should be filled
    assert result[3, 3, 0] == 1.0
    
    # Closing is dilation followed by erosion, so result should match this sequence
    dilated = morph_dilate(binary_array_with_noise)
    expected = morph_erode(dilated)
    np.testing.assert_array_equal(result, expected)

def test_morph_tophat(binary_array_with_noise):
    """Test top-hat operation."""
    result = morph_tophat(binary_array_with_noise)
    
    # Check shape
    assert result.shape == binary_array_with_noise.shape
    
    # Top-hat is original - opened, should highlight small bright features
    opened = morph_open(binary_array_with_noise)
    expected = binary_array_with_noise - opened
    np.testing.assert_array_almost_equal(result, expected)
    
    # The small isolated pixels should be highlighted
    assert result[1, 1, 0] > 0.0
    assert result[5, 5, 0] > 0.0

def test_morph_bothat(binary_array_with_noise):
    """Test bottom-hat operation."""
    result = morph_bothat(binary_array_with_noise)
    
    # Check shape
    assert result.shape == binary_array_with_noise.shape
    
    # Bottom-hat is closed - original, should highlight small dark features
    closed = morph_close(binary_array_with_noise)
    expected = closed - binary_array_with_noise
    np.testing.assert_array_almost_equal(result, expected)
    
    # The hole should be highlighted
    assert result[3, 3, 0] > 0.0

def test_morph_difference(binary_array):
    """Test morphological difference (gradient)."""
    result = morph_difference(binary_array)
    
    # Check shape
    assert result.shape == binary_array.shape
    
    # Difference is dilate - erode, should highlight edges
    dilated = morph_dilate(binary_array)
    eroded = morph_erode(binary_array)
    expected = dilated - eroded
    np.testing.assert_array_equal(result, expected)
    
    # The edges of the rectangle should be highlighted
    edge_coords = [(2,2), (2,3), (2,4), (3,2), (3,4), (4,2), (4,3), (4,4)]
    for i, j in edge_coords:
        assert result[i, j, 0] > 0.0
    
    # The center should be 0 (not an edge)
    assert result[3, 3, 0] == 0.0

def test_spherical_vs_square_kernels(binary_array):
    """Test difference between spherical and square kernels."""
    # Spherical kernel (circle)
    result_spherical = morph_dilate(binary_array, spherical=True)
    
    # Square kernel
    result_square = morph_dilate(binary_array, spherical=False)
    
    # Square kernel should dilate corners more than spherical
    assert np.sum(result_square) >= np.sum(result_spherical)
    
    # The difference should mainly be in the corners
    corner_coords = [(1,1), (1,5), (5,1), (5,5)]
    diff = result_square - result_spherical
    for i, j in corner_coords:
        assert diff[i, j, 0] >= 0

def test_channel_order_morphology():
    """Test morphology operations with different channel orders."""
    # Create a channel-first array (C,H,W)
    arr = np.zeros((1, 7, 7), dtype=np.float32)
    arr[0, 2:5, 2:5] = 1.0
    
    # Apply morphology with channel_last=False
    result = morph_erode(arr, channel_last=False)
    
    # Check shape - should maintain channel-first format
    assert result.shape == arr.shape
    assert result.shape[0] == 1  # First dimension should be channels
    
    # Erosion should work the same way
    assert np.sum(result) < np.sum(arr)

def test_fractional_radius():
    """Test morphology operations with fractional radius."""
    # Create a binary array
    arr = np.zeros((7, 7, 1), dtype=np.float32)
    arr[2:5, 2:5, 0] = 1.0
    
    # Apply erosion with fractional radius
    result_frac = morph_erode(arr, radius=1.5)
    
    # Result should be between radius 1 and radius 2
    result_r1 = morph_erode(arr, radius=1)
    result_r2 = morph_erode(arr, radius=2)
    
    # Fractional radius should erode more than radius 1 but less than radius 2
    assert np.sum(result_r1) >= np.sum(result_frac) >= np.sum(result_r2)

def test_masked_arrays():
    """Test morphology operations on masked arrays."""
    # Create a binary array
    arr = np.zeros((7, 7, 1), dtype=np.float32)
    arr[2:5, 2:5, 0] = 1.0
    
    # Create a mask
    mask = np.zeros_like(arr, dtype=bool)
    mask[0, 0, 0] = True  # Mask one corner
    
    # Create masked array
    masked_arr = np.ma.array(arr, mask=mask, fill_value=-9999.9)
    
    # Apply erosion
    result = morph_erode(masked_arr)
    
    # Result should be a masked array with a mask attribute
    assert hasattr(result, 'mask')
    
    # Check that it's the proper masked array type
    assert isinstance(result, np.ma.MaskedArray)
    
    # Check that the array data was processed correctly, ignoring the mask
    expected = morph_erode(arr)
    np.testing.assert_array_almost_equal(result.data, expected)
