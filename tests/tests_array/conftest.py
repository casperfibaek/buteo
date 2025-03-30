"""Fixtures for array tests."""

import pytest
import numpy as np

@pytest.fixture
def sample_rgb():
    """Create a sample RGB image array for testing.
    
    Returns:
        numpy.ndarray: 3-channel RGB test image with shape (3, 2, 2)
        - Channel order is RGB (0=Red, 1=Green, 2=Blue)
        - Contains red, green, blue, and gray test pixels
    """
    return np.array([
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],  # Red and Green
        [[0.0, 0.0, 1.0], [0.5, 0.5, 0.5]]   # Blue and Gray
    ], dtype=np.float32).transpose(2, 0, 1)

@pytest.fixture 
def sample_hsl():
    """Create a sample HSL image array for testing.
    
    Returns:
        numpy.ndarray: 3-channel HSL test image with shape (3, 2, 2)
        - Channel order is HSL (0=Hue, 1=Saturation, 2=Lightness)
        - Contains the same colors as sample_rgb but in HSL color space
    """
    return np.array([
        [[0.0, 1.0, 0.5], [0.33, 1.0, 0.5]],  # Red and Green
        [[0.67, 1.0, 0.5], [0.0, 0.0, 0.5]]   # Blue and Gray  
    ], dtype=np.float32).transpose(2, 0, 1)

@pytest.fixture
def sample_grayscale():
    """Create a sample grayscale image for testing.
    
    Returns:
        numpy.ndarray: Single-channel grayscale image with shape (1, 2, 2)
    """
    return np.array([[0.1, 0.5], [0.7, 0.9]], dtype=np.float32).reshape(1, 2, 2)

@pytest.fixture
def sample_array_2d():
    """Create a simple 2D array for testing.
    
    Returns:
        numpy.ndarray: 2D array with values from 0 to 3
    """
    return np.array([[0, 1], [2, 3]], dtype=np.float32)

@pytest.fixture
def sample_array_3d():
    """Create a simple 3D array for testing.
    
    Returns:
        numpy.ndarray: 3D array with 2 bands of values
    """
    return np.array([
        [[0, 1], [2, 3]],
        [[4, 5], [6, 7]]
    ], dtype=np.float32)
