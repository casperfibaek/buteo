# pylint: skip-file
# type: ignore

import pytest
import numpy as np
from buteo.array import color

def test_rgb_to_hsl(sample_rgb, sample_hsl):
    converted = color.color_rgb_to_hsl(sample_rgb)
    assert converted.shape == (3, 2, 2)
    np.testing.assert_array_almost_equal(converted, sample_hsl, decimal=1)

# There is a rounding error in these ones
def test_hsl_to_rgb(sample_rgb, sample_hsl):
    converted = color.color_hsl_to_rgb(sample_hsl)
    assert converted.shape == (3, 2, 2)
    np.testing.assert_array_almost_equal(converted, sample_rgb, decimal=1)

def test_rgb_to_hsl_input_validation():
    with pytest.raises(AssertionError):
        # Wrong number of dimensions
        color.color_rgb_to_hsl(np.zeros((3,3)))
    
    with pytest.raises(AssertionError):
        # Wrong number of channels
        color.color_rgb_to_hsl(np.zeros((4,3,3)))
        
    with pytest.raises(AssertionError):
        # Values outside [0,1] range
        color.color_rgb_to_hsl(np.ones((3,3,3)) * 2)

def test_hsl_to_rgb_input_validation():
    with pytest.raises(AssertionError):
        # Wrong number of dimensions
        color.color_hsl_to_rgb(np.zeros((3,3)))
    
    with pytest.raises(AssertionError):
        # Wrong number of channels 
        color.color_hsl_to_rgb(np.zeros((4,3,3)))
        
    with pytest.raises(AssertionError):
        # Values outside [0,1] range
        color.color_hsl_to_rgb(np.ones((3,3,3)) * 2)

def test_gray_conversion():
    gray_rgb = np.array([
        [[0.5, 0.5, 0.5]], # Single gray pixel
    ], dtype=np.float32).transpose(2, 0, 1)
    
    converted = color.color_rgb_to_hsl(gray_rgb)
    assert converted.shape == (3, 1, 1)
    assert converted[1,0,0] == 0.0  # Saturation should be 0 for gray
    assert converted[2,0,0] == 0.5  # Lightness should match input
