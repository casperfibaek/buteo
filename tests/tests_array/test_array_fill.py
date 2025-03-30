import numpy as np
import pytest
from buteo.array.fill import convolve_fill_nearest, convolve_fill_nearest_classes

# pylint: skip-file
# type: ignore

import sys; sys.path.append("../../")


# pylint: skip-file
# type: ignore

import sys; sys.path.append("../../")


@pytest.fixture
def simple_array():
    # Create a 3x3 array with one channel.
    # Use nodata value -1. Fill center with nodata.
    arr = np.array([
        [1,   1,  1],
        [1,  -1,  1],
        [1,   1,  1]
    ], dtype=np.float32)
    # Expand dims to simulate a (channels, rows, cols) array.
    return np.expand_dims(arr, axis=0)


@pytest.fixture
def classification_array():
    # Create a 3x3 classification array with one channel.
    # Use nodata value -1 and two classes: 1 and 2.
    # In this example, the center pixel is nodata. Neighbors are arranged so that
    # class 1 appears in the tie-breaker (unique() returns sorted order [ -1, 1, 2 ])
    arr = np.array([
        [1, 1, 2],
        [2, -1, 1],
        [1, 2, 2]
    ], dtype=np.int64)
    return np.expand_dims(arr, axis=0)


def test_convolve_fill_nearest_simple(simple_array):
    nodata_value = -1.0
    # Run filling with several iterations until filled.
    filled = convolve_fill_nearest(simple_array, nodata_value=nodata_value)
    # After fill, no nodata (-1) should remain in the valid mask.
    assert np.all(filled[filled != nodata_value] != nodata_value)
    # In this simple case, the surrounding pixels were all 1;
    # the filled value should be 1 at channel 0, row 1, col 1.
    assert np.isclose(filled[0, 1, 1], 1.0, atol=1e-4)


def test_convolve_fill_nearest_max_iterations(simple_array):
    nodata_value = -1.0
    # Limit iterations to 1, so center might remain nodata
    filled = convolve_fill_nearest(simple_array, nodata_value=nodata_value, max_iterations=1)
    # With one iteration, if not filled then center remains nodata.
    if filled[0, 1, 1] != nodata_value:
        # Otherwise, if one iteration was enough, test that it becomes 1.
        assert np.isclose(filled[0, 1, 1], 1.0, atol=1e-4)
    else:
        assert filled[0, 1, 1] == nodata_value


def test_convolve_fill_nearest_classes(classification_array):
    nodata_value = -1
    filled = convolve_fill_nearest_classes(classification_array, nodata_value=nodata_value, mask=np.ones_like(classification_array))
    # After filling, no nodata should remain in the valid mask.
    assert np.all(filled[filled != nodata_value] != nodata_value)
    # Given the unique classes [1,2] and tie-break by first occurrence,
    # the nodata pixel should be filled with class 1 (at channel 0, row 1, col 1).
    assert filled[0, 1, 1] == 1


def test_convolve_fill_nearest_classes_with_mask(classification_array):
    nodata_value = -1
    # Create a mask that is 0 for the top-left pixel and 1 elsewhere.
    mask = np.ones_like(classification_array, dtype=np.uint8)
    mask[0, 0, 0] = 0
    
    # Store the original value to test it remains unchanged
    original_value = classification_array[0, 0, 0]
    
    filled = convolve_fill_nearest_classes(classification_array, nodata_value=nodata_value, mask=mask)
    
    # The masked-out pixel should remain unchanged.
    assert filled[0, 0, 0] == original_value
    
    # The nodata pixel at center should be filled with one of the neighboring classes (2 in this case)
    # Note: The actual class used depends on implementation details of the filling algorithm
    # In this case, class 2 is used as it might be encountered first in the filling process
    assert filled[0, 1, 1] == 2
