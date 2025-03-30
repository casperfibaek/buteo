import pytest
import numpy as np
from buteo.array.loaders import MultiArray

# pylint: skip-file
# type: ignore

import sys; sys.path.append("../../")


@pytest.fixture
def sample_array_list():
    """Create a sample list of arrays for testing MultiArray.
    
    Returns:
        list: List of numpy arrays with different shapes.
    """
    # Create three sample arrays with different sizes
    arr1 = np.arange(10).reshape(5, 2)  # 5 items, each with shape (2,)
    arr2 = np.arange(15).reshape(5, 3)  # 5 items, each with shape (3,)
    arr3 = np.arange(8).reshape(4, 2)   # 4 items, each with shape (2,)
    
    return [arr1, arr2, arr3]


def test_multiarray_init(sample_array_list):
    """Test initialization of MultiArray"""
    multi_array = MultiArray(sample_array_list)
    
    # Test length
    assert len(multi_array) == 14  # 5 + 5 + 4
    
    # Test cumulative sizes
    expected_cumulative = np.array([0, 5, 10, 14])
    np.testing.assert_array_equal(multi_array.cumulative_sizes, expected_cumulative)


def test_multiarray_getitem(sample_array_list):
    """Test indexing functionality of MultiArray"""
    multi_array = MultiArray(sample_array_list)
    
    # Test first array items
    for i in range(5):
        np.testing.assert_array_equal(multi_array[i], sample_array_list[0][i])
    
    # Test second array items
    for i in range(5):
        np.testing.assert_array_equal(multi_array[i+5], sample_array_list[1][i])
    
    # Test third array items
    for i in range(4):
        np.testing.assert_array_equal(multi_array[i+10], sample_array_list[2][i])


def test_multiarray_negative_indexing(sample_array_list):
    """Test negative indexing functionality of MultiArray"""
    multi_array = MultiArray(sample_array_list)
    
    # Test negative indexing
    np.testing.assert_array_equal(multi_array[-1], sample_array_list[2][3])  # Last item
    np.testing.assert_array_equal(multi_array[-2], sample_array_list[2][2])  # Second-to-last item


def test_multiarray_out_of_bounds(sample_array_list):
    """Test out of bounds handling in MultiArray"""
    multi_array = MultiArray(sample_array_list)
    
    # Test out of bounds
    with pytest.raises(IndexError):
        _ = multi_array[14]  # Length is 14, so index 14 is out of bounds
    
    with pytest.raises(IndexError):
        _ = multi_array[-15]  # -15 is out of bounds for length 14


def test_multiarray_iteration(sample_array_list):
    """Test iteration through MultiArray"""
    multi_array = MultiArray(sample_array_list)
    
    # Collect expected items
    expected_items = []
    for arr in sample_array_list:
        for i in range(len(arr)):
            expected_items.append(arr[i])
    
    # Check iteration
    for i, item in enumerate(multi_array):
        np.testing.assert_array_equal(item, expected_items[i])


def test_multiarray_shuffle(sample_array_list):
    """Test shuffle functionality of MultiArray"""
    # Create MultiArray with shuffle enabled
    multi_array = MultiArray(sample_array_list, shuffle=True, seed=42)
    
    # Create a reference MultiArray without shuffle
    ref_array = MultiArray(sample_array_list, shuffle=False)
    
    # Verify the arrays are shuffled
    found_different = False
    for i in range(len(multi_array)):
        item = multi_array[i]
        ref_item = ref_array[i]
        if not np.array_equal(item, ref_item):
            found_different = True
            break
    
    assert found_different, "Shuffled array should be different from non-shuffled array"
    
    # Test shuffle indices access
    assert multi_array.shuffle_indices is not None
    assert len(multi_array.shuffle_indices) == len(multi_array)
    
    # Test disabling shuffle
    multi_array.disable_shuffle()
    assert not multi_array.shuffle
    
    # After disabling shuffle, array should be the same as reference
    for i in range(len(multi_array)):
        np.testing.assert_array_equal(multi_array[i], ref_array[i])
    
    # Test re-enabling shuffle with shuffle_index
    multi_array.shuffle_index()
    assert multi_array.shuffle
    assert multi_array.shuffle_indices is not None


def test_multiarray_random_sampling(sample_array_list):
    """Test random sampling functionality of MultiArray"""
    # Create MultiArray with random sampling enabled
    multi_array = MultiArray(sample_array_list, random_sampling=True, seed=42)
    
    # Check that random sampling doesn't fail
    for _ in range(10):
        item = multi_array[0]  # Index doesn't matter with random sampling
        assert isinstance(item, np.ndarray)
    
    # Test random sampling and shuffle can't be used together
    with pytest.raises(ValueError):
        MultiArray(sample_array_list, shuffle=True, random_sampling=True)


def test_multiarray_split(sample_array_list):
    """Test split functionality of MultiArray"""
    multi_array = MultiArray(sample_array_list)
    
    # Split at index 6 (mid-point of second array)
    first_part, second_part = multi_array.split(6)
    
    # Check lengths
    assert len(first_part) == 6
    assert len(second_part) == 8
    
    # Check first part content
    for i in range(6):
        np.testing.assert_array_equal(first_part[i], multi_array[i])
    
    # Check second part content
    for i in range(8):
        np.testing.assert_array_equal(second_part[i], multi_array[i+6])
    
    # Test splitting with float
    first_part, second_part = multi_array.split(0.25)  # 25% split point
    assert len(first_part) == 3  # 25% of 14 is 3.5, rounded down
    assert len(second_part) == 11
    
    # Test cannot split a subarray
    with pytest.raises(ValueError):
        first_part.split(1)
    
    # Test invalid split point
    with pytest.raises(ValueError):
        multi_array.split(20)  # Greater than length
