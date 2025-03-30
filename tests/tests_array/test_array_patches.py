# pylint: skip-file
# type: ignore

import pytest
import numpy as np
from buteo.array.patches import array_to_patches
from buteo.array.patches.extraction import _array_to_patches_single, _patches_to_array_single

@pytest.fixture
def sample_array_3d():
    """Create a sample 3D array for testing patches."""
    # Create a 10x10 array with 3 channels
    arr = np.zeros((10, 10, 3), dtype=np.float32)
    
    # Add a gradient pattern for easier visual verification
    for i in range(10):
        for j in range(10):
            arr[i, j, 0] = i / 10.0  # Red channel: vertical gradient
            arr[i, j, 1] = j / 10.0  # Green channel: horizontal gradient
            arr[i, j, 2] = (i + j) / 20.0  # Blue channel: diagonal gradient
    
    return arr

@pytest.fixture
def sample_array_channel_first():
    """Create a sample channel-first 3D array for testing patches."""
    # Create a 3x10x10 array (channel-first)
    arr = np.zeros((3, 10, 10), dtype=np.float32)
    
    # Add a gradient pattern
    for i in range(10):
        for j in range(10):
            arr[0, i, j] = i / 10.0  # Red channel: vertical gradient
            arr[1, i, j] = j / 10.0  # Green channel: horizontal gradient
            arr[2, i, j] = (i + j) / 20.0  # Blue channel: diagonal gradient
    
    return arr

def test_array_to_patches_single_basic(sample_array_3d):
    """Test basic functionality of _array_to_patches_single."""
    # Test with default offset
    patches = _array_to_patches_single(sample_array_3d, tile_size=5)
    
    # Should create 4 patches of size 5x5x3
    assert patches.shape == (4, 5, 5, 3)
    
    # Check the content of the first patch (top-left)
    expected_first_patch = sample_array_3d[0:5, 0:5, :]
    np.testing.assert_array_equal(patches[0], expected_first_patch)
    
    # Check the content of the last patch (bottom-right)
    expected_last_patch = sample_array_3d[5:10, 5:10, :]
    np.testing.assert_array_equal(patches[3], expected_last_patch)

def test_array_to_patches_single_with_offset(sample_array_3d):
    """Test _array_to_patches_single with custom offset."""
    # Test with offset [1, 1]
    patches = _array_to_patches_single(sample_array_3d, tile_size=5, offset=[1, 1])
    
    # Should create 1 patch of size 5x5x3 (due to offset and array size)
    assert patches.shape == (1, 5, 5, 3)
    
    # Check the content of the patch
    expected_patch = sample_array_3d[1:6, 1:6, :]
    np.testing.assert_array_equal(patches[0], expected_patch)

def test_patches_to_array_single_basic(sample_array_3d):
    """Test basic functionality of _patches_to_array_single."""
    # First create patches
    patches = _array_to_patches_single(sample_array_3d, tile_size=5)
    
    # Then reconstruct the array
    reconstructed = _patches_to_array_single(patches, shape=sample_array_3d.shape, tile_size=5)
    
    # The reconstructed array should match the original
    np.testing.assert_array_equal(reconstructed, sample_array_3d)

def test_patches_to_array_single_with_offset(sample_array_3d):
    """Test _patches_to_array_single with custom offset."""
    # Create patches with offset
    offset = [1, 1]
    patches = _array_to_patches_single(sample_array_3d, tile_size=5, offset=offset)
    
    # Reconstruct with same offset
    reconstructed = _patches_to_array_single(patches, shape=sample_array_3d.shape, tile_size=5, offset=offset)
    
    # Check that the reconstructed parts match the original
    np.testing.assert_array_equal(reconstructed[1:6, 1:6, :], sample_array_3d[1:6, 1:6, :])
    
    # Check that non-reconstructed parts have the background value (NaN by default)
    assert np.isnan(reconstructed[0, 0, 0])

def test_patches_to_array_single_with_background(sample_array_3d):
    """Test _patches_to_array_single with custom background value."""
    # Create patches
    patches = _array_to_patches_single(sample_array_3d, tile_size=5, offset=[1, 1])
    
    # Reconstruct with background value
    background_value = -1.0
    reconstructed = _patches_to_array_single(
        patches, 
        shape=sample_array_3d.shape, 
        tile_size=5, 
        offset=[1, 1], 
        background_value=background_value
    )
    
    # Check that non-reconstructed parts have the specified background value
    assert reconstructed[0, 0, 0] == background_value

def test_array_to_patches_basic(sample_array_3d):
    """Test basic functionality of array_to_patches."""
    # Test with default parameters
    patches = array_to_patches(sample_array_3d, tile_size=5)
    
    # Should create 4 patches of size 5x5x3
    assert patches.shape == (4, 5, 5, 3)
    
    # Check the content of the first patch (top-left)
    expected_first_patch = sample_array_3d[0:5, 0:5, :]
    np.testing.assert_array_equal(patches[0], expected_first_patch)

def test_array_to_patches_with_offsets(sample_array_3d):
    """Test array_to_patches with multiple offsets."""
    # Test with n_offsets=1
    patches = array_to_patches(sample_array_3d, tile_size=5, n_offsets=1)
    
    # Should create 4 patches for main grid + 4 patches for offset [2, 2]
    assert patches.shape[0] > 4
    
    # First 4 patches should match the non-offset patches
    expected_first_patch = sample_array_3d[0:5, 0:5, :]
    np.testing.assert_array_equal(patches[0], expected_first_patch)
    
    # There should be a patch containing the region with offset [2, 2]
    offset_patch_found = False
    for i in range(patches.shape[0]):
        if np.array_equal(patches[i], sample_array_3d[2:7, 2:7, :]):
            offset_patch_found = True
            break
    
    assert offset_patch_found, "No patch found with offset [2, 2]"

def test_array_to_patches_no_border_check(sample_array_3d):
    """Test array_to_patches with border_check=False."""
    # Test with border_check=False
    patches = array_to_patches(sample_array_3d, tile_size=6, border_check=False)
    
    # Should create only 1 patch of size 6x6x3 (since 10x10 can fit only one 6x6 patch without borders)
    assert patches.shape == (1, 6, 6, 3)
    
    # Check the content of the patch
    expected_patch = sample_array_3d[0:6, 0:6, :]
    np.testing.assert_array_equal(patches[0], expected_patch)

def test_array_to_patches_with_border_check(sample_array_3d):
    """Test array_to_patches with border_check=True."""
    # Test with border_check=True
    patches = array_to_patches(sample_array_3d, tile_size=6, border_check=True)
    
    # Should create 4 patches of size 6x6x3 (including borders)
    assert patches.shape == (4, 6, 6, 3)
    
    # Check the content of the border patches
    expected_border_patches = [
        sample_array_3d[0:6, 0:6, :],  # Top-left
        sample_array_3d[0:6, 4:10, :],  # Top-right
        sample_array_3d[4:10, 0:6, :],  # Bottom-left
        sample_array_3d[4:10, 4:10, :]  # Bottom-right
    ]
    
    for i, expected_patch in enumerate(expected_border_patches):
        patch_found = False
        for j in range(patches.shape[0]):
            if np.array_equal(patches[j], expected_patch):
                patch_found = True
                break
        
        assert patch_found, f"Border patch {i} not found"

def test_array_to_patches_channel_first(sample_array_channel_first):
    """Test array_to_patches with channel_first=False."""
    # Test with channel_last=False
    patches = array_to_patches(sample_array_channel_first, tile_size=5, channel_last=False)
    
    # Should create 4 patches of size 3x5x5 (channel-first)
    assert patches.shape == (4, 3, 5, 5)
    
    # Check the content of the first patch (top-left)
    expected_first_patch = sample_array_channel_first[:, 0:5, 0:5]
    np.testing.assert_array_equal(patches[0], expected_first_patch)

def test_array_to_patches_with_large_tile_size(sample_array_3d):
    """Test array_to_patches with a tile size equal to the array dimensions."""
    # Test with tile_size equal to array dimensions
    patches = array_to_patches(sample_array_3d, tile_size=10)
    
    # Should create 1 patch of size 10x10x3
    assert patches.shape == (1, 10, 10, 3)
    
    # The patch should be identical to the original array
    np.testing.assert_array_equal(patches[0], sample_array_3d)

def test_array_to_patches_with_small_array():
    """Test array_to_patches with a small array."""
    # Create a small 3D array
    small_array = np.zeros((5, 5, 3), dtype=np.float32)
    
    # Fill with test data
    for i in range(5):
        for j in range(5):
            small_array[i, j, 0] = i
            small_array[i, j, 1] = j
            small_array[i, j, 2] = i + j
    
    # Test with border_check=True
    patches = array_to_patches(small_array, tile_size=3, border_check=True)
    
    # Should create 4 patches of size 3x3x3 (including borders)
    assert patches.shape == (4, 3, 3, 3)
    
    # Check patch content for one of the patches
    expected_top_left = small_array[0:3, 0:3, :]
    assert np.any(np.all(patches == expected_top_left, axis=(1, 2, 3)))
