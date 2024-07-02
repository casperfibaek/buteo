""" Tests for raster/convolution.py """


# Standard library
import sys; sys.path.append("../")

# External
import numpy as np
import pytest

# Internal
from buteo.array.convolution import (
    pad_array,
    convolve_array,
)
from buteo.array.convolution_kernels import (
    kernel_base,
    kernel_get_offsets_and_weights,
)


def test_get_kernel_simple():
    """ Test get_kernel() """
    kernel = kernel_base(1)
    offsets, weights = kernel_get_offsets_and_weights(kernel, remove_zero_weights=True)

    assert kernel.shape == (3, 3)
    assert len(weights) > 0
    assert len(offsets) == len(weights)

    assert np.isclose(kernel.sum(), weights.sum())
    assert np.isclose(kernel.sum(), 1.0)

def test_get_kernel_complex_1():
    """ Test get_kernel() 1 """
    kernel = kernel_base(2)
    offsets, weights = kernel_get_offsets_and_weights(kernel, remove_zero_weights=True)

    assert kernel.shape == (5, 5)
    assert len(weights) > 0
    assert len(offsets) == len(weights)

    assert np.isclose(kernel.sum(), weights.sum())
    assert np.isclose(kernel.sum(), 1.0)

def test_pad_array():
    """ Basic tests for pad_array (view) """
    arr = np.random.rand(3, 3, 1)
    padded_arr = pad_array(arr, pad_size=1)

    assert padded_arr.shape == (5, 5, 1)
    assert np.allclose(padded_arr[1:4, 1:4], arr)

    arr = np.random.rand(3, 3, 3)
    padded_arr = pad_array(arr, pad_size=2)

    assert padded_arr.shape == (7, 7, 3)
    assert np.allclose(padded_arr[2:5, 2:5], arr)

    arr = np.zeros((3, 3, 1))
    padded_arr = pad_array(arr, pad_size=0)

    assert padded_arr.shape == (3, 3, 1)
    assert np.allclose(padded_arr, arr)

def test_convolve_array_invalid_method():
    """ Test convolve_array() with invalid method """
    arr = np.random.rand(15, 15, 1).astype("float32")
    kernel = kernel_base(3)
    offsets, weights = kernel_get_offsets_and_weights(kernel, remove_zero_weights=True)

    with pytest.raises(Exception):
        convolve_array(arr, offsets, weights, method=-1)

def test_convolve_array_sum():
    """ Test convolve_array() with method=sum """
    arr = np.random.rand(15, 15, 1).astype("float32")
    kernel = kernel_base(1, normalised=False, circular=True)
    offsets, weights = kernel_get_offsets_and_weights(kernel, remove_zero_weights=True)

    result = convolve_array(arr, offsets, weights, method=1) # sum

    assert np.all(result > arr), "Result should be greater than input"
    assert result.shape == (15, 15, 1)
