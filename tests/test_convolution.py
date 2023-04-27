""" Tests for raster/convolution.py """


# Standard library
import sys; sys.path.append("../")

# External
import numpy as np
import pytest

# Internal
from buteo.raster.convolution import (
    _weight_distance,
    get_kernel,
    pad_array,
    convolve_array,
    _METHOD_ENUMS,
)

# def test_weight_distance():
#     """ Test weight_distance() """
#     arr = np.array([1.0, 0.0, 0.0])
#     weights = weight_distance(arr)

#     assert weights == 1.0

#     weights = weight_distance(arr, method="linear", decay=0.5)
#     assert np.isclose(weights, 0.5)

#     weights = weight_distance(arr, method="sqrt", decay=0.5)
#     assert np.isclose(weights, np.sqrt(2) / 2)

#     weights = weight_distance(arr, method="power", decay=0.5)
#     assert np.isclose(weights, 0.25)

#     weights = weight_distance(arr, method="log")
#     assert np.isclose(weights, 1.0986122886681098)

#     weights = weight_distance(arr, method="gaussian", sigma=1)
#     assert np.isclose(weights, 0.6065306597126334)

#     with pytest.raises(ValueError):
#         weight_distance(arr, method="unknown")


# def test_get_kernel_simple():
#     """ Test get_kernel() """
#     kernel, weights, offsets = get_kernel(size=3, depth=1)

#     assert kernel.shape == (3, 3, 1)
#     assert len(weights) > 0
#     assert offsets.shape == (len(weights), 3)

#     with pytest.raises(AssertionError):
#         get_kernel(size=2)

#     with pytest.raises(AssertionError):
#         get_kernel(size=4)

#     assert np.isclose(kernel.sum(), weights.sum())
#     assert np.isclose(kernel.sum(), 1.0)

# def test_get_kernel_complex_1():
#     """ Test get_kernel() 1 """
#     kernel, weights, offsets = get_kernel(size=3, depth=3, distance_weight="gaussian", distance_sigma=1)

#     assert kernel.shape == (3, 3, 3)
#     assert len(weights) > 0
#     assert offsets.shape == (len(weights), 3)

#     assert np.isclose(kernel.sum(), weights.sum())
#     assert np.isclose(kernel.sum(), 3.0)

#     kernel, weights, offsets = get_kernel(size=3, depth=3, distance_weight="gaussian", distance_sigma=1, multi_dimensional=True)

#     assert kernel.shape == (3, 3, 3)
#     assert len(weights) > 0
#     assert offsets.shape == (len(weights), 3)

#     assert np.isclose(kernel.sum(), weights.sum())
#     assert np.isclose(kernel.sum(), 1.0)


# def test_get_kernel_complex_2():
#     """ Test get_kernel() 2 """
#     kernel, weights, offsets = get_kernel(size=7, depth=1, distance_weight="none", spherical=True)

#     assert kernel.shape == (7, 7, 1)
#     assert len(weights) > 0
#     assert offsets.shape == (len(weights), 3)

#     assert np.isclose(kernel.sum(), weights.sum())
#     assert np.isclose(kernel.sum(), 1.0)

# def test_get_kernel_complex_3():
#     """ Test get_kernel() 3 """
#     kernel, weights, offsets = get_kernel(size=5, depth=1, distance_weight="linear", spherical=True)

#     assert kernel.shape == (5, 5, 1)
#     assert len(weights) > 0
#     assert offsets.shape == (len(weights), 3)

#     assert np.isclose(kernel.sum(), weights.sum())
#     assert np.isclose(kernel.sum(), 1.0)

# def test_pad_array():
#     """ Basic tests for pad_array (view) """
#     arr = np.random.rand(3, 3, 1)
#     padded_arr = pad_array(arr, pad_size=1)

#     assert padded_arr.shape == (5, 5, 1)
#     assert np.allclose(padded_arr[1:4, 1:4], arr)

#     arr = np.random.rand(3, 3, 3)
#     padded_arr = pad_array(arr, pad_size=2)

#     assert padded_arr.shape == (7, 7, 3)
#     assert np.allclose(padded_arr[2:5, 2:5], arr)

#     arr = np.zeros((3, 3, 1))
#     padded_arr = pad_array(arr, pad_size=0)

#     assert padded_arr.shape == (3, 3, 1)
#     assert np.allclose(padded_arr, arr)

# def test_convolve_array_invalid_method():
#     """ Test convolve_array() with invalid method """
#     arr = np.random.rand(15, 15, 1).astype("float32")
#     _kernel, weights, offsets = get_kernel(3)

#     with pytest.raises(Exception):
#         convolve_array(arr, offsets, weights, method=-1)

def test_convolve_array_sum():
    """ Test convolve_array() with method=sum """
    arr = np.random.rand(15, 15, 1).astype("float32")
    _kernel, weights, offsets = get_kernel(3, normalise=False)

    result = convolve_array(arr, offsets, weights, method=_METHOD_ENUMS["sum"], normalise_edges=False) # sum

    assert np.alltrue(result > arr), "Result should be greater than input"
    assert result.shape == (15, 15, 1)

# def test_convolve_array_min():
#     """ Test convolve_array() with method=min """
#     arr = np.random.rand(5, 5, 1).astype("float32")
#     _kernel, weights, offsets = get_kernel(3, normalise=True)

#     result = convolve_array(arr, offsets, weights, method=METHOD_ENUMS["min"]) # min

#     assert np.alltrue(result <= arr), "Result should be less than or equal to the input"
#     assert result.shape == (5, 5, 1)

# def test_convolve_array_max():
#     """ Test convolve_array() with method=max """
#     arr = np.random.rand(7, 7, 3).astype("float32")
#     _kernel, weights, offsets = get_kernel(3, depth=3, normalise=True)

#     result = convolve_array(arr, offsets, weights, method=METHOD_ENUMS["max"]) # max

#     assert np.alltrue(result >= arr), "Result should be less than or equal to the input"
#     assert result.shape == (7, 7, 3)

# def test_convolve_array_min_v2():
#     """ Test convolve_array() with method=min """
#     arr = np.random.rand(7, 7, 3).astype("float32")
#     _kernel, weights, offsets = get_kernel(3, depth=3, normalise=True)

#     result = convolve_array(arr, offsets, weights, method=METHOD_ENUMS["min"]) # min

#     assert np.alltrue(result <= arr), "Result should be less than or equal to the input"
#     assert result.shape == (7, 7, 3)

# def test_convolve_array_same_size_result():
#     """ Test convolve_array() with all methods """
#     arr = np.random.rand(15, 15, 1).astype("float32")

#     for method in METHOD_ENUMS.values():
#         _kernel, weights, offsets = get_kernel(3)
#         result = convolve_array(arr, offsets, weights, method=method)
#         assert result.shape == (15, 15, 1)
