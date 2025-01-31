# pylint: skip-file
# type: ignore

import sys; sys.path.append("../../")

import pytest
import numpy as np
from buteo.array.utils_array import _create_grid


def test_create_grid_basic():
    rows = np.array([0, 1, 2])
    cols = np.array([0, 1, 2])
    rows_grid, cols_grid = _create_grid(rows, cols)
    
    assert rows_grid.shape == (3, 3)
    assert cols_grid.shape == (3, 3)
    
    expected_rows = np.array([
        [0, 0, 0],
        [1, 1, 1], 
        [2, 2, 2]
    ])
    expected_cols = np.array([
        [0, 1, 2],
        [0, 1, 2],
        [0, 1, 2]
    ])
    
    np.testing.assert_array_equal(rows_grid, expected_rows)
    np.testing.assert_array_equal(cols_grid, expected_cols)

# TODO: Fix this test
# def test_create_grid_rectangular():
#     rows = np.array([0, 1])
#     cols = np.array([0, 1, 2])
#     rows_grid, cols_grid = _create_grid(rows, cols)
    
#     assert rows_grid.shape == (2, 3)
#     assert cols_grid.shape == (2, 3)
    
#     expected_rows = np.array([
#         [0, 0, 0],
#         [1, 1, 1]
#     ])
#     expected_cols = np.array([
#         [0, 1, 2],
#         [0, 1, 2]
#     ])
    
#     np.testing.assert_array_equal(rows_grid, expected_rows) 
#     np.testing.assert_array_equal(cols_grid, expected_cols)

def test_create_grid_single():
    rows = np.array([0])
    cols = np.array([0])
    rows_grid, cols_grid = _create_grid(rows, cols)
    
    assert rows_grid.shape == (1, 1)
    assert cols_grid.shape == (1, 1)
    
    np.testing.assert_array_equal(rows_grid, np.array([[0]]))
    np.testing.assert_array_equal(cols_grid, np.array([[0]]))