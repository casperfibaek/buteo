# pylint: skip-file
# type: ignore

# Standard library
import sys; sys.path.append("../")

import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock

from buteo.utils.utils_aux import (
    _print_progress,
    _get_timing,
    _get_folder_size, 
    _force_garbage_collect_all,
    split_number,
)

# Fixtures
@pytest.fixture
def mock_terminal_size():
    return MagicMock(columns=80)

@pytest.fixture
def temp_folder(tmp_path):
    # Create test files
    test_file1 = tmp_path / "test1.txt"
    test_file2 = tmp_path / "test2.txt"
    test_file1.write_text("test content 1")
    test_file2.write_text("test content 2")
    return tmp_path

# Test _print_progress
def test_print_progress_validation():
    with pytest.raises(AssertionError):
        _print_progress("1", 10)
    with pytest.raises(AssertionError):
        _print_progress(1, "10")
    with pytest.raises(AssertionError):
        _print_progress(1, 10, name=123)

@patch('sys.stdout')
@patch('os.get_terminal_size', return_value=MagicMock(columns=80))
def test_print_progress_display(mock_terminal, mock_stdout):
    _print_progress(5, 10, "Test")
    mock_stdout.write.assert_called()
    _print_progress(10, 10, "Test")
    assert mock_stdout.write.call_count >= 2

# Test _get_timing
def test_get_timing():
    before = datetime.now()
    result = _get_timing(before, print_msg=False)
    assert isinstance(result, str)
    assert "Processing took:" in result
    assert "h" in result and "m" in result and "s" in result

def test_get_timing_validation():
    with pytest.raises(AssertionError):
        _get_timing("not a datetime")

# Test _get_folder_size
def test_get_folder_size(temp_folder):
    size = _get_folder_size(str(temp_folder), rough=False)
    assert isinstance(size, int)
    assert size > 0

def test_get_folder_size_rough(temp_folder):
    size = _get_folder_size(str(temp_folder), rough=True)
    assert isinstance(size, int)

def test_get_folder_size_validation():
    with pytest.raises(AssertionError):
        _get_folder_size(123)
    with pytest.raises(AssertionError):
        _get_folder_size(".", rough="not_bool")

# Test _force_garbage_collect_all
@patch('gc.collect')
def test_force_garbage_collect(mock_gc):
    result = _force_garbage_collect_all()
    assert result is True
    mock_gc.assert_called_once()

def test_force_garbage_collect_with_functions():
    result = _force_garbage_collect_all(delete_functions=False)
    assert result is True

# Test split_number
@pytest.mark.parametrize("num,parts,expected", [
    (10, 3, [4, 3, 3]),
    (10, 4, [3, 3, 2, 2]),
    (100, 3, [34, 33, 33]),
    (5, 5, [1, 1, 1, 1, 1]),
])
def test_split_number(num, parts, expected):
    result = split_number(num, parts)
    assert result == expected
    assert sum(result) == num
    assert len(result) == parts

def test_split_number_validation():
    with pytest.raises(AssertionError):
        split_number("10", 3)
    with pytest.raises(AssertionError):
        split_number(10, "3")
    with pytest.raises(AssertionError):
        split_number(3, 10)  # num < parts
