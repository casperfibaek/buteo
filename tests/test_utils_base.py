# pylint: skip-file
# type: ignore

# Standard library
import sys; sys.path.append("../")
import pytest


from buteo.utils.utils_base import (
    _get_variable_as_list, _get_unix_seconds_as_str, 
    _get_time_as_str, _check_variable_is_float,
    _check_variable_is_int, _check_variable_is_number_type,
    _get_variable_as_number, _ensure_negative, _ensure_positive,
    _check_variable_is_iterable_or_type, _type_check,
    _check_number_is_within_threshold, _check_number_is_within_range,
)

# Test _get_variable_as_list
def test_get_variable_as_list():
    assert _get_variable_as_list(None) == []
    assert _get_variable_as_list([1, 2, 3]) == [1, 2, 3]
    assert _get_variable_as_list("test") == ["test"]
    assert _get_variable_as_list(123) == [123]

# Test _check_variable_is_float
def test_check_variable_is_float():
    assert _check_variable_is_float(1.0) is True
    assert _check_variable_is_float("1.0") is True
    assert _check_variable_is_float(1) is False
    assert _check_variable_is_float("abc") is False
    assert _check_variable_is_float(None) is False

# Test _check_variable_is_int
def test_check_variable_is_int():
    assert _check_variable_is_int(1) is True
    assert _check_variable_is_int("1") is True
    assert _check_variable_is_int(1.0) is False
    assert _check_variable_is_int("1.0") is False
    assert _check_variable_is_int(None) is False

# Test _check_variable_is_number_type
def test_check_variable_is_number_type():
    assert _check_variable_is_number_type(1) is True
    assert _check_variable_is_number_type(1.0) is True
    assert _check_variable_is_number_type("1") is False
    assert _check_variable_is_number_type(None) is False

# Test _get_variable_as_number
def test_get_variable_as_number():
    assert _get_variable_as_number("1") == 1
    assert _get_variable_as_number("1.0") == 1.0
    assert _get_variable_as_number(1) == 1
    assert _get_variable_as_number(1.0) == 1.0
    
    with pytest.raises(ValueError):
        _get_variable_as_number(None)
    with pytest.raises(TypeError):
        _get_variable_as_number([1])
    with pytest.raises(ValueError):
        _get_variable_as_number("abc")

# Test _ensure_negative
def test_ensure_negative():
    assert _ensure_negative(-1) == -1
    assert _ensure_negative(1) == -1
    assert _ensure_negative(-1.5) == -1.5
    assert _ensure_negative(1.5) == -1.5
    
    with pytest.raises(ValueError):
        _ensure_negative(None)
    with pytest.raises(TypeError):
        _ensure_negative("1")

# Test _ensure_positive
def test_ensure_positive():
    assert _ensure_positive(1) == 1
    assert _ensure_positive(-1) == 1
    assert _ensure_positive(1.5) == 1.5
    assert _ensure_positive(-1.5) == 1.5
    
    with pytest.raises(ValueError):
        _ensure_positive(None)
    with pytest.raises(TypeError):
        _ensure_positive("1")

# Test _check_variable_is_iterable_or_type
def test_check_variable_is_iterable_or_type():
    assert _check_variable_is_iterable_or_type(None) is True
    assert _check_variable_is_iterable_or_type(str) is True
    assert _check_variable_is_iterable_or_type([str, int]) is True
    assert _check_variable_is_iterable_or_type((str, int)) is True
    assert _check_variable_is_iterable_or_type("string") is False
    assert _check_variable_is_iterable_or_type(123) is False

    # Test _type_check
    def test_type_check():
        # Basic type checking
        assert _type_check("test", [str]) is True
        assert _type_check(123, [int]) is True
        assert _type_check(1.0, [float]) is True
        assert _type_check(None, [None]) is True
        assert _type_check("test", [int]) is False

        # Multiple allowed types
        assert _type_check("test", [str, int]) is True 
        assert _type_check(123, [str, int]) is True
        assert _type_check(1.0, [str, int]) is False

        # List type checking
        assert _type_check([1, 2, 3], [[int]]) is True
        assert _type_check([1, "2", 3], [[int]]) is False
        assert _type_check([], [[int]]) is True
        assert _type_check(["a", "b"], [[str]]) is True

        # Mixed types
        assert _type_check([1, 2], [list, [int]]) is True
        assert _type_check(None, [str, None]) is True

        # Error cases
        with pytest.raises(TypeError):
            _type_check("test", "invalid_type")
        
        with pytest.raises(TypeError):
            _type_check("test", [1, 2, 3])

        with pytest.raises(ValueError):
            _type_check(123, [str], "variable", throw_error=True)

        # Name parameter testing
        with pytest.raises(TypeError):
            _type_check("test", [str], name=123)

        # Test error message format
        try:
            _type_check(123, [str], "test_var")
            assert False
        except ValueError as e:
            assert str(e) == "Type mismatch for 'test_var': Expected str, got int"

# Test _check_number_is_within_threshold
def test_check_number_is_within_threshold():
    assert _check_number_is_within_threshold(10, 12, 2) is True
    assert _check_number_is_within_threshold(10, 13, 2) is False
    assert _check_number_is_within_threshold(1.0, 1.1, 0.2) is True
    
    with pytest.raises(ValueError):
        _check_number_is_within_threshold(None, 10, 2)
    with pytest.raises(TypeError):
        _check_number_is_within_threshold("10", 10, 2)
    with pytest.raises(ValueError):
        _check_number_is_within_threshold(10, 12, -2)

# Test _check_number_is_within_range
def test_check_number_is_within_range():
    assert _check_number_is_within_range(5, 0, 10) is True
    assert _check_number_is_within_range(0, 0, 10) is True
    assert _check_number_is_within_range(10, 0, 10) is True
    assert _check_number_is_within_range(-1, 0, 10) is False
    assert _check_number_is_within_range(11, 0, 10) is False
    
    with pytest.raises(ValueError):
        _check_number_is_within_range(None, 0, 10)
    with pytest.raises(TypeError):
        _check_number_is_within_range("5", 0, 10)
    with pytest.raises(ValueError):
        _check_number_is_within_range(5, 10, 0)

# Test time-based functions
def test_time_functions():
    # Basic format checks
    unix_time = _get_unix_seconds_as_str()
    assert unix_time.isdigit()
    assert len(unix_time) >= 10
    
    time_str = _get_time_as_str()
    assert len(time_str) == 15
    assert "_" in time_str