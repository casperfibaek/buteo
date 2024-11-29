"""### Generic utility functions. ###

Functions that make interacting with the toolbox easier.
"""

# Standard Library
import time
from typing import Any, Union, List, Tuple



def _get_variable_as_list(
    variable_or_list: Union[List, Any],
) -> List[Any]:
    """Ensures that a variable is a list. If the variable is a list, return the list.
    If the variable is not a list, return a list with the variable as the only element.

    Parameters
    ----------
    variable_or_list : Union[List, Any]
        The variable to check.

    Returns
    -------
    List[Any]
        The variable as a list.
    """
    if variable_or_list is None:
        return []

    if isinstance(variable_or_list, list):
        return variable_or_list

    return [variable_or_list]


def _get_unix_seconds_as_str() -> str:
    """Get a string of the current UNIX time in seconds.

    Returns
    -------
    str
        A string of the current UNIX time in seconds.
    """
    return str(int(time.time()))


def _get_time_as_str() -> str:
    """Gets the current time as a string.
    in the format: YYYYMMDD_HHMMSS

    Returns
    -------
    str
        The current time as a string.
    """
    return time.strftime("%Y%m%d_%H%M%S")


def _check_variable_is_float(variable: Any) -> bool:
    """Check if a variable is a float.
    If it is a string, see if it a representation of a float.

    Parameters
    ----------
    variable : Any
        The variable to check.

    Returns
    -------
    bool
        True if the variable is a float or float representation, False otherwise.
    """
    if variable is None:
        return False

    if isinstance(variable, float):
        return True

    if isinstance(variable, str):
        try:
            float(variable)
            return True
        except ValueError:
            return False

    return False


def _check_variable_is_int(variable: Any) -> bool:
    """Check if a variable is an integer.
    If it is a string, see if it a representation of an integer.
    If it is a float, return False.

    Parameters
    ----------
    variable : Any
        The variable to check.

    Returns
    -------
    bool
        True if the variable is an integer or integer representation, False otherwise.
    """
    if variable is None:
        return False

    if isinstance(variable, int):
        return True

    if isinstance(variable, str):
        try:
            int(variable)
            return True
        except ValueError:
            return False

    return False


def _check_variable_is_number_type(variable: Any) -> bool:
    """Check if variable is a number.

    Parameters
    ----------
    variable : Any
        The variable to check.

    Returns
    -------
    bool
        True if the variable is a number type, False otherwise.
    """
    if variable is None:
        return False

    if isinstance(variable, (float, int)):
        return True

    return False


def _get_variable_as_number(
    variable: Any,
) -> Union[int, float]:
    """Attempts to convert a variable to a number.

    Parameters
    ----------
    variable : Any
        The variable to convert.

    Returns
    -------
    Union[int, float]
        The variable as an int or float.
    """
    if variable is None:
        raise ValueError("Cannot convert None to a number.")

    if not isinstance(variable, (str, int, float)):
        raise TypeError("value must be a string, integer or float.")

    if _check_variable_is_int(variable):
        return int(variable)

    if _check_variable_is_float(variable):
        return float(variable)

    raise ValueError(f"Could not convert {variable} to a number.")


def _ensure_negative(
    number: Union[int, float]
) -> Union[int, float]:
    """Ensures that a valid is negative. If the number is positive, it is made negative.

    Parameters
    ----------
    number : Union[int, float]
        A number.

    Returns
    -------
    Union[int, float]
        The number, made negative if it was positive.
    """
    if number is None:
        raise ValueError("Cannot process None value.")

    if not _check_variable_is_number_type(number):
        raise TypeError(f"number must be a number. Received: {type(number)}")

    if number <= 0:
        return number

    return -number


def _ensure_positive(
    number: Union[int, float]
) -> Union[int, float]:
    """Ensures that a valid is positive. If the number is negative, it is made positive.

    Parameters
    ----------
    number : Union[int, float]
        A number.

    Returns
    -------
    Union[int, float]
        The number, made positive if it was negative.
    """
    if number is None:
        raise ValueError("Cannot process None value.")

    if not _check_variable_is_number_type(number):
        raise TypeError(f"number must be a number. Received: {type(number)}")

    if number >= 0:
        return number

    return -number


def _check_variable_is_iterable_or_type(
    potential_type: Any
) -> bool:
    """Recursively check if a variable is a type, list, or tuple.

    Parameters
    ----------
    potential_type : Any
        The variable to check.

    Returns
    -------
    bool
        True if the variable is a type, list, or tuple, False otherwise.
    """
    if potential_type is None:
        return True

    if isinstance(potential_type, type):
        return True

    if isinstance(potential_type, (list, tuple)):
        return all(_check_variable_is_iterable_or_type(item) for item in potential_type)

    return False


def _normalise_type(t):
    """Convert a type specification to a standard format.
    
    Parameters
    ----------
    t : Any
        The type specification to normalize.
        
    Returns
    -------
    Union[type, Tuple[type]]
        The normalized type specification.
        
    Raises
    ------
    TypeError
        If the type specification is invalid.
    """
    if t is None:
        return type(None)
    if isinstance(t, type):
        return t
    if isinstance(t, (list, tuple)):
        if not all(isinstance(st, type) for st in t):
            raise TypeError(f"Invalid nested type specification: {t}")
        return tuple(t)
    raise TypeError(f"Invalid type specification: {t}")


def _type_check(
    variable: Any,
    types: Union[List[Union[type, List[type], None]], Tuple[Union[type, List[type], None], ...]],
    name: str = "",
    *,
    throw_error: bool = True,
) -> bool:
    """Type check function that supports nested types and collections.
    
    Parameters
    ----------
    variable : Any
        The variable to check.
    types : Union[List[Union[type, List[type], None]], Tuple[Union[type, List[type], None], ...]]
        The type or types to check against.
    name : str, optional
        The name of the variable to check.
    throw_error : bool, optional
        Whether to throw an error if the type check fails.

    Returns
    -------
    bool
        True if the variable matches any of the types, False otherwise.

    Raises
    ------
    TypeError
        If the variable is not found in the list of types

    Examples
    --------
    >>> _type_check("hello", [str])  # True
    >>> _type_check([1, 2, 3], [[int]])  # True
    >>> _type_check([1, "a"], [[int]])  # False
    >>> _type_check(None, [str, None])  # True
    >>> _type_check({"a": 1}, [dict])  # True
    """
    # Input validation
    if not isinstance(name, str):
        raise TypeError("name must be a string")
    if not isinstance(types, (list, tuple)):
        raise TypeError("types must be a list or tuple")

    # Normalize all type specifications
    valid_types = [_normalise_type(t) for t in types]

    # Check if variable matches any of the valid types
    for valid_type in valid_types:
        # Handle nested type checking (e.g., [[int]] for list of integers)
        if isinstance(valid_type, tuple):
            if isinstance(variable, (list, tuple)):
                if not variable or all(isinstance(item, valid_type) for item in variable):
                    return True
        # Handle simple type checking
        elif isinstance(variable, valid_type):
            return True

    # Handle error reporting
    if throw_error:
        actual_type = type(variable).__name__
        if isinstance(variable, (list, tuple)):
            actual_type = f"[{type(variable[0]).__name__}]" if variable else "[]"

        expected_types = []
        for t in valid_types:
            if isinstance(t, tuple):
                expected_types.append(f"[{','.join(st.__name__ for st in t)}]")
            else:
                expected_types.append(t.__name__)

        raise TypeError(
            f"Type mismatch for '{name}': Expected {' or '.join(expected_types)}, got {actual_type}"
        )

    return False


def _check_number_is_within_threshold(
    number: Union[int, float],
    target: Union[int, float],
    threshold: Union[int, float],
) -> bool:
    """Check if a number is within a threshold of a target.

    Parameters
    ----------
    number : Union[int, float]
        The number to check.
    target : Union[int, float]
        The target number.
    threshold : Union[int, float]
        The threshold value (must be positive).

    Returns
    -------
    bool
        True if the number is within the threshold of the target, False otherwise.

    Raises
    ------
    ValueError
        If any input is None or if threshold is negative.
    TypeError
        If any input is not a number.
    """
    if any(x is None for x in [number, target, threshold]):
        raise ValueError("None values are not allowed")

    if not all(_check_variable_is_number_type(x) for x in [number, target, threshold]):
        raise TypeError("All inputs must be numbers")

    if threshold < 0:
        raise ValueError("Threshold must be positive")

    return abs(number - target) <= threshold


def _check_number_is_within_range(
    number: Union[int, float],
    min_value: Union[int, float],
    max_value: Union[int, float],
) -> bool:
    """Check if a number is within a range.

    Parameters
    ----------
    number : Union[int, float]
        The number to check.
    min_value : Union[int, float]
        The minimum value.
    max_value : Union[int, float]
        The maximum value.

    Returns
    -------
    bool
        True if the number is within the range, False otherwise.

    Raises
    ------
    ValueError
        If any input is None or if min_value > max_value.
    TypeError
        If any input is not a number.
    """
    if any(x is None for x in [number, min_value, max_value]):
        raise ValueError("None values are not allowed")

    if not all(_check_variable_is_number_type(x) for x in [number, min_value, max_value]):
        raise TypeError("All inputs must be numbers")

    if min_value > max_value:
        raise ValueError("min_value cannot be greater than max_value")

    return min_value <= number <= max_value
