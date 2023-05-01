"""
### Generic utility functions ###

Functions that make interacting with the toolbox easier.
"""

# Standard Library
import time
from typing import Any, Union, List, Tuple



def _get_variable_as_list(
    variable_or_list: Union[List, Any],
) -> List[Any]:
    """
    Ensures that a variable is a list. If the variable is a list, return the list.
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
    if isinstance(variable_or_list, list):
        return variable_or_list

    return [variable_or_list]


def _get_unix_seconds_as_str() -> str:
    """
    Get a string of the current UNIX time in seconds.

    Returns
    -------
    str
        A string of the current UNIX time in seconds.
    """
    return str(int(time.time()))


def _check_variable_is_float(variable: Any) -> bool:
    """
    Check if a variable is a float.
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
    """
    Check if a variable is an integer.
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
    """
    Check if variable is a number.

    Parameters
    ----------
    variable : Any
        The variable to check.

    Returns
    -------
    bool
        True if the variable is a number type, False otherwise.
    """
    if isinstance(variable, (float, int)):
        return True

    return False


def _get_variable_as_number(
    variable: Any,
) -> float:
    """
    Attempts to convert a variable to a number.

    Parameters
    ----------
    variable : Any
        The variable to convert.

    Returns
    -------
    float
        The variable as a float.
    """
    assert isinstance(variable, (str, int, float)), "value must be a string, integer or float."
    if _check_variable_is_int(variable):
        return int(variable)

    if _check_variable_is_float(variable):
        return float(variable)

    raise RuntimeError(f"Could not convert {variable} to a number.")


def _ensure_negative(number: Union[int, float]) -> Union[int, float]:
    """
    Ensures that a valid is negative. If the number is positive, it is made negative.

    Parameters
    ----------
    number : int, float
        A number.

    Returns
    -------
    int, float
        The number, made negative if it was positive.
    """
    assert _check_variable_is_number_type(number), f"number must be a number. Received: {number}"

    if number <= 0:
        return number

    return -number


def _check_recursive_iterable_or_type(potential_type: Any) -> bool:
    """
    Recursively check if a variable is a type, list, or tuple.

    Parameters
    ----------
    potential_type : Any
        The variable to check.

    Returns
    -------
    bool
        True if the variable is a type, list, or tuple, False otherwise.
    """
    if isinstance(potential_type, type(None)):
        return True

    if isinstance(potential_type, type):
        return True

    if isinstance(potential_type, (list, tuple)):
        for item in potential_type:
            if not _check_recursive_iterable_or_type(item):
                return False
        return True

    return False


def type_check(
    variable: Any,
    types: Tuple[type, ...],
    name: str = "",
    *,
    throw_error: bool = True,
) -> bool:
    """
    Utility function to type check the inputs of a function. Checks recursively.
    This is useful for the external facing functions of a module.

    Use like this:
    ```python
    >>> def my_function(
    ...     my_variable: Optional[Union[str, float, List[str]]] = None
    ... ) -> None:
    ...     type_check(my_variable, [str, float, [str], None], "my_variable")
    ...     # Do stuff
    ```

    Parameters
    ----------
    variable : Any
        The variable to check.
    
    types : Tuple[type, ...]
        The types to check against.

    name : str, optional
        The name of the variable to check.

    throw_error : bool, optional
        If True, throw an error if the type check fails. If False, return False if the type check fails.
    
    Returns
    -------
    bool
        True if the type check passes, False otherwise.
    """
    assert isinstance(name, str), "name must be a string."
    assert _check_recursive_iterable_or_type(types), f"types must be a type, list, None, or tuple. not: {types}"

    if not isinstance(types, (list, tuple)):
        types = [types]

    valid_types = []
    for valid_type in types:
        if valid_type is None:
            valid_types.append(type(None))
        elif isinstance(valid_type, type):
            valid_types.append(valid_type)
        elif isinstance(valid_type, (list, tuple)):
            valid_types.append(valid_type)
        else:
            raise ValueError(f"Invalid type: {valid_type}")

    if not isinstance(variable, (list, tuple)):
        sublist_valid_types = []
        for valid_type in valid_types:
            if not isinstance(valid_type, (list, tuple)):
                sublist_valid_types.append(valid_type)

        for valid_type in sublist_valid_types:
            if isinstance(variable, valid_type):
                return True

    if type(variable) in valid_types:
        return True

    type_list = [type(val) for val in valid_types]

    if isinstance(variable, list) and type([]) in type_list:
        for sublist in valid_types:
            if not isinstance(sublist, list):
                continue

            if len(sublist) == 0:
                return True

            found = 0
            for item in variable:
                if type(item) in sublist:
                    found += 1

            if found == len(variable):
                return True

    if isinstance(variable, tuple) and type(()) in type_list:
        for sublist in valid_types:
            if not isinstance(sublist, tuple):
                continue

            if len(sublist) == 0:
                return True

            found = 0
            for item in variable:
                if type(item) in sublist:
                    found += 1

            if found == len(variable):
                return True
    if throw_error:
        raise ValueError(
            f"The type of variable {name} is not valid. Expected: {types}, got: {type(variable)}"
        )

    return False
