"""
### Generic utility functions ###

Functions that make interacting with the toolbox easier.
"""

# Standard Library
import os
import sys
import time
import shutil
from glob import glob
from uuid import uuid4
from datetime import datetime
from pathlib import PurePosixPath
from typing import Any, Union, List, Optional, Dict, Tuple
from warnings import warn

# Internal
from buteo.utils.gdal_enums import get_valid_raster_driver_extensions, get_valid_vector_driver_extensions

# External
import psutil
import numpy as np
from osgeo import gdal



def get_unix_seconds_as_str() -> str:
    """
    Get a string of the current UNIX time in seconds.

    Returns:
        str: A string of the current UNIX time in seconds.
    """
    return str(int(time.time()))


def is_float(value: Any) -> bool:
    """
    Check if a value is a float. If it is a string, try to convert it to a float.

    Args:
        value (any): The value to check.

    Returns:
        bool: True if the value is a float, False otherwise.
    """
    if isinstance(value, float):
        return True

    if isinstance(value, str):
        try:
            float(value)
            return True

        except ValueError:
            return False

    return False


def is_number(potential_number: Any) -> bool:
    """
    Check if variable is a number.

    Args:
        potential_number (any): The variable to check.

    Returns:
        bool: True if the variable is a number, False otherwise.
    """
    if isinstance(potential_number, float):
        return True

    if isinstance(potential_number, int):
        return True

    return False


def is_int(value: Any) -> bool:
    """
    Check if a value is an integer. If it is a string, try to convert it to an integer.

    Args:
        value (any): The value to check.

    Returns:
        bool: True if the variable is an int, False otherwise.
    """
    if isinstance(value, int):
        return True

    if isinstance(value, str):
        try:
            int(value)
            return True
        except ValueError:
            return False

    return False


def is_list_all_the_same(lst: list) -> bool:
    """
    Check if a list contains all the same elements.

    Args:
        lst (list): The list to check.

    Returns:
        bool: True if the list contains all the same elements, False otherwise.
    """
    assert isinstance(lst, list), "lst must be a list."

    if len(lst) == 0:
        return True

    first = lst[0]
    for idx, item in enumerate(lst):
        if idx == 0:
            continue

        if item != first:
            return False

    return True


def file_exists(path: str) -> bool:
    """
    Check if a file exists. Also checks vsimem.

    Args:
        path (str): The path to the file.

    Returns:
        bool: True if the file exists, False otherwise.
    """
    if not isinstance(path, str):
        return False

    if os.path.exists(path):
        return True

    try:
        if hasattr(gdal, "listdir"):
            if path in gdal.listdir("/vsimem"):
                return True
        elif hasattr(gdal, "ReadDir"):
            paths = ["/vsimem/" + ds for ds in gdal.ReadDir("/vsimem")]
            if path in paths:
                return True
        else:
            print("Warning, unable to access vsimem.")
    except AssertionError:
        pass

    return False


def folder_exists(path: str) -> bool:
    """
    Check if a folder exists.

    Args:
        path (str): The path to the folder.

    Returns:
        bool: True if the folder exists, False otherwise.
    """
    if isinstance(path, str):
        abs_dir = os.path.isdir(path)
        if abs_dir:
            return True

        abs_dir = os.path.isdir(os.path.abspath(path))
        if abs_dir:
            return True

    return False


def delete_files_in_folder(folder: str) -> bool:
    """
    Delete all files in a folder. Does not remove subfolders.

    Args:
        folder (str): The path to the folder.

    Returns:
        bool: True if the files were deleted, raises warning otherwise.
    """
    assert isinstance(folder, str), "folder must be a string."
    assert folder_exists(folder), "folder must exist."

    for file in glob(folder + "*.*"):
        try:
            os.remove(file)
        except Exception:
            warn(f"Warning. Could not remove: {file}", UserWarning)

    return True


def delete_folder(folder: str) -> bool:
    """
    Delete a folder.

    Args:
        folder (str): The path to the folder.

    Returns:
        bool: True if the folder was deleted, False otherwise.
    """
    assert isinstance(folder, str), "folder must be a string."
    assert folder_exists(folder), "folder must exist."

    try:
        shutil.rmtree(folder)
    except RuntimeError:
        warn(f"Warning. Could not remove: {folder}", UserWarning)

    return True


def delete_file(file: str) -> bool:
    """
    Delete a File

    Args:
        file (str): The path to the file.

    Returns:
        bool: True if the file was deleted, False otherwise.
    """
    assert isinstance(file, str), "file must be a string."
    assert file_exists(file), "file must exist."

    os.remove(file)

    if file_exists(file):
        return False

    return True


def to_number(variable):
    """
    Attempts to convert a variable to a number.

    Args:
        variable (any): The value to convert.

    Returns:
        (float): The value converted to a number.
    """
    assert isinstance(variable, (str, int, float)), "value must be a string, integer or float."
    if is_number(variable):
        return variable

    if is_float(variable):
        return float(variable)

    raise RuntimeError(f"Could not convert {variable} to a number.")


def make_dir_if_not_exists(path: str) -> str:
    """
    Make a directory if it doesn't exist.

    Args:
        path (str): The path to the directory.

    Returns:
        (str): The path to the created directory.
    """
    assert isinstance(path, str), "path must be a string."

    if not folder_exists(path):
        os.makedirs(path)
    
    if not folder_exists(path):
        raise RuntimeError(f"Could not create directory: {path}")

    return path


def path_to_ext(
    path: str,
    with_dot: bool = False,
):
    """
    Get the extension of a file. If the file has no extension, raise an error.

    Args:
        path (str): The path to the file.

    Kwargs:
        with_dot (bool = False): If True, return the extension with a dot.

    Returns:
        str: The extension of the file.
    """
    assert isinstance(path, str), "path must be a string."
    assert isinstance(with_dot, bool), "with_dot must be a boolean."

    basename = os.path.basename(path)
    basesplit = os.path.splitext(basename)
    ext = basesplit[1]

    if ext == "" or len(ext) == 1:
        raise RuntimeError (f"File: {path} has no extension.")

    if with_dot:
        return ext

    return ext[1:]


def path_to_folder(path: str) -> str:
    """
    Get the folder of a file.

    Args:
        path (str): The path to the file.

    Returns:
        str: The folder of the file.
    """
    assert isinstance(path, str), "path must be a string."

    dir_path = os.path.dirname(os.path.abspath(path))

    return dir_path


def change_path_ext(
    path: str,
    target_ext: str,
) -> str:
    """
    Change the extension of a file.

    Args:
        path (str): The path to the file.
        target_ext (str): The new extension.

    Returns:
        str: The path to the file with the new extension.
    """
    assert isinstance(path, str), "path must be a string."
    assert isinstance(target_ext, str), "target_ext must be a string."

    target_ext = target_ext.lstrip('.')
    basename = os.path.basename(path)
    basesplit = os.path.splitext(basename)
    ext = basesplit[1]

    if ext == "" or len(ext) == 1:
        raise RuntimeError(f"File: {path} has no extension.")

    return os.path.join(os.path.dirname(path), f"{basesplit[0]}.{target_ext}")



def is_valid_mem_path(path: str) -> bool:
    """
    Check if a path is a valid memory path that has an extension. vsizip also works.

    Args:
        path (str): The path to test.

    Returns:
        bool: True if path is a valid memory path, False otherwise.
    """
    if not isinstance(path, str):
        return False

    if len(path) < len("/vsimem/x"):
        return False

    ext = os.path.splitext(path)[1]

    if ext == "" or ext == ".":
        return False

    if path.startswith("/vsimem") or path.startswith("/vsizip"):
        return True

    return False


def is_valid_non_memory_path(path: str) -> bool:
    """
    Check if a path is valid, not in memory, and has an extension.

    Args:
        path (str): The path to the file.

    Returns:
        bool: True if the path has an extension, False otherwise.
    """
    if not isinstance(path, str):
        return False

    ext = os.path.splitext(path)[1]

    if ext in ["", ".", "/"]:
        return False

    if not folder_exists(path_to_folder(path)):
        return False

    if is_valid_mem_path(path):
        return False

    if os.path.isdir(path):
        return False

    return True


def is_valid_file_path(path: str) -> bool:
    """
    Check if a path is valid and has an extension. Path can be in memory.

    Args:
        path (str): The path to the file.

    Returns:
        bool: True if the path has an extension, False otherwise.
    """
    if is_valid_mem_path(path) or is_valid_non_memory_path(path):
        return True

    return False


def is_valid_output_path(path: str, *, overwrite: bool = True) -> bool:
    """
    Check if an output path is valid.

    Args:
        path (str): The path to the file.

    Keyword Args:
        overwrite (bool = True): True if the file should be overwritten, False otherwise.

    Returns:
        bool: True if the output path is valid, False otherwise.
    """
    if not is_valid_file_path(path):
        return False

    if file_exists(path):
        if overwrite:
            return True

        return False

    return True


def is_valid_output_path_list(output_list: List[str], *, overwrite: bool = True) -> bool:
    """
    Check if a list of output paths are valid.

    Args:
        output_list (list): The list of paths to the files.

    Keyword Args:
        overwrite (bool = True): True if the file should be overwritten, False otherwise.

    Returns:
        bool: True if the list of output paths are valid, False otherwise.
    """
    if not isinstance(output_list, list):
        return False

    if len(output_list) == 0:
        return False

    for path in output_list:
        if not is_valid_output_path(path, overwrite=overwrite):
            return False

    return True


def remove_if_required(path: str, overwrite: bool = True) -> bool:
    """
    Remove a file if overwrite is True.

    Args:
        path (str): The path to the file.
        overwrite (bool = True): If True, overwrite the file.

    Returns:
        bool: True if the file was removed, False otherwise.
    """
    assert is_valid_output_path(path), f"path must be a valid output path. {path}"

    if overwrite and path.startswith(f"{os.sep}vsimem"):
        gdal.Unlink(path)
        return True

    if overwrite and os.path.exists(path):
        try:
            os.remove(path)
            return True
        except:
            raise RuntimeError(f"Error while deleting file: {path}") from None

    return False


def remove_if_required_list(output_list: List[str], overwrite: bool) -> bool:
    """
    Remove a list of files if overwrite is True.

    Args:
        output_list (list): The list of paths to the files.
        overwrite (bool): If True, overwrite the files.

    Returns:
        bool: True if the files were removed, False otherwise.
    """
    assert is_valid_output_path_list(output_list), f"output_list must be a valid output path list. {output_list}"

    for path in output_list:
        remove_if_required(path, overwrite)

    return True


def get_augmented_path(
    path: str,
    *,
    prefix: str = "",
    suffix: str = "",
    add_uuid: bool = True,
    folder: Optional[str] = None,
) -> str:
    """
    Gets a basename from a string in the format:
    `dir/prefix_basename_time_uuid_suffix.ext`

    Args:
        path (str): The path to the original file.

    Keyword Args:
        prefix (str = ""): The prefix to add to the memory path.
        suffix (str = ""): The suffix to add to the memory path.
        add_uuid (bool = True): If True, add a uuid to the memory path.
        folder (str = None): The folder to save the file in.

    Returns:
        str: A string of the current UNIX time in seconds.
    """
    assert isinstance(path, str), "path must be a string."
    assert isinstance(folder, (type(None), str)), "folder must be None or a string"
    assert len(os.path.splitext(os.path.basename(path))[1]) > 1, f"Path must have an extension. {path}"

    if os.path.basename(os.path.abspath(path)) == path:
        path = PurePosixPath("/vsimem", path).as_posix()

    if folder is not None:

        if folder.startswith("/vsimem"):
            path = PurePosixPath("/vsimem", os.path.basename(path)).as_posix()
        else:
            assert folder_exists(folder), f"folder must exist. {folder}"
            path = os.path.join(folder, os.path.basename(path))

        if not path.startswith(f"{os.sep}vsimem") and not folder_exists(path_to_folder(folder)):
            raise ValueError(f"Folder: {folder} does not exist.")

    assert is_valid_file_path(path), f"path must be a valid file path. {path}"

    base = os.path.basename(path)
    split = list(os.path.splitext(base))

    uuid = ""
    if add_uuid:
        uuid = f"_{get_unix_seconds_as_str()}_{str(uuid4())}"

    if is_valid_mem_path(path):
        if split[1][1:] in get_valid_raster_driver_extensions():
            split[1] = ".tif"
        elif split[1][1:] in get_valid_vector_driver_extensions():
            split[1] = ".gpkg"
        else:
            raise ValueError("Unable to parse file extension as valid datasource.")

    basename = f"{prefix}{split[0]}{uuid}{suffix}{split[1]}"

    if is_valid_mem_path(path):
        out_path = PurePosixPath("/vsimem", basename).as_posix()
    else:
        out_path = os.path.join(os.path.dirname(os.path.abspath(path)), basename)

    return out_path


def get_size(
    start_path: str = ".",
    rough: bool = True,
) -> int:
    """
    Get the size of a folder.

    Keyword Args:
        start_path (str = "."): The path to the folder.
        rough (bool = True): If True, return a rough estimate.

    Returns:
        int: The size of the folder.
    """
    assert isinstance(start_path, str), "start_path must be a string."
    assert folder_exists(start_path), "start_path must exist."
    assert isinstance(rough, bool), "rough must be a boolean."

    total_size = 0
    for dirpath, _dirnames, filenames in os.walk(start_path):
        for file in filenames:
            file_path = os.path.join(dirpath, file)

            if not os.path.islink(file_path): # skip if symbolic link
                total_size += os.path.getsize(file_path)

    if rough is True:
        return total_size >> 20

    return total_size


def divide_into_steps(
    total: int,
    step: int,
):
    """
    Divide a number into steps.

    Args:
        total int: The total number.
        step int: The step size.

    Returns:
        list: The list of steps.
    """
    assert isinstance(total, int), "total must be an integer."
    assert isinstance(step, int), "step must be an integer."

    steps = []
    remainder = total % step
    divided = int(total / step)
    for _ in range(step):
        if remainder > 0:
            steps.append(divided + 1)
            remainder -= 1
        else:
            steps.append(divided)

    return steps


def divide_arr_into_steps(
    arr: List[Any],
    steps_length: int,
) -> List[List[Any]]:
    """
    Divide an array into steps.

    Args:
        arr (list): The array.
        steps_length (int): The length of each step.

    Returns:
        list: An array divided into steps.
    """
    assert isinstance(arr, list), "arr must be a list."
    assert isinstance(steps_length, int), "steps_length must be an integer."

    steps = divide_into_steps(len(arr), steps_length)

    ret_arr = []
    last = 0
    count = 0
    for step in steps:
        count += 1
        if count > len(arr):
            continue
        ret_arr.append(arr[last : step + last])
        last += step

    return ret_arr


def step_ranges(
    arr_with_steps: List[List[Any]],
) -> List[Dict[str, int]]:
    """
    Get the ranges of each step.

    Args:
        arr_with_steps (list): The array with steps.

    Returns:
        list: A list of dictionaries of type: { "id": int, "start": int, "stop": int}.
    """
    assert isinstance(arr_with_steps, list), "arr_with_steps must be a list."

    start_stop = []
    last = 0
    for idx, step_size in enumerate(arr_with_steps):
        fid = idx + 1

        start_stop.append(
            {
                "id": fid,
                "start": last,
                "stop": last + step_size,
            }
        )

        last += step_size

    return start_stop


def recursive_check_type_list_none_or_tuple(potential_type: Any) -> bool:
    """
    Recursively check if a type, list or tuple.

    Args:
        potential_type (any): The variable to test.

    Returns:
        bool: True if a type, list, or tuple, False otherwise.
    """
    if isinstance(potential_type, type(None)):
        return True

    if isinstance(potential_type, type):
        return True

    if isinstance(potential_type, (list, tuple)):
        for item in potential_type:
            if not recursive_check_type_list_none_or_tuple(item):
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
    Utility function to type check the inputs of a function. Check two levels down.

    Args:
        variable (any): The variable to check.
        types (tuple): The types to check against. (float, int, ...)

    Keyword Args:
        name (str = ""): The name printed in the error string if an error is thrown.
        throw_error (bool = True): If True, raise an error if the type is not correct.

    Returns:
        bool: A boolean indicating if the type is valid. If throw_error an error is raised if the input is not a valid type.
    """
    assert isinstance(name, str), "name must be a string."
    assert recursive_check_type_list_none_or_tuple(types), f"types must be a type, list, None, or tuple. not: {types}"

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


def is_list_all_val(
    arr: List[Any],
    val: Any,
) -> bool:
    """
    Check if a list is all a value. This also considers type.

    Args:
        arr (list): The list to check.
        val (any): The value to check against.

    Returns:
        bool: True if all elements are x, False otherwise.
    """
    assert isinstance(arr, list), "arr must be a list."

    for item in arr:
        if not isinstance(val, type(item)) or item != val:
            return False

    return True


def progress(
    count: int,
    total: int,
    name: str = "Processing",
) -> None:
    """
    Print a progress bar.

    Args:
        count (int): The current count.
        total (int): The total count.

    Keyword Args:
        name (str = "Processing"): The name to show in the progress bar. Default: "Processing".

    Returns:
        None.
    """
    assert isinstance(count, int), "count must be an integer."
    assert isinstance(total, int), "total must be an integer."
    assert isinstance(name, str), "name must be a string."

    sys.stdout.flush()

    try:
        bar_len = os.get_terminal_size().columns - 24
    except Exception:
        bar_len = shutil.get_terminal_size().columns - 24

    filled_len = int(round(bar_len * count / float(total)))
    display_name = name[:10] + "..: "

    progress_bar = "█" * filled_len + "." * (bar_len - filled_len)

    percents = round(100.0 * count / float(total), 1)

    if percents >= 100.0:
        percents = 100.0

    if count == total:
        sys.stdout.write(f"{display_name}[{progress_bar}] {percents} %\r")
        sys.stdout.flush()
        print("")
        return None
    else:
        sys.stdout.write(f"{display_name}[{progress_bar}] {percents} %\r")
        sys.stdout.flush()

    return None


def timing(
    before: datetime,
    print_msg: bool = True,
) -> None:
    """
    Get the time elapsed since the given time.

    Args:
        before (datetime): The time to compare.

    Keyword Args:
        print_msg (bool = True): If True, print the time elapsed.

    Returns:
        None.
    """
    assert isinstance(before, datetime), "before must be a datetime object."

    after = datetime.now()
    dif = (after - before).total_seconds()

    hours = int(dif / 3600)
    minutes = int((dif % 3600) / 60)
    seconds = f"{dif % 60:.2f}"

    message = f"Processing took: {hours}h {minutes}m {seconds}s"

    if print_msg:
        print(message)

    return message


def get_dynamic_memory_limit_bytes(
    *,
    percentage: float = 80.0,
    min_bytes: int = 1000000,
    available: bool = True,
) -> int:
    """
    Returns a dynamic memory limit taking into account total memory and CPU cores.

    Keyword Args:
        percentage (float = 80.0): The percentage of the total memory to use.
        min_bytes (int = 1000000): The minimum number of bytes to be returned.
        available (bool = True): If True, consider available memory instead of total memory.

    Returns:
        int: The dynamic memory limit in bytes.
    """
    assert isinstance(percentage, (int, str, float)), "percentage must be an integer."

    if percentage == "auto" or percentage is None:
        percentage = 80.0

    assert percentage > 0.0 and percentage <= 100.0, "percentage must be > 0 and <= 100."

    dyn_limit = min_bytes

    if available:
        dyn_limit = round(psutil.virtual_memory().available * (percentage / 100.0), 0)
    else:
        dyn_limit = round(psutil.virtual_memory().total * (percentage / 100.0), 0)

    if dyn_limit < min_bytes:
        dyn_limit = min_bytes

    return int(dyn_limit)


def is_str_a_glob(test_str: str) -> bool:
    """
    Check if a string is a glob.

    Args:
        test_str (str): The string to check.

    Returns:
        bool: True if the string is a glob, False otherwise.
    """
    if not isinstance(test_str, str):
        return False

    if len(test_str) < 6:
        return False

    if test_str[-5:] == ":glob":
        return True

    return False


def parse_glob_path(
    test_str: str,
) -> List[str]:
    """
    Parses a string containing a glob path.

    Args:
        test_str (str): The string to parse the pattern from.

    Returns:
        list: A list of the matching paths.
    """
    assert is_str_a_glob(test_str), "test_str must be a glob path."
    pre_glob = test_str[:-5]

    return glob(pre_glob)


def ensure_list(
    variable_or_list: Union[List, Any],
) -> List[Any]:
    """
    Ensure that a variable is a list.

    Args:
        variable_or_list (any): The variable to check.

    Returns:
        list: The variable as a list.
    """
    if isinstance(variable_or_list, str) and is_str_a_glob(variable_or_list):
        return parse_glob_path(variable_or_list)

    if isinstance(variable_or_list, list):
        return variable_or_list

    return [variable_or_list]


def all_arrays_are_same_size(
    list_of_arrays: List[np.ndarray],
) -> bool:
    """
    Check if all arrays in a list are the same size.

    Args:
        list_of_arrays (list): The list of numpy arrays to check.

    Returns:
        bool: True if all arrays are the same size, False otherwise.
    """
    assert isinstance(list_of_arrays, list), "list_of_arrays must be a list."

    for arr in list_of_arrays:
        assert isinstance(arr, np.ndarray), "list_of_arrays must be a list of arrays."

    if len(list_of_arrays) == 0:
        return True

    shape = list_of_arrays[0].shape

    for arr in list_of_arrays:
        if arr.shape != shape:
            return False

    return True
