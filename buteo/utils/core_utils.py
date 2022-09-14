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

# External
import psutil
from osgeo import gdal

from buteo.utils.gdal_enums import get_valid_raster_driver_extensions, get_valid_vector_driver_extensions



def get_unix_seconds_as_str():
    """
    Get a string of the current UNIX time in seconds.

    ## Returns:
    (_str_): A string of the current UNIX time in seconds.
    """
    return str(int(time.time()))


def is_float(value):
    """
    Check if a value is a float. If it is a string, try to convert it to a float.

    ## Args:
    `value` (_any_): The value to check.

    ## Returns:
    (_bool_): **True** if the value is a float, **False** otherwise.
    """
    if isinstance(value, float):
        return True
    elif isinstance(value, str):
        try:
            float(value)
            return True
        except ValueError:
            return False
    else:
        return False


def is_number(potential_number):
    """
    Check if variable is a number.

    ## Args:
    `potential_number` (_any_): The variable to check. </br>

    ## Returns:
    (_bool_): **True** if the variable is a number, **False** otherwise.
    """
    if isinstance(potential_number, float):
        return True

    if isinstance(potential_number, int):
        return True

    return False


def is_int(value):
    """
    Check if a value is an integer. If it is a string, try to convert it to an integer.

    ## Args:
    `value` (_any_): The value to check. </br>

    ## Returns:
    (_bool_): **True** if the value is an integer, **False** otherwise.
    """
    if isinstance(value, int):
        return True
    elif isinstance(value, str):
        try:
            int(value)
            return True
        except ValueError:
            return False
    else:
        return False


def is_list_all_the_same(lst):
    """
    Check if a list contains all the same elements.

    ## Args:
    `lst` (_list_): The list to check. </br>

    ## Returns:
    (_bool_): **True** if the list contains all the same elements, **False** otherwise.
    """
    assert isinstance(lst, list), "lst must be a list."

    first = lst[0]
    for idx, item in enumerate(lst):
        if idx == 0:
            continue

        if item != first:
            return False

    return True


def file_exists(path):
    """
    Check if a file exists. Also checks vsimem.

    ## Args:
    `path` (_str_): The path to the file. </br>

    ## Returns:
    (_bool_): **True** if the file exists, **False** otherwise.
    """
    assert isinstance(path, str), "path must be a string."

    if os.path.exists(path):
        return True

    try:
        if path in gdal.listdir("/vsimem"):
            return True
    except:  # pylint: disable=bare-except
        pass

    return False


def folder_exists(path):
    """
    Check if a folder exists.

    ## Args:
    `path` (_str_): The path to the folder. </br>

    ## Returns:
    (_bool_): **True** if the folder exists, **False** otherwise.
    """
    if isinstance(path, str):
        abs_dir = os.path.isdir(path)
        if abs_dir:
            return True

        abs_dir = os.path.isdir(os.path.abspath(path))
        if abs_dir:
            return True

    return False


def delete_files_in_folder(folder):
    """
    Delete all files in a folder. Does not remove subfolders.

    ## Args:
    `folder` (_str_): The path to the folder. </br>

    ## Returns:
    (_bool_): **True** if the files were deleted, throws exception otherwise..
    """
    assert isinstance(folder, str), "folder must be a string."
    assert folder_exists(folder), "folder must exist."

    for file in glob(folder + "*.*"):
        try:
            os.remove(file)
        except Exception:
            print(f"Warning. Could not remove: {file}")

    return True


def delete_folder(folder):
    """
    Delete a folder.

    ## Args:
    `folder` (_str_): The path to the folder. </br>

    ## Returns:
    (_bool_): **True** if the folder was deleted, **False** otherwise.
    """
    assert isinstance(folder, str), "folder must be a string."
    assert folder_exists(folder), "folder must exist."

    shutil.rmtree(folder)

    return True


def delete_file(file):
    """
    Delete a File

    ## Args:
    `file` (_str_): The path to the file.

    ## Returns:
    (_bool_): **True** if the file was deleted, **False** otherwise.
    """
    assert isinstance(file, str), "file must be a string."
    assert file_exists(file), "file must exist."

    os.remove(file)

    if file_exists(file):
        return False

    return True


def to_number(value):
    """
    Convert a value to a number.

    ## Args:
    `value` (_any_): The value to convert.

    ## Returns:
    (_float_): The value converted to a number.
    """
    assert isinstance(value, (str, int, float)), "value must be a string, integer or float."
    if is_number(value):
        return value

    if is_float(value):
        return float(value)

    raise Exception(f"Could not convert {value} to a number.")


def make_dir_if_not_exists(path):
    """
    Make a directory if it doesn't exist.

    ## Args:
    `path` (_str_): The path to the directory. </br>

    ## Returns:
    (_str_): The path to the created directory.
    """
    assert isinstance(path, str), "path must be a string."

    if not folder_exists(path):
        os.makedirs(path)

    return path


def path_to_ext(path, with_dot=False):
    """
    Get the extension of a file.

    ## Args:
    `path` (_str_): The path to the file. </br>

    ## Kwargs:
    `with_dot` (_bool_): If True, return the extension with a dot. (**Default**: `False`) </br>

    ## Returns:
    (_str_): The extension of the file. (_without the dot_)
    """
    assert isinstance(path, str), "path must be a string."
    assert isinstance(with_dot, bool), "with_dot must be a boolean."

    basename = os.path.basename(path)
    basesplit = os.path.splitext(basename)
    ext = basesplit[1]

    if ext == "" or len(ext) == 1:
        raise Exception(f"File: {path} has no extension.")

    if with_dot:
        return ext

    return ext[1:]


def path_to_folder(path):
    """
    Get the folder of a file.

    ## Args:
    `path` (_str_): The path to the file. </br>

    ## Returns:
    (_str_): The folder of the file.
    """
    assert isinstance(path, str), "path must be a string."

    return os.path.dirname(os.path.abspath(path))


def change_path_ext(path, target_ext):
    """
    Change the extension of a file.

    ## Args:
    `path` (_str_): The path to the file. </br>
    `target_ext` (_str_): The new extension. </br>

    ## Returns:
    (_str_): The path to the file with the new extension.
    """
    assert isinstance(path, str), "path must be a string."
    assert isinstance(target_ext, str), "target_ext must be a string."

    basename = os.path.basename(path)
    basesplit = os.path.splitext(basename)
    ext = basesplit[1]

    if ext == "" or len(ext) == 1:
        raise Exception(f"File: {path} has no extension.")

    return os.path.join(os.path.dirname(path), f"{basesplit[0]}.{target_ext}")



def is_valid_mem_path(path):
    """
    Check if a path is a valid memory path that has an extension.

    ## Args:
    `path` (_str_): The path to test. </br>

    ## Returns:
    (_bool_): **True** if path is a valid memory path, **False** otherwise.
    """
    if not isinstance(path, str):
        return False

    if len(path) < len("/vsimem/x"):
        return False

    ext = os.path.splitext(path)[1]

    if ext == "" or ext == ".":
        return False

    if path.startswith("/vsimem"):
        return True

    return False


def is_valid_non_memory_path(path):
    """
    Check if a path is valid, not in memory, and has an extension.

    ## Args:
    `path` (_str_): The path to the file. </br>

    ## Returns:
    (_bool_): **True** if the path has an extension, **False** otherwise.
    """
    if not isinstance(path, str):
        return False

    ext = os.path.splitext(path)[1]

    if ext == "" or ext == ".":
        return False

    if not folder_exists(path_to_folder(path)):
        return False

    if is_valid_mem_path(path):
        return False

    return True


def is_valid_file_path(path):
    """
    Check if a path is valid and has an extension. Path can be in memory.

    ## Args:
    `path` (_str_): The path to the file. </br>

    ## Returns:
    (_bool_): **True** if the path has an extension, **False** otherwise.
    """
    if is_valid_mem_path(path) or is_valid_non_memory_path(path):
        return True

    return False


def is_valid_output_path(path, *, overwrite=True):
    """
    Check if an output path is valid.

    ## Args:
    `path` (_str_): The path to the file. </br>

    ## Kwargs:
    `overwrite` (_bool_): **True** if the file should be overwritten, **False** otherwise. </br>

    ## Returns:
    (_bool_): **True** if the output path is valid, **False** otherwise.
    """
    if not is_valid_file_path(path):
        return False

    if file_exists(path):
        if overwrite:
            return True

        return False

    return True


def is_valid_output_path_list(output_list, *, overwrite=True):
    """
    Check if a list of output paths are valid.

    ## Args:
    `output_list` (_list_): The list of paths to the files. </br>

    ## Kwargs:
    `overwrite` (_bool_): **True** if the file should be overwritten, **False** otherwise. </br>

    ## Returns:
    (_bool_): **True** if the list of output paths are valid, **False** otherwise.
    """
    if not isinstance(output_list, list):
        return False

    if len(output_list) == 0:
        return False

    for path in output_list:
        if not is_valid_output_path(path, overwrite=overwrite):
            return False

    return True


def remove_if_required(path, overwrite):
    """
    Remove a file if overwrite is True.

    ## Args:
    `path` (_str_): The path to the file. </br>
    `remove` (_bool_): If True, remove the file. </br>

    ## Returns:
    (_bool_): **True** if the file was removed, **False** otherwise.
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


def remove_if_required_list(output_list, overwrite):
    """
    Remove a list of files if overwrite is True.

    ## Args:
    `output_list` (_list_): The list of paths to the files. </br>
    `remove` (_bool_): If True, remove the files. </br>

    ## Returns:
    (_bool_): **True** if the files were removed, **False** otherwise.
    """
    assert is_valid_output_path_list(output_list), f"output_list must be a valid output path list. {output_list}"

    for path in output_list:
        remove_if_required(path, overwrite)

    return True


def get_augmented_path(path, *, prefix="", suffix="", add_uuid=True, folder=None):
    """
    Gets a basename from a string in the format: </br>
    `dir/prefix_basename_time_uuid_suffix.ext`

    ## Args:
    `path` (_str_): The path to the original file. </br>

    ## Kwargs:
    `prefix` (_str_): The prefix to add to the memory path. (**Default**: `""`) </br>
    `suffix` (_str_): The suffix to add to the memory path. (**Default**: `""`) </br>
    `add_uuid` (_bool_): If True, add a uuid to the memory path. (**Default**: `True`) </br>

    ## Returns:
    (_str_): A string of the current UNIX time in seconds.
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


def get_size(start_path=".", rough=True):
    """
    Get the size of a folder.

    ## Kwargs:
    `start_path` (_str_): The path to the folder. (**Default**: `"."`) </br>
    `rough` (_bool_): If True, return a rough estimate. (**Default**: `True`) </br>

    ## Returns:
    (_int_): The size of the folder.
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


def divide_into_steps(total, step):
    """
    Divide a number into steps.

    ## Args:
    `total` (_int_): The total number. </br>
    `step` (_int_): The step size. </br>

    ## Returns:
    (_list_): The list of steps.
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


def divide_arr_into_steps(arr, steps_length):
    """
    Divide an array into steps.

    ## Args:
    `arr` (_list_): The array. </br>
    `steps_length` (_int_): The length of each step. </br>

    ## Returns:
    (_list_): An array divided into steps.
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


def step_ranges(arr_with_steps):
    """
    Get the ranges of each step.

    ## Args:
    `arr_with_steps` (_list_): The array with steps. </br>

    ## Returns:
    (_dict_): A dictionary of type: `{ "id": _int_, "start": _int_, "stop": _int_}`.
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


def recursive_check_type_list_none_or_tuple(potential_type):
    """
    Recursively check if a type, list or tuple.

    ## Args:
    `potential_type` (_any_): The variable to test. </br>

    ## Returns:
    (_bool_): **True** if a type, list or tuple, **False** otherwise.
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
    variable,
    types,
    name="",
    *,
    throw_error=True,
):
    """
    Utility function to type check the inputs of a function.

    ## Args:
    `variable` (_any_): The variable to check. </br>
    `types` (_tuple_): The types to check against. `(float, int, ...)` </br>

    ## Kargs:
    `name` (_str_): The name printed in the error string if an error is thrown. (**Default**: `""`)</br>
    `throw_error` (_bool_): If True, raise an error if the type is not correct. (**Default**: `True`)</br>

    ## Returns:
    (_bool_): A boolean indicating if the type is valid. If throw_error an error is raised if the input is not a valid type.
    """
    assert isinstance(types, list), "types must be a list."
    assert isinstance(name, str), "name must be a string."
    assert recursive_check_type_list_none_or_tuple(types), f"types must be a type, list, None, or tuple. not: {types}"

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

    if isinstance(variable, (list, tuple)):
        found = []
        for valid_type in valid_types:
            if not isinstance(valid_type, (list, tuple)):
                continue

            for item in variable:
                if type_check(item, valid_type, name, throw_error=throw_error):
                    found.append(True)

        if len(found) == len(variable):
            return True
    else:
        for valid_type in valid_types:
            if isinstance(valid_type, (list, tuple)):
                continue

            if isinstance(variable, valid_type):
                return True

    if throw_error:
        raise ValueError(
            f"The type of variable {name} is not valid. Expected: {types}, got: {type(variable)}"
        )

    return False


def is_list_all_val(arr, val):
    """
    Check if a list is all a value. This also considers type.

    ## Args:
    `arr` (_list_): The list to check. </br>
    `val` (_any_): The value to check against. </br>

    ## Returns:
    (_bool_): **True** if all elements are x, **False** otherwise.
    """
    assert isinstance(arr, list), "arr must be a list."

    for item in arr:
        if not isinstance(val, type(item)) or item != val:
            return False

    return True


def progress(count, total, name="Processing"):
    """
    Print a progress bar.

    ## Args:
    `count` (_int_): The current count. </br>
    `total` (_int_): The total count. </br>

    ## Kwargs:
    `name` (_str_): The name to show in the progress bar. (**Default**: `"Processing"`) </br>

    ## Returns:
    (_None_): Returns None.
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


def timing(before, print_msg=True):
    """
    Get the time elapsed since the given time.

    ## Args:
    `before` (_datetime_): The time to compare. </br>

    ## Kwargs:
    `print_msg` (_bool_): If True, print the time elapsed. (**Default**: `True`) </br>

    ```python
    >>> before = datetime.now()
    >>> long_running_calculation()
    >>> timing(before)
    >>> Processing took: 1h 1m 1s
    ```
    """
    assert isinstance(before, datetime), "before must be a datetime object."

    after = time.time()
    dif = after - before

    hours = int(dif / 3600)
    minutes = int((dif % 3600) / 60)
    seconds = f"{dif % 60:.2f}"

    message = f"Processing took: {hours}h {minutes}m {seconds}s"

    if print_msg:
        print(message)

    return message


def get_dynamic_memory_limit_bytes(*, percentage=80.0, min_bytes=1000000, available=True):
    """
    Returns a dynamic memory limit taking into account total memory and CPU cores.

    ## Args:
    `percentage` (_int_): The percentage of the total memory to use. (Default: **80**)

    ## Returns:
    (_int_): The dynamic memory limit in bytes.
    """
    assert isinstance(percentage, int), "percentage must be an integer."
    assert percentage > 0.0 and percentage <= 100.0, "percentage must be > 0 and <= 100."

    dyn_limit = min_bytes

    if available:
        dyn_limit = round(psutil.virtual_memory().available * (percentage / 100.0), 0)
    else:
        dyn_limit = round(psutil.virtual_memory().total * (percentage / 100.0), 0)

    if dyn_limit < min_bytes:
        dyn_limit = min_bytes

    return int(dyn_limit)


def is_str_a_glob(test_str):
    """
    Check if a string is a glob.

    ## Args:
    `test_str` (_str_): The string to check.

    ## Returns:
    (_bool_): **True** if the string is a glob, **False** otherwise.
    """
    if not isinstance(test_str, str):
        return False

    if len(test_str) < 6:
        return False

    if test_str[-5:] == ":glob":
        return True

    return False


def parse_glob_path(test_str):
    """
    Parses a string containing a glob path.

    ## Args:
    `test_str` (_str_): The string to parse the pattern from.

    ## Returns:
    (_list_): A list of the matching paths.
    """
    assert is_str_a_glob(test_str), "test_str must be a glob path."
    pre_glob = test_str[:-5]

    return glob(pre_glob)


def ensure_list(variable_or_list):
    """
    Ensure that a variable is a list.

    ## Args:
    `variable_or_list` (_any_): The variable to check. </br>

    ## Returns:
    (_list_): The variable as a list.
    """
    if isinstance(variable_or_list, str) and is_str_a_glob(variable_or_list):
        return parse_glob_path(variable_or_list)

    if isinstance(variable_or_list, list):
        return variable_or_list

    return [variable_or_list]
