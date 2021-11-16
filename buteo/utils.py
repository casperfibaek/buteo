import os
import sys
import time
import psutil
import shutil
import linecache
import tracemalloc

from typing import Any
from glob import glob


def make_dir_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
    
    return path


def display_top(snapshot, key_type="lineno", limit=3):
    # usage:

    # Memory profiling
    # from collections import Counter
    # import tracemalloc
    # from buteo.utils import display_top

    # tracemalloc.start()
    # counts = Counter()

    # snapshot = tracemalloc.take_snapshot()
    # display_top(snapshot)

    # from https://stackoverflow.com/questions/552744/how-do-i-profile-memory-usage-in-python
    snapshot = snapshot.filter_traces(
        (
            tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
            tracemalloc.Filter(False, "<unknown>"),
        )
    )
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print(
            "#%s: %s:%s: %.1f KiB" % (index, filename, frame.lineno, stat.size / 1024)
        )
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print("    %s" % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))


def progress(count, total, name="Processing"):
    sys.stdout.flush()

    try:
        bar_len = os.get_terminal_size().columns - 24
    except:
        bar_len = shutil.get_terminal_size().columns - 24

    filled_len = int(round(bar_len * count / float(total)))
    display_name = name[:10] + "..: "

    bar = "█" * filled_len + "." * (bar_len - filled_len)

    percents = round(100.0 * count / float(total), 1)

    if percents >= 100.0:
        percents = 100.0

    if count == total:
        sys.stdout.write(f"{display_name}[{bar}] {percents} %\r")
        sys.stdout.flush()
        print("")
        return None
    else:
        sys.stdout.write(f"{display_name}[{bar}] {percents} %\r")
        sys.stdout.flush()

    return None


def timing(before, print_msg=True):
    after = time.time()
    dif = after - before
    hours = int(dif / 3600)
    minutes = int((dif % 3600) / 60)
    seconds = "{0:.2f}".format(dif % 60)
    message = f"Processing took: {hours}h {minutes}m {seconds}s"
    if print_msg:
        print(message)

    return message


def path_to_ext(path):
    basename = os.path.basename(path)
    basesplit = os.path.splitext(basename)
    ext = basesplit[1]

    return ext


def path_to_name(path, with_ext=False):
    basename = os.path.basename(path)
    basesplit = os.path.splitext(basename)
    name = basesplit[0]

    if with_ext:
        return basename

    return name


def file_exists(path):
    return os.path.exists(path)


def folder_exists(path):
    return os.path.isdir(path)


def is_number(potential_number):
    if isinstance(potential_number, float):
        return True

    if isinstance(potential_number, int):
        return True

    return False


def overwrite_required(path, overwrite):
    if path is not None:
        exists = file_exists(path)
        if exists and not overwrite:
            raise Exception(f"File: {path} already exists and overwrite is False.")


def remove_if_overwrite(path, overwrite):
    if path is not None:
        exists = file_exists(path)
        if exists and overwrite:
            os.remove(path)
        elif exists:
            raise Exception(f"File: {path} already exists and overwrite is False.")


def get_size(start_path=".", rough=True):
    total_size = 0
    for dirpath, _dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    if rough is True:
        return total_size >> 20
    else:
        return total_size


def divide_steps(total, step):
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


def divide_into_steps(arr, steps_length):
    steps = divide_steps(len(arr), steps_length)

    ret_arr = []
    last = 0
    count = 0
    for x in steps:
        count += 1
        if count > len(arr):
            continue
        ret_arr.append(arr[last : x + last])
        last += x

    return ret_arr


def step_ranges(steps):
    start_stop = []
    last = 0
    for num in range(0, len(steps)):
        step_size = steps[num]
        id = num + 1

        start_stop.append(
            {
                "id": id,
                "start": last,
                "stop": last + step_size,
            }
        )

        last += step_size

    return start_stop


def file_in_use(path):
    for process in psutil.process_iter():
        try:
            for item in process.open_files():
                if path == item.path:
                    return True
        except Exception:
            pass

    return False


def type_check(
    variable: Any,
    types: list,
    name: str = "",
    allow_none=False,
    throw_error=True,
) -> bool:
    """
        Utility function to type check the inputs of a function.

    Args:
        variable (any): The variable to typecheck

        types (tuple): A tuple of type classes. e.g. int, float, etc.

    **kwargs:
        name (str): The name printed in the error string if an error is thrown.

        allow_none (bool): Allow the type to be None

        throw_error (bool): Should the function throw an error if the type
        is wrong or return a boolean.

    Returns:
        A boolean indicating if the type is valid. If throw_error an error is
        raised if the input is not a valid type.
    """

    if variable is None:
        if allow_none:
            return True

        if throw_error:
            raise ValueError(
                f"Variable: {name} is type None when type None is not allowed."
            )

        return False

    is_valid_type = isinstance(variable, tuple(types))

    if is_valid_type:
        return True

    if allow_none and variable is None:
        return True

    type_names = []
    valid_types = list(types)
    if allow_none:
        valid_types.append(None)

    for t in valid_types:
        type_names.append(t.__name__)

    if throw_error:
        raise ValueError(
            f"Variable: {name} must be type(s): {' '.join(type_names)} - Received type: {type(variable).__name__}, variable: {variable}"
        )

    return False


def delete_files_in_folder(folder):
    for f in glob(folder + "*.*"):
        try:
            os.remove(f)
        except:
            pass
