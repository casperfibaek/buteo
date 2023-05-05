
"""
### Generic utility functions ###

Functions that make interacting with the toolbox easier.
"""

# Standard Library
import os
import gc
import sys
import shutil
from datetime import datetime
from typing import Optional

# External
import psutil
import numpy as np


def _print_progress(
    count: int,
    total: int,
    name: str = "Processing",
) -> None:
    """
    Print a progress bar.

    `progress(10, 100, "Processing..")`

    Parameters
    ----------
    count : int
        The current count.

    total : int
        The total count.

    name : str, optional.
        The name of the process. Default: "Processing".

    Returns
    -------
    None
    """
    assert isinstance(count, int), "count must be an integer."
    assert isinstance(total, int), "total must be an integer."
    assert isinstance(name, str), "name must be a string."

    sys.stdout.flush()

    try:
        bar_len = os.get_terminal_size().columns - 24
    except RuntimeError:
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


def _get_timing(
    before: datetime,
    print_msg: bool = True,
) -> str:
    """
    Get the time elapsed since the given time.

    Parameters
    ----------
    before : datetime
        The time to compare to.
    
    print_msg : bool, optional.
        If True, print the message. Default: True.

    Returns
    -------
    str
        The message.

    Examples
    --------
    ```python
    >>> from datetime import datetime
    >>> from buteo.utils import timing

    >>> before = datetime.now()
    >>> # Do something
    >>> timing(before)
    "Processing took: 0h 2m 3.1s"
    ```
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


def _get_folder_size(
    start_path: str = ".",
    rough: bool = True,
) -> int:
    """
    Get the size of a folder.

    Parameters
    ----------
    start_path : str, optional.
        The path to the folder. Default: ".".
    
    rough : bool, optional.
        If True, return the size in MB. Default: True.

    Returns
    -------
    int
        The size of the folder in bytes or MB.
    """
    assert isinstance(start_path, str), "start_path must be a string."
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


def _get_dynamic_memory_limit(
    proportion: float = 0.8,
    *,
    min_mb: int = 100,
    available: bool = False,
) -> int:
    """
    Returns a dynamic memory limit taking into account total memory and CPU cores.
    The return is in mbytes. For GDAL.

    The value is interpreted as being in megabytes if the value is less than 10000. For values >=10000, this is interpreted as bytes.

    Parameters
    ----------
    percentage : float, optional.
        The percentage of the total memory to use. Default: 0.8.
    
    min_mb : int, optional.
        The minimum number of megabytes to be returned. Default: 100.
    
    available : bool, optional.
        If True, consider available memory instead of total memory. Default: False.

    Returns
    -------
    int
        The dynamic memory limit in bytes.
    """
    assert isinstance(proportion, (int, float)), "percentage must be an integer."
    assert isinstance(min_mb, int), "min_mb must be an integer."
    assert isinstance(available, bool), "available must be a boolean."
    assert min_mb > 0, "min_mb must be > 0."
    assert proportion > 0.0 and proportion <= 1.0, "percentage must be > 0 and <= 1."

    if available:
        dyn_limit = np.rint(
            (psutil.virtual_memory().available * proportion)  / (1024 ** 2),
        )
    else:
        dyn_limit = np.rint(
            (psutil.virtual_memory().total * proportion)  / (1024 ** 2),
        )

    if dyn_limit < min_mb:
        dyn_limit = min_mb

    # GDALWarpMemoryLimit() expects the value in bytes if it is >= 10000
    if dyn_limit > 10000:
        dyn_limit = dyn_limit * (1024 ** 2)

    return int(dyn_limit)


def _force_garbage_collect_all(
    delete_functions: bool = True,
) -> None:
    """
    Clears the memory by deleting all objects in the main namespace.
    Very aggresive. Use with caution.
    
    Parameters
    ----------
    delete_functions : bool, optional.
        If True, delete functions as well. Default: True.

    Returns
    -------
    None
    """
    # Get a list of all objects
    all_objects = sys.modules['__main__'].__dict__.copy()

    # Iterate over the objects and delete them if possible
    for key, value in all_objects.items():
        if key in ["sys", "os", "gc", "buteo"]:
            continue
        if key.startswith("__"):
            continue  # Skip built-in objects
        if not delete_functions and callable(value):
            continue

        del sys.modules['__main__'].__dict__[key]  # Remove the object from the namespace

    # Collect garbage
    gc.collect()
