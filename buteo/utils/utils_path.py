"""
### Generic utility functions ###

Functions that make interacting with the toolbox easier.
"""

# Standard Library
import sys; sys.path.append("../../")
import os
import shutil
import fnmatch
from glob import glob
from uuid import uuid4
from typing import List, Optional, Union
from warnings import warn

# External
from osgeo import gdal, ogr

# Internal
from buteo.utils import utils_base, utils_gdal



def _get_vsimem_content(
    folder_name: Optional[str] = None,
) -> List[str]:
    """ 
    Get the content of the vsimem folder.
    vsimem is the virtual memory folder of GDAL.
    By default, the content of the root folder is returned.

    Parameters
    ----------
    folder_name: str, optional
        The name of the folder to get the content of. Default: None (root folder)

    Returns
    -------
    List[str]
        A list of the content of the vsimem folder.
    """
    if folder_name is None:
        folder_name = "/vsimem/"

    if not isinstance(folder_name, str):
        raise TypeError("folder_name must be a string.")

    if not folder_name.endswith("/"):
        folder_name += "/"

    try:
        if hasattr(gdal, "listdir"):
            vsimem = gdal.listdir(folder_name)
            vsimem = [folder_name + v.name for v in vsimem]
        elif hasattr(gdal, "ReadDir"):
            vsimem = [folder_name + ds for ds in gdal.ReadDirRecursive(folder_name)]
        else:
            warn("WARNING: Unable to access vsimem. Is GDAL installed?")
            return False

    except RuntimeError as e:
        warn(f"WARNING: Failed to access vsimem. Is GDAL installed? Error: {e}")

    paths = _get_unix_path(vsimem)

    return paths


def _glob_vsimem(
    pattern: str,
) -> List[str]:
    """
    Find files in vsimem using glob.

    Example:
    `_glob_vsimem("*/patches/*.tif")` will return all tif files in the patches folder of all subfolders.

    Parameters
    ----------
    pattern: str
        The pattern to match.

    Returns
    -------
    List[str]
        A list of the files matching the pattern.
    """
    assert isinstance(pattern, str), "pattern must be a string."

    virtual_fs = _get_vsimem_content()
    matches = [path for path in virtual_fs if fnmatch.fnmatch(path, pattern)]

    return matches


def _get_unix_path(path: Union[str, List[str]]) -> Union[str, List[str]]:
    """
    Convert a path or list of paths to a unix path(s).

    Parameters
    ----------
    path: str
        The path to convert.

    Returns
    -------
    Union[str, List[str]]
        The converted path[s].
    """
    input_is_list = False
    if isinstance(path, list):
        input_is_list = True

    path = [path] if isinstance(path, str) else path
    for p in path:
        assert isinstance(p, str), "path must be a string."
        assert len(p) > 0, "path must not be empty."

    out_path = []
    for p in path:
        out_path.append(
            "/".join(os.path.normpath(p).split("\\"))
        )

    if input_is_list:
        return out_path

    return out_path[0]


def _check_file_exists(path: str) -> bool:
    """
    Check if a file exists. Also checks vsimem.
    Handles both absolute and relative paths and with or without trailing slash.

    Parameters
    ----------
    path: str
        The path to the file.

    Returns
    -------
    bool
        True if the file exists, False otherwise.
    """
    if not isinstance(path, str):
        return False

    if os.path.isfile(path):
        return True

    if os.path.isfile(os.path.abspath(path)):
        return True

    vsimem = _get_vsimem_content()
    if _get_unix_path(path) in vsimem:
        return True

    return False


def _check_file_exists_vsimem(path: str) -> bool:
    """
    Check if a file exists in vsimem.

    Parameters
    ----------
    path: str
        The path to the file.

    Returns
    -------
    bool
        True if the file exists, False otherwise.
    """
    if not isinstance(path, str):
        return False

    vsimem = _get_vsimem_content()
    if _get_unix_path(path) in vsimem:
        return True

    return False


def _check_dir_exists(
    path: str,
) -> bool:
    """
    Check if a folder exists. Also checks vsimem.
    Handles both absolute and relative paths and with or without trailing slash.

    Parameters
    ----------
    path: str
        The path to the folder.

    Returns
    -------
    bool
        True if the folder exists, False otherwise.
    """
    if not isinstance(path, str):
        return False

    if os.path.isdir(os.path.normpath(path)):
        return True

    if os.path.isdir(os.path.abspath(path)):
        return True

    split = os.path.normpath(os.path.dirname(path)).split("\\")

    if "vsimem" in split:
        return True

    return False


def _check_dir_exists_vsimem(path: str) -> bool:
    """
    Check if a folder exists in vsimem.

    Parameters
    ----------
    path: str
        The path to the folder.

    Returns
    -------
    bool
        True if the folder exists, False otherwise.
    """
    if not isinstance(path, str):
        return False

    vsimem = _get_vsimem_content()
    if os.path.normpath(path) in vsimem:
        return True

    return False


def _delete_dir_content(
    folder: str,
    delete_subfolders: bool = True,
) -> bool:
    """
    Delete all files and folders in a folder.
    If only the files are to be deleted, set delete_subfolders to False.

    Parameters
    ----------
    folder: str
        The path to the folder.

    delete_subfolders: bool, optional. 
        If True, delete subfolders as well. Default: True

    Returns
    -------
    bool
        True if the folder was deleted, False otherwise.
    """
    assert isinstance(folder, str), "folder must be a string."
    assert len(folder) > 0, "folder must not be a non-empty string."
    assert _check_dir_exists(folder), "folder must exist."

    # Folder is on disk
    if not _check_dir_exists_vsimem(folder):
        for f in os.listdir(folder):
            try:
                path = os.path.join(folder, f)
                if os.path.isfile(path):
                    os.remove(path)

                elif delete_subfolders and os.path.isdir(path):
                    shutil.rmtree(path)

            except RuntimeError:
                warn(f"Warning. Could not remove: {path}", UserWarning)

                return False

    # Folder is in vsimem
    else:
        # Delete the files and folder in VSIMEM
        vsimem = _get_vsimem_content(folder)
        for f in vsimem:
            gdal.Unlink(f)

    if _check_dir_exists(folder):
        warn(f"Warning. Failed to remove: {folder}", UserWarning)
        return False

    return True


def _delete_dir(folder: str) -> bool:
    """
    Delete a folder and all its content. Also deletes vsimem folders.

    Args:
        folder (str): The path to the folder.

    Returns:
        bool: True if the folder was deleted, False otherwise.
    """
    assert isinstance(folder, str), "folder must be a string."
    assert len(folder) > 0, "folder must be a non-empty string."
    assert _check_dir_exists(folder), "folder must exist."

    # Folder is on disk
    if not _check_dir_exists_vsimem(folder):
        try:
            shutil.rmtree(folder)

        except RuntimeError:
            warn(f"Warning. Could not remove: {folder}", UserWarning)

            return False

        return True

    # Delete the files and folder in VSIMEM
    vsimem = _get_vsimem_content(folder)
    for f in vsimem:
        gdal.Unlink(f)

    gdal.Unlink(folder)

    if _check_dir_exists(folder):
        warn(f"Warning. Failed to remove: {folder}", UserWarning)
        return False

    return True


def _delete_file(file: str) -> bool:
    """
    Delete a File

    Args:
        file (str): The path to the file.

    Returns:
        bool: True if the file was deleted, False otherwise.
    """
    assert isinstance(file, str), "file must be a string."
    assert len(file) > 0, "file must be a non-empty string."
    assert _check_file_exists(file), "file must exist."

    # File is on disk
    if not _check_file_exists_vsimem(file):
        try:
            os.remove(file)

        except RuntimeError:
            warn(f"Warning. Could not remove: {file}", UserWarning)

            return False

        return True

    # Delete the file in VSIMEM
    gdal.Unlink(file)

    if _check_file_exists_vsimem(file):
        warn(f"Warning. Failed to remove: {file}", UserWarning)
        return False

    return True


def _create_dir_if_not_exists(path: str) -> str:
    """
    Make a directory if it does not exist.
    This does not work for creating directories in GDAL vsimem.
    On vsimem, it is not possible to create a folder without a file in it.

    If the folder already exists in vsimem, the path to the folder is returned.

    Parameters
    ----------
    path: str
        The path to the folder.

    Returns
    -------
    str
        The path to the folder.
    """
    assert isinstance(path, str), "path must be a string."
    assert len(path) > 0, "path must not be empty."

    if not _check_dir_exists(path):

        # Folder is on disk
        if not _check_dir_exists_vsimem(path):
            os.makedirs(path)

        # Folder is in vsimem
        return path

    if not _check_dir_exists(path):
        raise RuntimeError(f"Could not create folder: {path}.")

    return path


def _get_dir_from_path(path: str) -> str:
    """
    Get the directory of a file. Also works for folders and vsimem.

    Parameters
    ----------
    path: str
        The path to the file.

    Returns
    -------
    str
        The directory of the file.
    """
    assert isinstance(path, str), "path must be a string."
    assert len(path) > 0, "path must not be empty."

    if "/vsimem" in path:
        dirname = "/vsimem/"
    elif "vsizip" in path:
        dirname = "/vsizip/"
    else:
        dirname = _get_unix_path(os.path.dirname(os.path.abspath(path))) + "/"

    return dirname


def _get_filename_from_path(
    path: str,
    with_ext: bool = True,
) -> str:
    """
    Get the filename of a file. Also works for vsimem.

    Parameters
    ----------
    path: str
        The path to the file.

    with_ext: bool
        If True, the extension is included in the filename.

    Returns
    -------
    str
        The filename of the file.
    """
    assert isinstance(path, str), "path must be a string."
    assert len(path) > 0, "path must not be empty."

    basename = os.path.basename(path)

    if with_ext:
        return basename

    basesplit = os.path.splitext(basename)
    return basesplit[0]


def _get_ext_from_path(path: str) -> str:
    """
    Get the extension of a file. If the file has no extension, raise an error.
    The extension is returned without a dot.

    Parameters
    ----------
    path: str
        The path to the file.

    Returns
    -------
    str
        The extension of the file.
    """
    assert isinstance(path, str), "path must be a string."
    assert len(path) > 0, "path must not be empty."

    basename = os.path.basename(path)
    basesplit = os.path.splitext(basename)
    ext = basesplit[1]

    if ext == "" or len(ext) == 1:
        raise RuntimeError (f"File: {path} has no extension.")

    return ext[1:]


def _get_changed_path_ext(
    path: str,
    target_ext: str,
) -> str:
    """
    Update the extension of a file.

    Parameters
    ----------
    path: str
        The path to the file.

    target_ext: str
        The new extension (with or without dot.)
    
    Returns
    -------
    str
        The path to the file with the new extension.
    """
    assert isinstance(path, str), "path must be a string."
    assert isinstance(target_ext, str), "target_ext must be a string."
    assert len(path) > 0, "path must not be empty."
    assert len(target_ext) > 0, "target_ext must not be empty."

    target_ext = target_ext.lstrip('.')
    basename = os.path.basename(path)
    basesplit = os.path.splitext(basename)
    ext = basesplit[1].lstrip('.')

    if ext == "":
        raise RuntimeError(f"File: {path} has no extension.")

    if target_ext == "":
        return os.path.join(os.path.dirname(path), basesplit[0])

    return os.path.join(os.path.dirname(path), f"{basesplit[0]}.{target_ext}")



def _check_is_valid_mem_filepath(path: str) -> bool:
    """
    Check if a path is a valid memory path that has an extension. vsizip also works.

    Parameters
    ----------
    path: str
        The path to test.

    Returns
    -------
    bool
        True if path is a valid memory path, False otherwise.
    """
    if not isinstance(path, str):
        return False

    if len(path) == 0:
        return False

    path_chunks = os.path.normpath(path).split(os.path.sep)

    if "vsimem" not in path_chunks and "vsizip" not in path_chunks:
        return False

    if _get_ext_from_path(path) == "":
        return False

    return True


def _check_is_valid_non_mem_filepath(path: str) -> bool:
    """
    Check if a path is a valid path that has an extension.
    Does not check if the file exists and does not check if the path is in memory.

    Parameters
    ----------
    path: str
        The path to test.

    Returns
    -------
    bool
        True if path is a valid path, False otherwise.
    """
    if not isinstance(path, str):
        return False

    if len(path) == 0:
        return False

    if _get_ext_from_path(path) == "":
        return False

    folder = _get_dir_from_path(path)
    if not _check_dir_exists(folder):
        return False

    return True


def _check_is_valid_filepath(path: str) -> bool:
    """
    Check if a path is a valid path that has an extension. Path can be in memory.

    Parameters
    ----------
    path: str
        The path to test.

    Returns
    -------
    bool
        True if path is a valid path, False otherwise.
    """
    if _check_is_valid_mem_filepath(path) or _check_is_valid_non_mem_filepath(path):
        return True

    return False


def _check_is_valid_filepath_list(path_list: List[str]) -> bool:
    """
    Check if a list of paths is a valid list of paths that have an extension. Paths can be in memory.

    Parameters
    ----------
    path_list: List[str]
        The list of paths to test.

    Returns
    -------
    bool
        True if path_list is a valid list of paths, False otherwise.
    """
    if not isinstance(path_list, list):
        return False

    if len(path_list) == 0:
        return False

    for path in path_list:
        if not _check_is_valid_filepath(path):
            return False

    return True


def _check_is_valid_output_filepath(
    path: str,
    overwrite: bool = True,
):
    """
    Check if a path is a valid output path that has an extension. Path can be in memory.
    If the file already exists, and overwrite is false, return False.

    Parameters
    ----------
    path: str
        The path to test.
    
    overwrite: bool
        True if the file could be overwritten, False otherwise.

    Returns
    -------
    bool
        True if path is a valid path, False otherwise.
    """
    if not _check_is_valid_filepath(path):
        return False

    if not overwrite:
        if _check_file_exists(path):
            return False

    return True


def _check_is_valid_output_path_list(
    output_list: List[str],
    overwrite: bool = True,
) -> bool:
    """
    Check if a list of output paths are valid.

    Parameters
    ----------
    output_list: list[str]
        The list of paths to the files.

    overwrite: bool
        True if the file should be overwritten, False otherwise.

    Returns
    -------
    bool
        True if the list of output paths are valid, False otherwise.
    """
    if not isinstance(output_list, list):
        return False

    if len(output_list) == 0:
        return False

    for path in output_list:

        if not _check_is_valid_output_filepath(path, overwrite=overwrite):
            return False

    return True


def _delete_if_required(
    path: str,
    overwrite: bool = True,
) -> bool:
    """
    Delete a file if overwrite is True.

    Parameters
    ----------
    path: str
        The path to the file.

    overwrite: bool
        If True, overwrite the file.

    Returns
    -------
    bool
        True if successful, raises error otherwise.
    """
    assert isinstance(path, str), "path must be a string."
    assert len(path) > 0, "path must not be non-empty string."
    assert isinstance(overwrite, bool), "overwrite must be a bool."

    if not overwrite:
        return True

    if not _check_file_exists(path):
        return True

    if _check_is_valid_mem_filepath(path):
        gdal.Unlink(path)
    else:
        try:
            os.remove(path)
        except RuntimeError as e:
            raise RuntimeError(f"Error while deleting file: {path}, {e}") from None

    if _check_file_exists(path):
        raise RuntimeError(f"Error while deleting file: {path}")

    return True


def _delete_if_required_list(
    output_list: List[str],
    overwrite: bool = True,
) -> bool:
    """
    Delete a list of files if overwrite is True.

    Parameters
    ----------
    output_list: list[str]
        The list of paths to the files.

    overwrite: bool
        If True, overwrite the files.

    Returns
    -------
    bool
        True if the files were deleted, False otherwise.
    """
    if not isinstance(output_list, list):
        return False

    if len(output_list) == 0:
        return False

    success = []
    for path in output_list:
        success.append(_delete_if_required(path, overwrite=overwrite))

    if not all(success):
        raise RuntimeError(f"Error while deleting files: {output_list}")

    return True


def _get_changed_folder(
    path: str,
    target_folder: str,
) -> str:
    """
    Change the folder of a path.

    Parameters
    ----------
    path: str
        The path to the file.

    target_folder: str
        The target folder.

    Returns
    -------
    str
        The path with the new folder.
    """
    assert isinstance(path, str), "path must be a string."
    assert len(path) > 0, "path must not be non-empty string."
    assert isinstance(target_folder, str), "target_folder must be a string."
    assert len(target_folder) > 0, "target_folder must not be non-empty string."

    filename = _get_filename_from_path(path)
    joined = os.path.join(target_folder, filename)

    unix_path = _get_unix_path(joined)

    return unix_path


def _check_is_path_glob(path: str) -> bool:
    """
    Check if a path is a glob.

    Parameters
    ----------
    path: str
        The path to check.

    Returns
    -------
    bool
        True if the path is a glob, False otherwise.
    """
    assert isinstance(path, str), "path must be a string."
    assert len(path) > 0, "path must not be non-empty string."

    if path[-5:] == ":glob":
        return True

    return False


def _get_paths_from_glob(path: str) -> List[str]:
    """
    Get a list of paths from a glob.

    Parameters
    ----------
    path: str
        The path to the glob.

    Returns
    -------
    list[str]
        The list of paths.
    """
    assert isinstance(path, str), "path must be a string."
    assert len(path) > 0, "path must not be non-empty string."
    assert _check_is_path_glob(path), "path must be a glob."

    pre_glob = path[:-5]

    if len(pre_glob) > 6 and "vsimem" in pre_glob[:10]:
        return _glob_vsimem(pre_glob)

    return glob(pre_glob)


def _parse_path(path: str) -> str:
    """
    Parse a path to an absolute unix path.

    Parameters
    ----------
    path: str
        The path to parse.

    Returns
    -------
    str
        The parsed path.
    """
    assert isinstance(path, str), "path must be a string."

    abspath = os.path.abspath(path)
    if "\\vsimem\\" in abspath:
        abspath = "/" + abspath.replace(os.path.abspath(os.sep), "")

    abspath = _get_unix_path(abspath)
    return abspath


def _get_augmented_path(
    path: str,
    prefix: str = "",
    suffix: str = "",
    change_ext: Optional[str] = None,
    folder: Optional[str] = None,
    add_uuid: bool = False,
    add_timestamp: bool = False,
) -> str:
    """
    Augments a path with a prefix, suffix, and uuid.
    Can also change the output directory.

    `{prefix}{filename}{uuid}{timestamp}{suffix}.{ext}`

    Parameters
    ----------
    path: str
        The path to the original file.

    prefix: str
        The prefix to add to the path. Default: "".

    suffix: str
        The suffix to add to the path. Default: "".

    change_ext: str. Optional.
        The extension to change the file to. Default: None.

    add_uuid: bool
        If True, add a uuid the path. Default: False.

    add_timestamp: bool
        If True, add a timestamp to the path. Default: False.
        Format: YYYYMMDD_HHMMSS.

    folder: str. Optional.
        The folder to save the file in. This can be specified as
        /vsimem/ to save the file in memory. Default: None.

    Returns
    -------
    str
        The augmented path.
    """
    assert isinstance(path, str), "path must be a string."
    assert isinstance(prefix, str), "prefix must be a string."
    assert isinstance(suffix, str), "suffix must be a string."
    assert isinstance(add_uuid, bool), "add_uuid must be a bool."
    assert folder is None or isinstance(folder, str), "folder must be a string."
    assert change_ext is None or isinstance(change_ext, str), "change_ext must be a string."
    assert len(path) > 0, "path must not be non-empty string."

    path = os.path.abspath(path)
    if "\\vsimem\\" in path:
        path = "/" + path.replace(os.path.abspath(os.sep), "")

    path = _get_unix_path(path)

    # Find the target folder
    target_folder = _get_dir_from_path(path)
    if folder is not None:
        assert len(folder) > 0, "folder must not be non-empty string."
        target_folder = _get_dir_from_path(folder)

    # Find the target extension
    ext = _get_ext_from_path(path)
    if change_ext is not None:
        assert len(change_ext) > 0, "change_ext must not be non-empty string."
        ext = ext.lstrip(".").lower()

    filename = _get_filename_from_path(path, with_ext=False)

    if add_uuid:
        uuid = "_" + str(uuid4().int)
    else:
        uuid = ""

    if add_timestamp:
        timestamp = "_" + utils_base._get_time_as_str()
    else:
        timestamp = ""

    augmented_filename = f"{prefix}{filename}{uuid}{timestamp}{suffix}.{ext}"
    augmented_path = os.path.join(target_folder, augmented_filename)

    augmented_path = _get_unix_path(augmented_path)

    return augmented_path


def _get_augmented_path_list(
    path_list: List[str],
    prefix: str = "",
    suffix: str = "",
    change_ext: Optional[str] = None,
    folder: Optional[str] = None,
    add_uuid: bool = False,
    add_timestamp: bool = False,
) -> List[str]:
    """
    Augments a list of paths with a prefix, suffix, and uuid.
    Can also change the output directory.

    Parameters
    ----------
    path_list: list[str]
        The list of paths to the original files.

    prefix: str
        The prefix to add to the path. Default: "".

    suffix: str
        The suffix to add to the path. Default: "".

    change_ext: str. Optional.
        The extension to change the file to. Default: None.

    folder: str. Optional.
        The folder to save the file in. This can be specified as
        /vsimem/ to save the file in memory.

    add_uuid: bool
        If True, add a uuid the path. Default: False.

    Returns
    -------
    list[str]
        The augmented paths.
    """
    assert isinstance(path_list, list), "path_list must be a list."
    assert len(path_list) > 0, "path_list must not be empty."
    assert isinstance(prefix, str), "prefix must be a string."
    assert isinstance(suffix, str), "suffix must be a string."
    assert isinstance(add_uuid, bool), "add_uuid must be a bool."
    assert change_ext is None or isinstance(change_ext, str), "change_ext must be a string."
    assert folder is None or isinstance(folder, str), "folder must be a string."
    assert change_ext is None or isinstance(change_ext, str), "change_ext must be a string."
    assert isinstance(add_timestamp, bool), "add_timestamp must be a bool."

    augmented_path_list = []
    for path in path_list:
        augmented_path_list.append(
            _get_augmented_path(
                path,
                prefix=prefix,
                suffix=suffix,
                change_ext=change_ext,
                folder=folder,
                add_uuid=add_uuid,
                add_timestamp=add_timestamp,
            )
        )

    return augmented_path_list


def _get_temp_filepath(
    name: Union[str, gdal.Dataset, ogr.DataSource] = "temp",
    *,
    ext: Optional[str] = None,
    prefix: str = "",
    suffix: str = "",
    add_uuid: bool = False,
    add_timestamp: bool = False,
) -> str:
    """
    Get a temporary filepath in vsimem.

    Parameters
    ----------
    name: Union[str, gdal.Dataset, ogr.DataSource], Optional.
        The name of the file. Default: "temp".

    ext: str, Optional.
        The extension of the file. Default: ".tif".

    prefix: str, Optional.
        The prefix to add to the path. Default: "".

    suffix: str, Optional.
        The suffix to add to the path. Default: "".

    add_uuid: bool, Optional.
        If True, add a uuid the path. Default: True.

    add_timestamp: bool, Optional.
        If True, add a timestamp to the path. Default: True.
        Format: YYYYMMDD_HHMMSS.

    Returns
    -------
    str
        The temporary filepath. (e.g. /vsimem/temp_20210101_000000_123456789.tif)
    """
    assert isinstance(name, (str, ogr.DataSource, gdal.Dataset)), "name must be a string or dataset"
    assert isinstance(ext, (str, type(None))), "ext must be a string or None."
    assert isinstance(prefix, str), "prefix must be a string."
    assert isinstance(suffix, str), "suffix must be a string."
    assert isinstance(add_uuid, bool), "add_uuid must be a bool."
    assert isinstance(add_timestamp, bool), "add_timestamp must be a bool."

    if isinstance(name, gdal.Dataset):
        path = name.GetDescription()
        name = os.path.splitext(os.path.basename(path))[0]
    elif isinstance(name, ogr.DataSource):
        path = name.GetDescription()
        name = os.path.splitext(os.path.basename(path))[0]
    else:
        path = name
        name = os.path.splitext(os.path.basename(name))[0]

    if add_uuid:
        uuid = "_" + str(uuid4().int)
    else:
        uuid = ""

    if add_timestamp:
        timestamp = "_" + utils_base._get_time_as_str()
    else:
        timestamp = ""

    if ext is None:
        ext = _get_ext_from_path(path)

    assert utils_gdal._check_is_valid_ext(ext), f"ext must be a valid extension. {ext} is not valid."

    filename = f"{prefix}{name}{uuid}{timestamp}{suffix}.{ext.lstrip('.').lower()}"
    filepath = os.path.join("/vsimem/", filename)

    # Add _1, _2, etc. if the file already exists
    if _check_file_exists(filepath):
        i = 1
        while _check_file_exists(filepath):
            filename = f"{prefix}{name}{uuid}{timestamp}{suffix}_{i}.{ext.lstrip('.').lower()}"
            filepath = os.path.join("/vsimem/", filename)
            i += 1

    return filepath
