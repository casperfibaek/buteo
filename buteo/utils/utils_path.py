"""### Generic utility functions. ###

Functions that make interacting with the toolbox easier.
"""

# Standard Library
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
    """Get the content of the vsimem folder.
    vsimem is the virtual memory folder of GDAL.
    By default, the content of the root folder is returned.

    Parameters
    ----------
    folder_name: Optional[str]
        The name of the folder to get the content of. Default: None (root folder)

    Returns
    -------
    List[str]
        A list of paths to files in the vsimem folder.

    Raises
    ------
    TypeError
        If folder_name is not None or str
    RuntimeError
        If GDAL lacks directory listing functionality or access fails
    """
    # Handle default case
    if folder_name is None:
        folder_name = "/vsimem/"
    elif not isinstance(folder_name, str):
        raise TypeError("folder_name must be None or str")

    # Ensure folder name ends with /
    folder_name = folder_name if folder_name.endswith("/") else folder_name + "/"

    # Initialize empty list for consistent return type
    vsimem: List[str] = []

    try:
        if hasattr(gdal, "listdir"):
            contents = gdal.listdir(folder_name)
            vsimem = [folder_name + item.name for item in contents]
        elif hasattr(gdal, "ReadDirRecursive"):
            contents = gdal.ReadDirRecursive(folder_name)
            vsimem = [folder_name + item for item in contents] if contents else []
        else:
            raise RuntimeError("GDAL installation lacks required directory listing functionality")

    except RuntimeError as e:
        raise RuntimeError(f"Failed to access vsimem folder {folder_name}. Error: {e}") from None

    # Convert paths to unix style
    converted = _get_unix_path(vsimem) if vsimem else []
    return converted if isinstance(converted, list) else [converted]


def _glob_vsimem(
    pattern: str,
) -> List[str]:
    """Find files in vsimem using glob.

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

    Raises
    ------
    TypeError
        If pattern is not a string.
    ValueError
        If pattern is empty.
    """
    if not isinstance(pattern, str):
        raise TypeError("pattern must be a string.")

    if not pattern:
        raise ValueError("pattern cannot be empty.")

    try:
        virtual_fs = _get_vsimem_content()
        if not virtual_fs:
            return []

        matches = [path for path in virtual_fs if fnmatch.fnmatch(path, pattern)]
        return matches
    except (RuntimeError, OSError) as e:
        warn(f"Error while searching vsimem: {str(e)}", UserWarning)
        return []


def _get_unix_path(path: Union[str, List[str], None]) -> Union[str, List[str]]:
    """Convert a path or list of paths to unix style path(s).
    Handles None values, empty strings, and ensures consistent return types.

    Parameters
    ----------
    path: Union[str, List[str], None]
        The path(s) to convert. Can be a single path string or list of paths.

    Returns
    -------
    Union[str, List[str]]
        The converted unix style path(s). Returns empty string for None input.
        Returns list of strings if input was a list, otherwise returns single string.

    Raises
    ------
    TypeError
        If path contains non-string elements
    ValueError
        If any path string is empty
    """
    # Handle None case
    if path is None:
        return ""

    # Track input type for consistent return
    input_is_list = isinstance(path, list)

    # Convert to list for unified processing
    paths = path if input_is_list else [path]

    # Validate inputs
    if not all(isinstance(p, str) for p in paths):
        raise TypeError("All paths must be strings")
    if any(len(p.strip()) == 0 for p in paths):
        raise ValueError("Paths cannot be empty strings")

    # Convert paths to unix style
    unix_paths = [
        "/".join(os.path.normpath(p).split(os.sep))
        for p in paths
    ]

    # Return same type as input
    return unix_paths if input_is_list else unix_paths[0]


def _check_file_exists(path: str) -> bool:
    """Check if a file exists. Also checks vsimem.
    Handles both absolute and relative paths and with or without trailing slash.

    Parameters
    ----------
    path: str
        The path to the file.

    Returns
    -------
    bool
        True if the file exists, False otherwise.

    Raises
    ------
    TypeError
        If path is not a string
    ValueError
        If path is empty
    """
    # Type checking
    if not isinstance(path, str):
        raise TypeError("path must be a string")
    if not path.strip():
        raise ValueError("path cannot be empty")

    try:
        # Check physical filesystem
        if os.path.isfile(path):
            return True

        abs_path = os.path.abspath(path)
        if os.path.isfile(abs_path):
            return True

        # Check virtual filesystem (vsimem)
        vsimem = _get_vsimem_content()
        if not vsimem:
            return False

        unix_path = _get_unix_path(path)
        return unix_path in vsimem

    except (OSError, RuntimeError):
        return False


def _check_file_exists_vsimem(path: str) -> bool:
    """Check if a file exists in vsimem (GDAL virtual memory filesystem).

    Parameters
    ----------
    path: str
        The path to the file.

    Returns
    -------
    bool
        True if the file exists in vsimem, False otherwise.

    Raises
    ------
    TypeError
        If path is not a string
    ValueError
        If path is empty
    """
    # Type checking
    if not isinstance(path, str):
        raise TypeError("path must be a string")
    if not path.strip():
        raise ValueError("path cannot be empty")

    try:
        # Get vsimem content and normalize path
        vsimem = _get_vsimem_content()
        if not vsimem:
            return False

        unix_path = _get_unix_path(path)
        return unix_path in vsimem

    except (RuntimeError, OSError):
        return False


def _check_dir_exists(path: str) -> bool:
    """Check if a folder exists. Also checks vsimem.
    Handles both absolute and relative paths and with or without trailing slash.

    Parameters
    ----------
    path: str
        The path to the folder.

    Returns
    -------
    bool
        True if the folder exists, False otherwise.

    Raises
    ------
    TypeError
        If path is not a string
    ValueError
        If path is empty
    """
    # Type checking
    if not isinstance(path, str):
        raise TypeError("path must be a string")
    if not path.strip():
        raise ValueError("path cannot be empty")

    try:
        # Check physical filesystem
        norm_path = os.path.normpath(path)
        if os.path.isdir(norm_path):
            return True

        abs_path = os.path.abspath(path)
        if os.path.isdir(abs_path):
            return True

        # Check for virtual filesystems
        path_parts = os.path.normpath(os.path.dirname(path)).split(os.sep)
        if any(vfs in path_parts for vfs in ["vsimem", "vsizip"]):
            # Verify the virtual directory exists in GDAL's virtual filesystem
            try:
                vsimem = _get_vsimem_content()
                if not vsimem:
                    return False

                unix_path = _get_unix_path(path)
                unix_path = unix_path[0] if isinstance(unix_path, list) else unix_path

                return any(p.startswith(unix_path) for p in vsimem)
            except (RuntimeError, OSError):
                return False

        return False

    except (OSError, RuntimeError):
        return False


def _check_dir_exists_vsimem(path: str) -> bool:
    """Check if a folder exists in vsimem (GDAL virtual memory filesystem).
    Handles both absolute and relative paths and with or without trailing slash.

    Parameters
    ----------
    path: str
        The path to the folder.

    Returns
    -------
    bool
        True if the folder exists in vsimem, False otherwise.

    Raises
    ------
    TypeError
        If path is not a string
    ValueError
        If path is empty
    """
    # Type checking
    if not isinstance(path, str):
        raise TypeError("path must be a string")
    if not path.strip():
        raise ValueError("path cannot be empty")

    try:
        # Get vsimem content and normalize paths
        vsimem = _get_vsimem_content()
        if not vsimem:
            return False

        # Convert path to unix style and normalize
        unix_path = _get_unix_path(path)
        unix_path = unix_path[0] if isinstance(unix_path, list) else unix_path
        unix_path = unix_path if unix_path.endswith("/") else unix_path + "/"

        # Check if any files in vsimem start with this path
        return any(p.startswith(unix_path) for p in vsimem)

    except (RuntimeError, OSError):
        return False


def _delete_dir_content(
    folder: str,
    delete_subfolders: bool = True,
) -> bool:
    """Delete all files and folders in a folder.
    If only the files are to be deleted, set delete_subfolders to False.

    Parameters
    ----------
    folder: str
        The path to the folder.
    delete_subfolders: bool
        If True, delete subfolders as well. Default: True

    Returns
    -------
    bool
        True if all content was successfully deleted, False otherwise.

    Raises
    ------
    TypeError
        If folder is not a string or delete_subfolders is not a bool
    ValueError
        If folder is empty string
    RuntimeError
        If folder doesn't exist
    """
    # Type checking
    if not isinstance(folder, str):
        raise TypeError("folder must be a string")
    if not isinstance(delete_subfolders, bool):
        raise TypeError("delete_subfolders must be a bool")

    # Value checking
    if not folder.strip():
        raise ValueError("folder cannot be empty")
    if not _check_dir_exists(folder):
        raise RuntimeError(f"folder does not exist: {folder}")

    try:
        # Handle physical filesystem
        if not _check_dir_exists_vsimem(folder):
            for item in os.listdir(folder):
                try:
                    path = os.path.join(folder, item)
                    if os.path.isfile(path):
                        os.remove(path)
                    elif os.path.isdir(path) and delete_subfolders:
                        shutil.rmtree(path)
                except (OSError, RuntimeError) as e:
                    warn(f"Failed to remove {path}: {str(e)}", UserWarning)
                    return False

        # Handle virtual filesystem (vsimem)
        else:
            try:
                vsimem = _get_vsimem_content(folder)
                if vsimem:  # Only attempt deletion if there are files
                    for f in vsimem:
                        gdal.Unlink(f)
            except RuntimeError as e:
                warn(f"Failed to access or clear vsimem folder {folder}: {str(e)}", UserWarning)
                return False

        # Verify deletion was successful
        remaining_content = (
            os.listdir(folder) if not _check_dir_exists_vsimem(folder)
            else _get_vsimem_content(folder)
        )

        if remaining_content and (delete_subfolders or
            all(not os.path.isdir(os.path.join(folder, x)) for x in remaining_content)):
            warn(f"Failed to remove all content from: {folder}", UserWarning)
            return False

        return True

    except (OSError, RuntimeError, PermissionError) as e:
        warn(f"Error while clearing folder {folder}: {str(e)}", UserWarning)
        return False


def _delete_dir(folder: str) -> bool:
    """Delete a folder and all its content. Handles both physical and virtual (vsimem) folders.

    Parameters
    ----------
    folder: str
        The path to the folder to delete.

    Returns
    -------
    bool
        True if the folder was successfully deleted, False otherwise.

    Raises
    ------
    TypeError
        If folder is not a string
    ValueError
        If folder is empty string
    RuntimeError
        If folder doesn't exist
    """
    # Type checking
    if not isinstance(folder, str):
        raise TypeError("folder must be a string")
    if not folder.strip():
        raise ValueError("folder cannot be empty")
    if not _check_dir_exists(folder):
        raise RuntimeError(f"folder does not exist: {folder}")

    try:
        # Handle physical filesystem
        if not _check_dir_exists_vsimem(folder):
            try:
                shutil.rmtree(folder)
                return not _check_dir_exists(folder)
            except (OSError, PermissionError) as e:
                warn(f"Failed to remove folder {folder}: {str(e)}", UserWarning)
                return False

        # Handle virtual filesystem (vsimem)
        try:
            vsimem = _get_vsimem_content(folder)
            # Delete all files in the folder
            for f in vsimem:
                gdal.Unlink(f)
            # Delete the folder itself
            if folder != "/vsimem/":  # Don't delete root vsimem
                gdal.Unlink(folder)
            return not _check_dir_exists_vsimem(folder)
        except RuntimeError as e:
            warn(f"Failed to remove vsimem folder {folder}: {str(e)}", UserWarning)
            return False

    except (OSError, RuntimeError, PermissionError) as e:
        warn(f"Error while deleting folder {folder}: {str(e)}", UserWarning)
        return False


def _delete_file(file: str) -> bool:
    """Delete a file from physical or virtual (vsimem) filesystem.

    Parameters
    ----------
    file: str
        The path to the file to delete.

    Returns
    -------
    bool
        True if the file was successfully deleted, False otherwise.

    Raises
    ------
    TypeError
        If file is not a string
    ValueError
        If file is empty string
    RuntimeError
        If file doesn't exist
    """
    # Type checking
    if not isinstance(file, str):
        raise TypeError("file must be a string")
    if not file.strip():
        raise ValueError("file cannot be empty")
    if not _check_file_exists(file):
        raise RuntimeError(f"file does not exist: {file}")

    try:
        # Handle physical filesystem
        if not _check_file_exists_vsimem(file):
            try:
                os.remove(file)
                return not _check_file_exists(file)
            except (OSError, PermissionError) as e:
                warn(f"Failed to remove file {file}: {str(e)}", UserWarning)
                return False

        # Handle virtual filesystem (vsimem)
        try:
            gdal.Unlink(file)
            return not _check_file_exists_vsimem(file)
        except RuntimeError as e:
            warn(f"Failed to remove vsimem file {file}: {str(e)}", UserWarning)
            return False

    except (OSError, RuntimeError) as e:
        warn(f"Error while deleting file {file}: {str(e)}", UserWarning)
        return False


def _create_dir_if_not_exists(path: str) -> str:
    """Make a directory if it does not exist.
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
        The path to the folder (normalized to unix-style).

    Raises
    ------
    TypeError
        If path is not a string
    ValueError
        If path is empty
    RuntimeError
        If directory creation fails
    """
    # Type checking
    if not isinstance(path, str):
        raise TypeError("path must be a string")
    if not path.strip():
        raise ValueError("path cannot be empty")

    # Normalize path to unix-style
    unix_path = _get_unix_path(path)
    unix_path = unix_path[0] if isinstance(unix_path, list) else unix_path

    # Check if directory already exists (either on disk or in vsimem)
    if _check_dir_exists(unix_path):
        return unix_path

    try:
        # Handle physical filesystem
        if not _check_dir_exists_vsimem(unix_path):
            os.makedirs(unix_path, exist_ok=True)

            if not _check_dir_exists(unix_path):
                raise RuntimeError(f"Failed to create directory: {unix_path}")

        # For vsimem, we don't create the directory as it's not possible
        # without creating a file in it. We just return the path.
        return unix_path

    except (OSError, PermissionError) as e:
        raise RuntimeError(f"Failed to create directory {unix_path}: {str(e)}") from None


def _get_dir_from_path(path: str) -> str:
    """Get the directory of a file or folder path. Also works for vsimem and vsizip paths.
    For regular paths, returns the absolute normalized unix-style path with trailing slash.
    For virtual paths (vsimem/vsizip), returns the virtual root with trailing slash.

    Parameters
    ----------
    path: str
        The path to get the directory from.

    Returns
    -------
    str
        The directory path with trailing slash.

    Raises
    ------
    TypeError
        If path is not a string
    ValueError
        If path is empty
    """
    # Type checking
    if not isinstance(path, str):
        raise TypeError("path must be a string")
    if not path.strip():
        raise ValueError("path cannot be empty")

    try:
        # Handle virtual filesystems
        path_parts = os.path.normpath(path).split(os.sep)
        if any(vfs in path_parts for vfs in ["vsimem", "vsizip"]):
            if "vsimem" in path_parts:
                return "/vsimem/"
            return "/vsizip/"

        # Handle physical filesystem
        if _check_dir_exists(path):
            # If path is already a directory, normalize it
            abs_path = os.path.abspath(path)
            unix_path = _get_unix_path(abs_path)
            unix_path = unix_path[0] if isinstance(unix_path, list) else unix_path
            return unix_path if unix_path.endswith("/") else f"{unix_path}/"
        else:
            # Get directory of file path
            dir_path = os.path.dirname(os.path.abspath(path))
            unix_path = _get_unix_path(dir_path)
            unix_path = unix_path[0] if isinstance(unix_path, list) else unix_path
            return unix_path if unix_path.endswith("/") else f"{unix_path}/"

    except (OSError, RuntimeError, AttributeError, TypeError, ValueError) as e:
        raise RuntimeError(f"Failed to get directory from path {path}: {str(e)}") from None


def _get_filename_from_path(
    path: str,
    with_ext: bool = True,
) -> str:
    """Get the filename of a file. Also works for vsimem paths.

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

    Raises
    ------
    TypeError
        If path is not a string or with_ext is not a bool
    ValueError
        If path is empty
    """
    # Type checking
    if not isinstance(path, str):
        raise TypeError("path must be a string")
    if not isinstance(with_ext, bool):
        raise TypeError("with_ext must be a bool")

    # Value checking
    if not path.strip():
        raise ValueError("path cannot be empty")

    try:
        # Handle paths with backslashes
        path = path.replace("\\", "/")

        # Get basename, handling both Unix and Windows paths
        basename = path.rstrip("/").split("/")[-1]

        if with_ext:
            return basename

        # Split at last dot to handle multiple dots in filename
        name_parts = basename.rsplit(".", 1)
        return name_parts[0] if len(name_parts) > 1 else basename

    except (TypeError, ValueError, AttributeError) as e:
        raise RuntimeError(f"Failed to get filename from path {path}: {str(e)}") from None


def _get_ext_from_path(path: str) -> str:
    """Get the extension of a file. If the file has no extension, raise an error.
    The extension is returned without a dot.

    Parameters
    ----------
    path: str
        The path to the file.

    Returns
    -------
    str
        The extension of the file without the leading dot.

    Raises
    ------
    TypeError
        If path is not a string
    ValueError
        If path is empty
    RuntimeError
        If file has no extension
    """
    # Type checking
    if not isinstance(path, str):
        raise TypeError("path must be a string")
    if not path.strip():
        raise ValueError("path cannot be empty")

    try:
        # Handle paths with backslashes
        path = path.replace("\\", "/")

        # Get basename, handling both Unix and Windows paths
        basename = path.rstrip("/").split("/")[-1]

        # Split at last dot to handle multiple dots in filename
        ext_parts = basename.rsplit(".", 1)

        if len(ext_parts) < 2 or not ext_parts[1]:
            raise RuntimeError(f"File has no extension: {path}")

        return ext_parts[1].lower()

    except Exception as e:
        if isinstance(e, RuntimeError):
            raise
        raise RuntimeError(f"Failed to get extension from path {path}: {str(e)}") from None


def _get_changed_path_ext(
    path: str,
    target_ext: str,
) -> str:
    """Update the extension of a file path.
    The path can be a physical path or a virtual path (vsimem/vsizip).

    Parameters
    ----------
    path: str
        The path to the file.
    target_ext: str
        The new extension (with or without dot).

    Returns
    -------
    str
        The path with the new extension in unix style.

    Raises
    ------
    TypeError
        If path or target_ext are not strings or are None
    ValueError
        If path or target_ext are empty or contain only whitespace
    RuntimeError
        If path has no extension or path manipulation fails
    """
    # Type checking
    if path is None:
        raise TypeError("path cannot be None")
    if target_ext is None:
        raise TypeError("target_ext cannot be None")
    if not isinstance(path, str):
        raise TypeError("path must be a string")
    if not isinstance(target_ext, str):
        raise TypeError("target_ext must be a string")

    # Value checking
    if not path.strip():
        raise ValueError("path cannot be empty or whitespace")
    if not target_ext.strip():
        raise ValueError("target_ext cannot be empty or whitespace")

    try:
        # Normalize and validate the path
        norm_path = path.strip()
        if not _check_is_valid_filepath(norm_path):
            raise RuntimeError(f"Invalid file path: {path}")

        # Strip leading dots and normalize extension
        target_ext = target_ext.strip().lstrip('.').lower()

        # Get directory and filename components
        directory = _get_dir_from_path(norm_path)
        filename = _get_filename_from_path(norm_path, with_ext=False)

        # Handle special case where target_ext is empty
        if not target_ext:
            new_path = os.path.join(directory, filename)
        else:
            new_path = os.path.join(directory, f"{filename}.{target_ext}")

        # Convert to unix style path and ensure string return type
        unix_path = _get_unix_path(new_path)
        if isinstance(unix_path, list):
            if not unix_path:
                raise RuntimeError("Path conversion failed")
            return unix_path[0]

        return unix_path

    except Exception as e:
        if isinstance(e, (TypeError, ValueError, RuntimeError)):
            raise
        raise RuntimeError(f"Failed to change extension for path {path}: {str(e)}") from None


def _check_is_dir(path: str) -> bool:
    """Check if a path is a directory. Works with both physical and virtual filesystems.
    For virtual filesystems (vsimem/vsizip), checks if the path exists and is a directory.
    For physical filesystems, converts to absolute path and checks if it exists and is a directory.

    Parameters
    ----------
    path: str
        The path to test.

    Returns
    -------
    bool
        True if path is a directory, False otherwise.

    Raises
    ------
    TypeError
        If path is not a string
    ValueError
        If path is empty string or consists only of whitespace
    """
    # Type checking
    if not isinstance(path, str):
        raise TypeError("path must be a string")

    # Value checking
    if not path.strip():
        raise ValueError("path cannot be empty or whitespace")

    try:
        # Normalize path by removing trailing slashes
        norm_path = path.rstrip("/").rstrip("\\")

        # Handle root virtual filesystem paths
        if norm_path in ["/vsimem", "/vsizip"]:
            return True

        # Check for virtual filesystem paths
        if any(vfs in ["vsimem", "vsizip"] for vfs in norm_path.split(os.sep)):
            return _check_dir_exists_vsimem(norm_path)

        # Handle physical filesystem
        try:
            # Convert to absolute path to handle relative paths
            abs_path = os.path.abspath(norm_path)
            return os.path.exists(abs_path) and os.path.isdir(abs_path)
        except (OSError, RuntimeError):
            return False

    except (OSError, RuntimeError) as e:
        warn(f"Error checking if path is directory: {str(e)}", UserWarning)
        return False


def _check_is_valid_mem_filepath(path: str) -> bool:
    """Check if a path is a valid memory path that has an extension.
    Validates paths in GDAL's virtual filesystems (vsimem, vsizip).

    Parameters
    ----------
    path: str
        The path to test.

    Returns
    -------
    bool
        True if path is a valid memory path with extension, False otherwise.

    Raises
    ------
    TypeError
        If path is not a string
    """
    # Type checking
    if not isinstance(path, str):
        raise TypeError("path must be a string")

    try:
        # Handle empty or whitespace-only strings
        if not path.strip():
            return False

        # Convert path to normalized form
        path = path.replace("\\", "/")  # Handle Windows paths
        path_parts = path.strip("/").split("/")

        # Check for virtual filesystem prefixes
        if not any(vfs in ["vsimem", "vsizip"] for vfs in path_parts):
            return False

        # Verify path has an extension
        try:
            ext = _get_ext_from_path(path)
            if not ext:
                return False
        except (RuntimeError, ValueError):
            return False

        # Validate path structure
        if len(path_parts) < 2:  # Need at least /vfs/filename.ext
            return False

        return True

    except (TypeError, ValueError, AttributeError) as e:
        warn(f"Error validating memory filepath: {str(e)}", UserWarning)
        return False


def _check_is_valid_non_mem_filepath(path: str) -> bool:
    """Check if a path is a valid path that has an extension.
    Does not check if the file exists and does not check if the path is in memory.

    Parameters
    ----------
    path: str
        The path to test.

    Returns
    -------
    bool
        True if path is a valid path with an extension, False otherwise.

    Raises
    ------
    TypeError
        If path is not a string
    """
    # Type checking
    if not isinstance(path, str):
        raise TypeError("path must be a string")

    try:
        # Check for empty path
        if not path.strip():
            return False

        # Check for extension
        try:
            if not _get_ext_from_path(path):
                return False
        except (RuntimeError, ValueError):
            return False

        # Check if parent directory exists
        try:
            folder = _get_dir_from_path(path)
            if not _check_dir_exists(folder):
                return False
        except (RuntimeError, ValueError):
            return False

        return True

    except (AttributeError, TypeError):
        return False


def _check_is_valid_filepath(path: str) -> bool:
    """Check if a path is a valid path that has an extension. Path can be in memory.

    Parameters
    ----------
    path: str
        The path to test.

    Returns
    -------
    bool
        True if path is a valid path, False otherwise.

    Raises
    ------
    TypeError
        If path is not a string
    """
    # Type checking
    if not isinstance(path, str):
        raise TypeError("path must be a string")

    try:
        # Handle empty or whitespace-only paths
        if not path.strip():
            return False

        # Check if path is valid memory path or non-memory path
        return (_check_is_valid_mem_filepath(path) or
                _check_is_valid_non_mem_filepath(path))

    except (TypeError, ValueError, AttributeError):
        # Catch any unexpected errors from sub-functions
        return False


def _check_is_valid_filepath_list(path_list: List[str]) -> bool:
    """Check if a list of paths is a valid list of paths that have an extension. Paths can be in memory.

    Parameters
    ----------
    path_list: List[str]
        The list of paths to test.

    Returns
    -------
    bool
        True if path_list is a valid list of paths, False otherwise.

    Raises
    ------
    TypeError
        If path_list is not a list
        If any path in the list is not a string
    """
    # Type check for path_list
    if not isinstance(path_list, list):
        raise TypeError("path_list must be a list")

    # Handle empty list case
    if not path_list:
        return False

    try:
        # Check each path is a string and valid
        for path in path_list:
            if not isinstance(path, str):
                raise TypeError("All paths must be strings")

            if not path or path.isspace():
                return False

            if not _check_is_valid_filepath(path):
                return False

        return True

    except (TypeError, ValueError, AttributeError) as e:
        if isinstance(e, TypeError):
            raise
        return False


def _check_is_valid_output_filepath(
    path: str,
    overwrite: bool = True,
) -> bool:
    """Check if a path is a valid output path that has an extension. Path can be in memory.
    If the file already exists and overwrite is false, return False.

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

    Raises
    ------
    TypeError
        If path is not a string or overwrite is not a bool
    ValueError
        If path is empty or None
    """
    # Type checking
    if not isinstance(path, str):
        raise TypeError("path must be a string")
    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be a bool")

    # Value checking
    if not path.strip():
        raise ValueError("path cannot be empty")

    try:
        # Check if path is valid
        if not _check_is_valid_filepath(path):
            return False

        # Check if file exists and overwrite is allowed
        if not overwrite and _check_file_exists(path):
            return False

        # Get directory path and check if parent directory exists/is writable
        dir_path = _get_dir_from_path(path)
        if not _check_dir_exists(dir_path):
            return False

        return True

    except (TypeError, ValueError, RuntimeError) as e:
        if isinstance(e, (TypeError, ValueError)):
            raise
        return False


def _check_is_valid_output_path_list(
    output_list: List[str],
    overwrite: bool = True,
) -> bool:
    """Check if a list of output paths are valid.

    Parameters
    ----------
    output_list: List[str]
        The list of paths to the files.
    overwrite: bool
        True if existing files can be overwritten, False otherwise.

    Returns
    -------
    bool
        True if all paths in the list are valid output paths, False otherwise.

    Raises
    ------
    TypeError
        If output_list is not a list or overwrite is not a bool
        If any path in the list is not a string
    ValueError
        If output_list is empty or any path in it is empty
    """
    # Type checking
    if not isinstance(output_list, list):
        raise TypeError("output_list must be a list")
    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be a bool")

    # Check for empty list
    if not output_list:
        raise ValueError("output_list cannot be empty")

    try:
        # Check each path in the list
        for path in output_list:
            if path is None:
                raise TypeError("Paths cannot be None")
            if not isinstance(path, str):
                raise TypeError("All paths must be strings")
            if not path.strip():
                raise ValueError("Paths cannot be empty strings")

            if not _check_is_valid_output_filepath(path, overwrite=overwrite):
                return False

        return True

    except (TypeError, ValueError):
        return False


def _delete_if_required(
    path: str,
    overwrite: bool = True,
) -> bool:
    """Delete a file if overwrite is True and the file exists.

    Parameters
    ----------
    path: str
        The path to the file to potentially delete.
    overwrite: bool
        If True, delete the file if it exists.

    Returns
    -------
    bool
        True if deletion was successful or not needed, False if deletion failed.

    Raises
    ------
    TypeError
        If path is not a string or overwrite is not a bool
    ValueError
        If path is empty or None
    """
    # Type checking
    if not isinstance(path, str):
        raise TypeError("path must be a string")
    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be a bool")

    # Value checking
    if not path.strip():
        raise ValueError("path cannot be empty")

    try:
        # If overwrite is False or file doesn't exist, no action needed
        if not overwrite or not _check_file_exists(path):
            return True

        # Handle virtual filesystem (vsimem) files
        if _check_is_valid_mem_filepath(path):
            try:
                gdal.Unlink(path)
            except RuntimeError as e:
                warn(f"Failed to delete vsimem file {path}: {str(e)}", UserWarning)
                return False
        # Handle physical files
        else:
            try:
                os.remove(path)
            except (OSError, PermissionError) as e:
                warn(f"Failed to delete file {path}: {str(e)}", UserWarning)
                return False

        # Verify deletion was successful
        if _check_file_exists(path):
            warn(f"File still exists after deletion attempt: {path}", UserWarning)
            return False

        return True

    except (OSError, RuntimeError, PermissionError) as e:
        warn(f"Error while deleting file {path}: {str(e)}", UserWarning)
        return False


def _delete_if_required_list(
    output_list: List[str],
    overwrite: bool = True,
) -> bool:
    """Delete a list of files if overwrite is True and the files exist.

    Parameters
    ----------
    output_list: List[str]
        The list of paths to the files.
    overwrite: bool
        If True, delete files if they exist.

    Returns
    -------
    bool
        True if all deletions were successful or not needed, False if any deletion failed.

    Raises
    ------
    TypeError
        If output_list is not a list or overwrite is not a bool
        If any path in output_list is not a string
    ValueError
        If output_list is empty or contains empty strings
    RuntimeError
        If any deletion operation fails
    """
    # Type checking
    if not isinstance(output_list, list):
        raise TypeError("output_list must be a list")
    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be a bool")

    # Check for empty list
    if not output_list:
        raise ValueError("output_list cannot be empty")

    try:
        # Validate all paths before attempting any deletions
        for path in output_list:
            if path is None:
                raise TypeError("Paths cannot be None")
            if not isinstance(path, str):
                raise TypeError("All paths must be strings")
            if not path.strip():
                raise ValueError("Paths cannot be empty strings")

        # Attempt deletion for each file
        deletion_results = []
        for path in output_list:
            try:
                result = _delete_if_required(path, overwrite=overwrite)
                deletion_results.append(result)
            except (OSError, PermissionError, RuntimeError) as e:
                warn(f"Failed to delete {path}: {str(e)}", UserWarning)
                deletion_results.append(False)

        # Check if all deletions were successful
        if not all(deletion_results):
            failed_paths = [p for p, r in zip(output_list, deletion_results) if not r]
            raise RuntimeError(f"Failed to delete files: {failed_paths}")

        return True

    except (OSError, RuntimeError, PermissionError, TypeError, ValueError) as e:
        warn(f"Error during deletion operations: {str(e)}", UserWarning)
        return False

# TODO: MADE IT TO HERE

def _get_changed_folder(
    path: str,
    target_folder: str,
) -> str:
    """Change the folder of a path.

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
    """Check if a path is a glob.

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
    """Get a list of paths from a glob.

    Parameters
    ----------
    path: str
        The path to the glob.

    Returns
    -------
    List[str]
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
    """Parse a path to an absolute unix path.

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
    """Augments a path with a prefix, suffix, and uuid.
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
        assert isinstance(change_ext, str), "change_ext must be a string."
        ext = change_ext

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
    """Augments a list of paths with a prefix, suffix, and uuid.
    Can also change the output directory.

    Parameters
    ----------
    path_list: List[str]
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
    List[str]
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
    """Get a temporary filepath in vsimem.

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
