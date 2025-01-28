"""### Generic utility functions. ###

Functions that make interacting with the toolbox easier.
"""

# Standard Library
import os
import fnmatch
from glob import glob
from uuid import uuid4
from typing import List, Optional, Union
from warnings import warn

# External
from osgeo import gdal, ogr

# Internal
from buteo.utils import utils_base, utils_gdal



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
    converted = _get_unix_path_list(vsimem) if vsimem else []
    return converted if isinstance(converted, list) else [converted]


def _get_unix_path(path: Union[str, None]) -> str:
    """Convert a path to unix style path.
    Handles None values, empty strings, and ensures consistent return types.
    Converts backslashes and forward slashes to unix style forward slashes.

    To handle lists, please use _get_unix_path_list. This function is for single paths.    

    Parameters
    ----------
    path: Union[str, None]
        The path to convert.

    Returns
    -------
    str
        The converted unix style path.

    Raises
    ------
    TypeError
        If path is None
        If path is not a string
        If path is a list
    ValueError
        If path is empty or only whitespace
    RuntimeError
        If path conversion fails
    """
    # Type checking
    if path is None:
        raise TypeError("path cannot be None")
    if isinstance(path, list):
        raise TypeError("path must not be a list")
    if not isinstance(path, str):
        raise TypeError("path must be a string")

    # Value checking
    if not path.strip():
        raise ValueError("path cannot be empty or whitespace")

    try:
        # Normalize path (handles '.' and '..' and multiple slashes)
        normalized = os.path.normpath(path)

        # Convert to unix style with forward slashes
        unix_path = "/".join(normalized.split(os.sep))

        # Ensure special paths retain leading slash
        if path.startswith("/") and not unix_path.startswith("/"):
            unix_path = f"/{unix_path}"

        # Handle virtual filesystem paths
        if "vsimem" in unix_path and not unix_path.startswith("/"):
            unix_path = f"/{unix_path}"

        # Return empty string for empty paths (defensive)
        return unix_path if unix_path else ""

    except Exception as e:
        raise RuntimeError(f"Failed to convert path to unix style: {str(e)}") from None


def _get_unix_path_list(paths: List[str]) -> List[str]:
    """Convert a list of paths to unix style paths.
    Handles empty lists and ensures consistent return types.

    Parameters
    ----------
    paths: List[str]
        The list of paths to convert.

    Returns
    -------
    List[str]
        The converted unix style paths.
        Returns empty list if input is empty list.

    Raises
    ------
    TypeError
        If paths is not a list or contains non-string elements
        If paths is None
    ValueError
        If any path string is empty or only whitespace
    """
    # Handle None case
    if paths is None:
        raise TypeError("paths cannot be None")

    # Type checking
    if not isinstance(paths, list):
        raise TypeError("paths must be a list")

    # Handle empty list
    if not paths:
        return []

    # Validate inputs
    if not all(isinstance(p, str) for p in paths):
        raise TypeError("All paths must be strings")

    # Check for empty strings
    empty_paths = [p for p in paths if not p.strip()]
    if empty_paths:
        raise ValueError(f"Found empty paths at indices: {[paths.index(p) for p in empty_paths]}")

    try:
        # Convert paths to unix style
        unix_paths = [_get_unix_path(p) for p in paths]

        return unix_paths

    except (AttributeError, TypeError) as e:
        raise RuntimeError(f"Failed to convert paths: {str(e)}") from None


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
        unix_path = unix_path if unix_path.endswith("/") else unix_path + "/"

        # Check if any files in vsimem start with this path
        return any(p.startswith(unix_path) for p in vsimem)

    except (RuntimeError, OSError):
        return False


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

            return unix_path if unix_path.endswith("/") else f"{unix_path}/"
        else:
            # Get directory of file path
            dir_path = os.path.dirname(os.path.abspath(path))
            unix_path = _get_unix_path(dir_path)

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

        # For vsimem paths, check if any files exist in this directory
        if "vsimem" in norm_path:
            dir_path = norm_path if norm_path.endswith("/") else norm_path + "/"
            vsimem_contents = _get_vsimem_content()
            return any(p.startswith(dir_path) for p in vsimem_contents)

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


def _check_dir_is_vsimem(path: str) -> bool:
    """Check if a path is a vsimem path.

    Parameters
    ----------
    path: str
        The path to test.
    
    Returns
    -------
    bool
        True if path is a vsimem path, False otherwise.
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
        return any(vfs in ["vsimem", "vsizip"] for vfs in path_parts)

    except (TypeError, ValueError, AttributeError):
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

        # Get directory path and check if parent directory exists/is writable or vsimem
        if not _check_dir_is_vsimem(path) and not _check_dir_exists(_get_dir_from_path(path)):
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


def _get_changed_folder(
    path: str,
    target_folder: str,
) -> str:
    """Change the folder of a path while preserving the filename.
    Works with both physical and virtual (vsimem) paths.

    Parameters
    ----------
    path: str
        The path to modify. Must contain a filename with extension.
    target_folder: str
        The new folder path. Can be physical or virtual (vsimem) path.

    Returns
    -------
    str
        The path with the new folder in unix style.

    Raises
    ------
    TypeError
        If path or target_folder are not strings
    ValueError
        If path or target_folder are empty strings
        If path doesn't contain a filename with extension
    RuntimeError
        If path manipulation fails
    """
    # Type checking
    if not isinstance(path, str):
        raise TypeError("path must be a string")
    if not isinstance(target_folder, str):
        raise TypeError("target_folder must be a string")

    # Value checking
    if not path.strip():
        raise ValueError("path cannot be empty")
    if not target_folder.strip():
        raise ValueError("target_folder cannot be empty")

    try:
        # Validate path has filename and extension
        if not _check_is_valid_filepath(path):
            raise ValueError(f"Invalid file path: {path}")

        # Get filename and normalize target folder
        filename = _get_filename_from_path(path)
        norm_folder = _get_dir_from_path(target_folder)

        # Join paths and convert to unix style
        new_path = os.path.join(norm_folder, filename)
        unix_path = _get_unix_path(new_path)

        return unix_path

    except (OSError, RuntimeError) as e:
        raise RuntimeError(f"Failed to change folder for path {path}: {str(e)}") from None


def _check_is_path_glob(path: str) -> bool:
    """Check if a path is a glob pattern by determining if it ends with ':glob'.

    Parameters
    ----------
    path: str
        The path to check.

    Returns
    -------
    bool
        True if the path ends with ':glob', False otherwise.

    Raises
    ------
    TypeError
        If path is not a string
    ValueError
        If path is empty or only whitespace
    """
    # Type checking
    if not isinstance(path, str):
        raise TypeError("path must be a string")

    # Value checking
    if not path.strip():
        raise ValueError("path cannot be empty or whitespace")

    try:
        # Check if path ends with ':glob'
        # Using endswith is more robust than string slicing
        return path.endswith(":glob")

    except (AttributeError, TypeError) as e:
        # Handle any unexpected string operation errors
        raise TypeError(f"Failed to check if path is glob: {str(e)}") from None


def _get_paths_from_glob(path: str) -> List[str]:
    """Get a list of paths from a glob pattern. The path must end with ':glob'.
    Works with both physical filesystem and GDAL virtual filesystems (vsimem).

    Parameters
    ----------
    path: str
        The path to the glob pattern. Must end with ':glob'.

    Returns
    -------
    List[str]
        The list of matched paths in unix style. Empty list if no matches found.

    Raises
    ------
    TypeError
        If path is not a string
    ValueError
        If path is empty or does not end with ':glob'
    RuntimeError
        If glob pattern evaluation fails
    """
    # Type checking
    if not isinstance(path, str):
        raise TypeError("path must be a string")

    # Value checking
    if not path.strip():
        raise ValueError("path cannot be empty")
    if not _check_is_path_glob(path):
        raise ValueError("path must end with ':glob'")

    try:
        # Remove ':glob' suffix and normalize path
        pattern = path[:-5].strip()
        if not pattern:
            raise ValueError("glob pattern cannot be empty")

        # Handle virtual filesystem (vsimem)
        if pattern.startswith("/vsimem/") or "\\vsimem\\" in pattern:
            matches = _glob_vsimem(pattern)
        else:
            matches = glob(pattern)

        try:
            return_paths = [_get_unix_path(p) for p in matches]

            for p in return_paths:
                if not _check_is_valid_filepath(p):
                    raise RuntimeError(f"Invalid file path: {p}")

            return return_paths

        except (OSError, RuntimeError) as e:
            raise RuntimeError(f"Failed to evaluate glob pattern: {str(e)}") from None

    except Exception as e:
        if isinstance(e, (TypeError, ValueError, RuntimeError)):
            raise
        raise RuntimeError(f"Failed to get paths from glob: {str(e)}") from None


def _parse_path(path: str) -> str:
    """Parse a path to an absolute unix-style path.
    Works with both physical filesystem and GDAL virtual filesystem paths.

    Parameters
    ----------
    path: str
        The path to parse.

    Returns
    -------
    str
        The parsed absolute unix-style path.

    Raises
    ------
    TypeError
        If path is not a string or is None
    ValueError
        If path is empty or consists only of whitespace
    RuntimeError
        If path conversion fails
    """
    # Type checking
    if not isinstance(path, str):
        raise TypeError("path must be a string")

    # Value checking
    if not path.strip():
        raise ValueError("path cannot be empty or whitespace")

    try:
        # Handle virtual filesystem paths
        if path.startswith("/vsimem/") or path.startswith("/vsizip/"):
            unix_path = _get_unix_path(path)
        else:
            # Convert to absolute path and handle Windows paths
            abspath = os.path.abspath(path)
            if "\\vsimem\\" in abspath:
                abspath = "/" + abspath.replace(os.path.abspath(os.sep), "")
            unix_path = _get_unix_path(abspath)

        return unix_path

    except (OSError, RuntimeError) as e:
        raise RuntimeError(f"Failed to parse path {path}: {str(e)}") from None


def _get_augmented_path(
    path: str,
    prefix: str = "",
    suffix: str = "",
    change_ext: Optional[str] = None,
    folder: Optional[str] = None,
    add_uuid: bool = False,
    add_timestamp: bool = False,
) -> str:
    """Augments a path with a prefix, suffix, and optional components.
    Format: {prefix}{filename}{uuid}{timestamp}{suffix}.{ext}

    Parameters
    ----------
    path: str
        The path to the original file.
    prefix: str
        The prefix to add to the path. Default: "".
    suffix: str
        The suffix to add to the path. Default: "".
    change_ext: Optional[str]
        The extension to change the file to. Default: None.
    folder: Optional[str]
        The folder to save the file in. Can be /vsimem/ for memory. Default: None.
    add_uuid: bool
        If True, add a uuid to the path. Default: False.
    add_timestamp: bool
        If True, add a timestamp (YYYYMMDD_HHMMSS). Default: False.

    Returns
    -------
    str
        The augmented path in unix style.

    Raises
    ------
    TypeError
        If inputs are not of correct type
    ValueError
        If path is empty or invalid
    RuntimeError
        If path manipulation fails
    """
    # Type checking
    if not isinstance(path, str):
        raise TypeError("path must be a string")
    if not isinstance(prefix, str):
        raise TypeError("prefix must be a string")
    if not isinstance(suffix, str):
        raise TypeError("suffix must be a string")
    if not isinstance(add_uuid, bool):
        raise TypeError("add_uuid must be a bool")
    if not isinstance(add_timestamp, bool):
        raise TypeError("add_timestamp must be a bool")
    if change_ext is not None and not isinstance(change_ext, str):
        raise TypeError("change_ext must be None or string")
    if folder is not None and not isinstance(folder, str):
        raise TypeError("folder must be None or string")

    # Value checking
    if not path.strip():
        raise ValueError("path cannot be empty")
    if folder is not None and not folder.strip():
        raise ValueError("folder cannot be empty")
    if change_ext is not None and not change_ext.strip():
        raise ValueError("change_ext cannot be empty")

    try:
        # Parse and normalize input path
        parsed_path = _parse_path(path)
        if not _check_is_valid_filepath(parsed_path):
            raise ValueError(f"Invalid file path: {parsed_path}")

        # Handle target folder
        target_folder = _get_dir_from_path(parsed_path)
        if folder is not None:
            target_folder = _get_dir_from_path(folder)
            if not _check_dir_exists(target_folder):
                raise ValueError(f"Target folder does not exist: {target_folder}")

        # Handle extension
        try:
            ext = _get_ext_from_path(parsed_path)
            if change_ext is not None:
                ext = change_ext.lstrip(".").lower()
        except (RuntimeError, ValueError) as e:
            raise ValueError(f"Failed to process extension: {str(e)}") from None

        # Get base filename without extension
        filename = _get_filename_from_path(parsed_path, with_ext=False)

        # Add optional components
        uuid_str = f"_{uuid4().hex}" if add_uuid else ""
        timestamp_str = f"_{utils_base._get_time_as_str()}" if add_timestamp else ""

        # Construct new filename
        augmented_filename = f"{prefix}{filename}{uuid_str}{timestamp_str}{suffix}.{ext}"

        # Join with target folder and normalize
        augmented_path = os.path.join(target_folder, augmented_filename)
        unix_path = _get_unix_path(augmented_path)

        return unix_path

    except Exception as e:
        if isinstance(e, (TypeError, ValueError, RuntimeError)):
            raise
        raise RuntimeError(f"Failed to augment path: {str(e)}") from None


def _get_augmented_path_list(
    path_list: List[str],
    prefix: str = "",
    suffix: str = "",
    change_ext: Optional[str] = None,
    folder: Optional[str] = None,
    add_uuid: bool = False,
    add_timestamp: bool = False,
) -> List[str]:
    """Augments a list of paths with prefix, suffix, and optional components.
    Format for each path: {prefix}{filename}{uuid}{timestamp}{suffix}.{ext}

    Parameters
    ----------
    path_list: List[str]
        The list of paths to augment.
    prefix: str
        The prefix to add to each path. Default: "".
    suffix: str
        The suffix to add to each path. Default: "".
    change_ext: Optional[str]
        The extension to change the files to. Default: None.
    folder: Optional[str]
        The folder to save files in. Can be /vsimem/ for memory. Default: None.
    add_uuid: bool
        If True, add a uuid to each path. Default: False.
    add_timestamp: bool
        If True, add a timestamp to each path. Default: False.

    Returns
    -------
    List[str]
        The list of augmented paths in unix style.

    Raises
    ------
    TypeError
        If inputs are not of correct type
    ValueError
        If path_list is empty or contains invalid paths
    RuntimeError
        If path augmentation fails
    """
    # Type checking
    if not isinstance(path_list, list):
        raise TypeError("path_list must be a list")
    if not isinstance(prefix, str):
        raise TypeError("prefix must be a string")
    if not isinstance(suffix, str):
        raise TypeError("suffix must be a string")
    if not isinstance(add_uuid, bool):
        raise TypeError("add_uuid must be a bool")
    if not isinstance(add_timestamp, bool):
        raise TypeError("add_timestamp must be a bool")
    if change_ext is not None and not isinstance(change_ext, str):
        raise TypeError("change_ext must be None or string")
    if folder is not None and not isinstance(folder, str):
        raise TypeError("folder must be None or string")

    # Value checking
    if not path_list:
        raise ValueError("path_list cannot be empty")
    if folder is not None and not folder.strip():
        raise ValueError("folder cannot be empty")
    if change_ext is not None and not change_ext.strip():
        raise ValueError("change_ext cannot be empty")

    try:
        augmented_paths: List[str] = []

        for path in path_list:
            if path is None:
                raise TypeError("Paths in path_list cannot be None")
            if not isinstance(path, str):
                raise TypeError("All paths must be strings")
            if not path.strip():
                raise ValueError("Paths cannot be empty strings")

            try:
                augmented_path = _get_augmented_path(
                    path,
                    prefix=prefix,
                    suffix=suffix,
                    change_ext=change_ext,
                    folder=folder,
                    add_uuid=add_uuid,
                    add_timestamp=add_timestamp,
                )
                augmented_paths.append(augmented_path)
            except (TypeError, ValueError, RuntimeError) as e:
                raise RuntimeError(f"Failed to augment path {path}: {str(e)}") from None

        if not augmented_paths:  # Defensive check
            raise RuntimeError("No paths were successfully augmented")

        return augmented_paths

    except Exception as e:
        if isinstance(e, (TypeError, ValueError, RuntimeError)):
            raise
        raise RuntimeError(f"Failed to process path list: {str(e)}") from None


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
    name: Union[str, gdal.Dataset, ogr.DataSource]
        The name or dataset to base the filename on. Default: "temp".
    ext: Optional[str]
        The extension of the file. If None, tries to derive from name. Default: None.
    prefix: str
        The prefix to add to the path. Default: "".
    suffix: str
        The suffix to add to the path. Default: "".
    add_uuid: bool
        If True, add a uuid to the path. Default: False.
    add_timestamp: bool
        If True, add a timestamp (YYYYMMDD_HHMMSS). Default: False.

    Returns
    -------
    str
        The temporary filepath in unix style (e.g. /vsimem/temp_20210101_000000_123456789.tif)

    Raises
    ------
    TypeError
        If inputs are not of correct type
    ValueError
        If inputs are invalid or empty
    RuntimeError
        If path creation fails
    """
    # Type checking
    if not isinstance(name, (str, gdal.Dataset, ogr.DataSource)):
        raise TypeError("name must be a string or GDAL/OGR dataset")
    if not isinstance(prefix, str):
        raise TypeError("prefix must be a string")
    if not isinstance(suffix, str):
        raise TypeError("suffix must be a string")
    if not isinstance(add_uuid, bool):
        raise TypeError("add_uuid must be a bool")
    if not isinstance(add_timestamp, bool):
        raise TypeError("add_timestamp must be a bool")
    if ext is not None and not isinstance(ext, str):
        raise TypeError("ext must be None or string")

    try:
        # Extract base name from input
        if isinstance(name, (gdal.Dataset, ogr.DataSource)):
            path = name.GetDescription()
            if not path:
                raise ValueError("Dataset has no description/path")
            base_name = _get_filename_from_path(path, with_ext=False)
        else:
            if not name.strip():
                raise ValueError("name cannot be empty")
            base_name = _get_filename_from_path(name, with_ext=False)

        # Handle extension
        if ext is None:
            try:
                if isinstance(name, (str, gdal.Dataset, ogr.DataSource)):
                    ext = _get_ext_from_path(name if isinstance(name, str) else name.GetDescription())
                else:
                    ext = "tif"  # Default extension
            except (RuntimeError, ValueError):
                ext = "tif"  # Fallback extension
        else:
            ext = ext.lstrip(".").lower()

        if not utils_gdal._check_is_valid_ext(ext):
            raise ValueError(f"Invalid extension: {ext}")

        # Add optional components
        uuid_str = f"_{uuid4().hex}" if add_uuid else ""
        timestamp_str = f"_{utils_base._get_time_as_str()}" if add_timestamp else ""

        # Construct filename
        filename = f"{prefix}{base_name}{uuid_str}{timestamp_str}{suffix}.{ext}"
        filepath = _get_unix_path(os.path.join("/vsimem/", filename))

        # Handle filename collisions
        if _check_file_exists(filepath):
            counter = 1
            while _check_file_exists(filepath):
                new_filename = f"{prefix}{base_name}{uuid_str}{timestamp_str}{suffix}_{counter}.{ext}"
                filepath = _get_unix_path(os.path.join("/vsimem/", new_filename))
                counter += 1

        return filepath

    except Exception as e:
        if isinstance(e, (TypeError, ValueError)):
            raise
        raise RuntimeError(f"Failed to create temporary filepath: {str(e)}") from None
