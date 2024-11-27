"""### Utility functions read and write buteo files. ###"""

# Standard Library
import os
import shutil
from typing import Optional, Union, List
from warnings import warn

# External
from osgeo import gdal, ogr

# Internal
from buteo.utils import utils_path, utils_base, utils_gdal



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
    if not utils_path._check_dir_exists(folder):
        raise RuntimeError(f"folder does not exist: {folder}")

    try:
        # Handle physical filesystem
        if not utils_path._check_dir_exists_vsimem(folder):
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
                vsimem = utils_path._get_vsimem_content(folder)
                if vsimem:  # Only attempt deletion if there are files
                    for f in vsimem:
                        gdal.Unlink(f)
            except RuntimeError as e:
                warn(f"Failed to access or clear vsimem folder {folder}: {str(e)}", UserWarning)
                return False

        # Verify deletion was successful
        remaining_content = (
            os.listdir(folder) if not utils_path._check_dir_exists_vsimem(folder)
            else utils_path._get_vsimem_content(folder)
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
    if not utils_path._check_dir_exists(folder):
        raise RuntimeError(f"folder does not exist: {folder}")

    try:
        # Handle physical filesystem
        if not utils_path._check_dir_exists_vsimem(folder):
            try:
                shutil.rmtree(folder)
                return not utils_path._check_dir_exists(folder)
            except (OSError, PermissionError) as e:
                warn(f"Failed to remove folder {folder}: {str(e)}", UserWarning)
                return False

        # Handle virtual filesystem (vsimem)
        try:
            vsimem = utils_path._get_vsimem_content(folder)
            # Delete all files in the folder
            for f in vsimem:
                gdal.Unlink(f)
            # Delete the folder itself
            if folder != "/vsimem/":  # Don't delete root vsimem
                gdal.Unlink(folder)
            return not utils_path._check_dir_exists_vsimem(folder)
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
    if not utils_path._check_file_exists(file):
        raise RuntimeError(f"file does not exist: {file}")

    try:
        # Handle physical filesystem
        if not utils_path._check_file_exists_vsimem(file):
            try:
                os.remove(file)
                return not utils_path._check_file_exists(file)
            except (OSError, PermissionError) as e:
                warn(f"Failed to remove file {file}: {str(e)}", UserWarning)
                return False

        # Handle virtual filesystem (vsimem)
        try:
            gdal.Unlink(file)
            return not utils_path._check_file_exists_vsimem(file)
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
    unix_path = utils_path._get_unix_path(path)

    # Check if directory already exists (either on disk or in vsimem)
    if utils_path._check_dir_exists(unix_path):
        return unix_path

    try:
        # Handle physical filesystem
        if not utils_path._check_dir_exists_vsimem(unix_path):
            os.makedirs(unix_path, exist_ok=True)

            if not utils_path._check_dir_exists(unix_path):
                raise RuntimeError(f"Failed to create directory: {unix_path}")

        # For vsimem, we don't create the directory as it's not possible
        # without creating a file in it. We just return the path.
        return unix_path

    except (OSError, PermissionError) as e:
        raise RuntimeError(f"Failed to create directory {unix_path}: {str(e)}") from None


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
        if not overwrite or not utils_path._check_file_exists(path):
            return True

        # Handle virtual filesystem (vsimem) files
        if utils_path._check_is_valid_mem_filepath(path):
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
        if utils_path._check_file_exists(path):
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


def _get_input_paths(
    inputs: Union[gdal.Dataset, ogr.DataSource, str, List[Union[gdal.Dataset, ogr.DataSource, str]]],
    input_type: str = "mixed",
) -> List[str]:
    """Parses the input data to a list of paths.

    Parameters
    ----------
    inputs : Union[gdal.Dataset, ogr.DataSource, str, List[Union[gdal.Dataset, ogr.DataSource, str]]]
        The input data to parse.

    input_type : str, optional
        The input type. Can be "raster", "vector" or "mixed". Default: "mixed".

    Returns
    -------
    List[str]
        The list of paths.

    Raises
    ------
    ValueError
        If input_type is invalid or inputs cannot be located.
        If inputs is None or Empty
    TypeError
        If inputs are of invalid type.
    """
    if inputs is None:
        raise ValueError("inputs cannot be None")

    if isinstance(inputs, list) and len(inputs) == 0:
        raise ValueError("inputs cannot be empty")

    valid_types = ["raster", "vector", "mixed"]
    if input_type not in valid_types:
        raise ValueError(f"input_type must be one of {valid_types}")

    if not isinstance(inputs, (gdal.Dataset, ogr.DataSource, str, list)):
        raise TypeError("inputs must be gdal.Dataset, ogr.DataSource, str, list, or None")

    parsed_inputs: List[str] = []

    if isinstance(inputs, str):
        if utils_path._check_is_path_glob(inputs):
            parsed_inputs = utils_path._get_paths_from_glob(inputs)
        else:
            parsed_inputs = utils_base._get_variable_as_list(inputs)

    elif isinstance(inputs, (gdal.Dataset, ogr.DataSource)):
        desc = inputs.GetDescription()
        parsed_inputs = [desc] if desc else []

    elif isinstance(inputs, list):
        if not all(isinstance(val, (gdal.Dataset, ogr.DataSource, str)) for val in inputs):
            raise TypeError("All list elements must be gdal.Dataset, ogr.DataSource, or str")

        parsed_inputs = utils_gdal._get_path_from_dataset_list(inputs, allow_mixed=True)

    # Validate paths
    parsed_inputs = [utils_path._parse_path(path) for path in parsed_inputs if path]
    parsed_inputs = [path for path in parsed_inputs if utils_path._check_file_exists(path)]

    if not parsed_inputs:
        raise ValueError("No valid input paths found")

    # Type validation
    if input_type == "raster":
        if not all(utils_gdal._check_is_raster(path) for path in parsed_inputs):
            raise TypeError("Not all inputs are valid raster files")
    elif input_type == "vector":
        if not all(utils_gdal._check_is_vector(path) for path in parsed_inputs):
            raise TypeError("Not all inputs are valid vector files")
    else:  # mixed
        if not all(utils_gdal._check_is_raster_or_vector(path) for path in parsed_inputs):
            raise TypeError("Not all inputs are valid raster or vector files")

    return parsed_inputs


def _get_output_paths(
    inputs: Union[str, gdal.Dataset, ogr.DataSource, List[Union[str, gdal.Dataset, ogr.DataSource]]],
    output_path: Optional[Union[str, List[str]]] = None,
    in_place: bool = False,
    *,
    prefix: str = "",
    suffix: str = "",
    change_ext: Optional[str] = None,
    add_uuid: bool = False,
    add_timestamp: bool = False,
) -> List[str]:
    """Get the output path(s) for file(s) using the input path(s) and the output path.
    The output path can be None, in which case the created path will be in memory.
    Augmentations can be made to the output paths.

    Parameters
    ----------
    inputs : Union[str, gdal.Dataset, ogr.DataSource, List[Union[str, gdal.Dataset, ogr.DataSource]]]
        The input file(s) or dataset(s)

    output_path : Optional[Union[str, List[str]]], optional
        The output path(s), by default None

    in_place : bool, optional
        If True, outputs will be same as inputs, by default False

    prefix : str, optional
        Prefix to add to filenames, by default ""

    suffix : str, optional
        Suffix to add to filenames, by default ""

    change_ext : Optional[str], optional
        New extension for output files, by default None

    add_uuid : bool, optional
        Add unique identifier to filenames, by default False

    add_timestamp : bool, optional
        Add timestamp to filenames, by default False

    Returns
    -------
    List[str]
        List of output paths

    Raises
    ------
    ValueError
        If inputs or output paths are invalid
    TypeError
        If input types are incorrect
    """
    if inputs is None:
        raise ValueError("inputs cannot be None")

    if not isinstance(prefix, str) or not isinstance(suffix, str):
        raise TypeError("prefix and suffix must be strings")

    if not all(isinstance(x, bool) for x in [in_place, add_uuid, add_timestamp]):
        raise TypeError("boolean parameters must be bool type")

    if change_ext is not None and not isinstance(change_ext, str):
        raise TypeError("change_ext must be None or string")

    # Get validated input paths
    input_paths = _get_input_paths(inputs, input_type="mixed")
    if not input_paths:
        raise ValueError("No valid input paths found")

    # Handle in-place operations
    if in_place:
        return input_paths.copy()

    # Handle memory output
    if output_path is None:
        return [
            utils_path._get_temp_filepath(
                path,
                prefix=prefix,
                suffix=suffix,
                add_uuid=add_uuid,
                add_timestamp=add_timestamp,
                ext=change_ext,
            )
            for path in input_paths
        ]

    # Handle directory output
    if isinstance(output_path, str) and utils_path._check_is_dir(output_path):
        return [
            utils_path._get_augmented_path(
                path,
                prefix=prefix,
                suffix=suffix,
                add_uuid=add_uuid,
                add_timestamp=add_timestamp,
                change_ext=change_ext,
                folder=output_path,
            )
            for path in input_paths
        ]

    # Handle single file output
    if isinstance(output_path, str):
        if not utils_path._check_is_valid_output_filepath(output_path):
            raise ValueError(f"Invalid output path: {output_path}")
        if len(input_paths) > 1:
            raise ValueError("Single output path provided for multiple inputs")
        return [
            utils_path._get_augmented_path(
                output_path,
                prefix=prefix,
                suffix=suffix,
                add_uuid=add_uuid,
                add_timestamp=add_timestamp,
                change_ext=change_ext,
                folder=None,
            )
        ]

    # Handle multiple file outputs
    if isinstance(output_path, list):
        if not utils_path._check_is_valid_filepath_list(output_path):
            raise ValueError("Invalid output paths in list")
        if len(output_path) != len(input_paths):
            raise ValueError("Number of output paths must match number of input paths")
        return [
            utils_path._get_augmented_path(
                path,
                prefix=prefix,
                suffix=suffix,
                add_uuid=add_uuid,
                add_timestamp=add_timestamp,
                change_ext=change_ext,
                folder=None,
            )
            for path in output_path
        ]

    raise ValueError("Invalid output_path type")


def _check_overwrite_policy(
    output_paths: List[str],
    overwrite: Union[bool, List[bool]]
) -> bool:
    """Check if output files can be written based on the overwrite policy.

    Parameters
    ----------
    output_paths : List[str]
        List of output file paths.

    overwrite : Union[bool, List[bool]]
        Overwrite flag(s). Can be a single boolean or a list of booleans corresponding to each output path.

    Returns
    -------
    bool
        True if all files can be written according to the overwrite policy.

    Raises
    ------
    FileExistsError
        If overwrite is False and a file already exists at any output path.

    ValueError
        If the length of overwrite list does not match the number of output paths.

    TypeError
        If overwrite is not a boolean or a list of booleans.
    """
    utils_base._type_check(output_paths, [list], "output_paths")
    utils_base._type_check(overwrite, [bool, list], "overwrite")

    if isinstance(overwrite, bool):
        overwrite_flags = [overwrite] * len(output_paths)
    elif isinstance(overwrite, list):
        if len(overwrite) != len(output_paths):
            raise ValueError("Length of overwrite flags does not match number of output paths")
        if not all(isinstance(flag, bool) for flag in overwrite):
            raise TypeError("All overwrite flags must be booleans")
        overwrite_flags = overwrite
    else:
        raise TypeError("overwrite must be a boolean or a list of booleans")

    for path, can_overwrite in zip(output_paths, overwrite_flags):
        if can_overwrite:
            continue
        if utils_path._check_file_exists(path):
            raise FileExistsError(f"File exists and overwrite is False: {path}")

    return True
