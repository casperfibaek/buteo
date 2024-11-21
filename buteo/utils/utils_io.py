"""### Utility functions read and write buteo files. ###"""

# Standard Library
from typing import Optional, Union, List

# External
from osgeo import gdal, ogr

# Internal
from buteo.utils import utils_path, utils_base, utils_gdal



def _get_input_paths(
    inputs: Union[gdal.Dataset, ogr.DataSource, str, List[Union[gdal.Dataset, ogr.DataSource, str]], None],
    input_type: str = "mixed",
) -> List[str]:
    """Parses the input data to a list of paths.

    Parameters
    ----------
    inputs : Union[gdal.Dataset, ogr.DataSource, str, List[Union[gdal.Dataset, ogr.DataSource, str]], None]
        The input data to parse. If None, returns empty list.

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
    TypeError
        If inputs are of invalid type.
    """
    if inputs is None:
        return []

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
        if not inputs:
            return []

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
    overwrite: bool = True,
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
    overwrite : bool, optional
        Allow overwriting existing files, by default True

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

    if not all(isinstance(x, bool) for x in [in_place, add_uuid, add_timestamp, overwrite]):
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
