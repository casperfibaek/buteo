"""
### Utility functions read and write buteo files ###
"""

# Standard Library
import sys; sys.path.append("../../")
from typing import Optional, Union, List

# External
from osgeo import gdal, ogr

# Internal
from buteo.utils import utils_path, utils_base, utils_gdal



def _get_input_paths(
    inputs: Union[gdal.Dataset, ogr.DataSource, str, List[Union[gdal.Dataset, ogr.DataSource, str]]],
    input_type: str,
) -> List[str]:
    """
    Parses the input data to a list of paths.

    Parameters
    ----------
    input_data : Union[gdal.Dataset, ogr.DataSource, str, List[Union[gdal.Dataset, ogr.DataSource, str]]]
        The input data to parse.

    input_type : str, optional
        The input type. Can be "raster", "vector" or "mixed". Default: "mixed".

    Returns
    -------
    List[str]
        The list of paths.
    """
    assert input_type in ["raster", "vector", "mixed"], "Invalid input type."
    assert isinstance(inputs, (gdal.Dataset, ogr.DataSource, str, list)), "Invalid type for input data."

    if isinstance(inputs, str):
        if utils_path._check_is_path_glob(inputs):
            inputs = utils_path._get_paths_from_glob(inputs)
        else:
            inputs = utils_base._get_variable_as_list(inputs)

    elif isinstance(inputs, (gdal.Dataset, ogr.DataSource)):
        inputs = [inputs.GetDescription()]

    # Input is list
    elif isinstance(inputs, list):
        if len(inputs) == 0:
            raise ValueError("Input data cannot be empty.")

        if not all([isinstance(val, (gdal.Dataset, ogr.DataSource, str)) for val in inputs]):
            raise TypeError("Invalid type for input data.")

        inputs = utils_gdal._get_path_from_dataset_list(inputs, allow_mixed=True)

        if input_type == "mixed":
            if not all([utils_gdal._check_is_file_valid_ext(val) for val in inputs]):
                raise TypeError("Invalid type for input data.")
        elif input_type == "raster":
            if not all([utils_gdal._check_is_raster(val) for val in inputs]):
                raise TypeError("Invalid type for input data.")
        elif input_type == "vector":
            if not all([utils_gdal._check_is_vector(val) for val in inputs]):
                raise TypeError("Invalid type for input data.")

    else:
        raise TypeError("Invalid type for input data.")

    if not utils_path._check_is_valid_filepath_list(inputs):
        raise ValueError("Invalid input data.")

    if input_type == "raster":
        for val in inputs:
            if not utils_gdal._check_is_raster(val):
                raise TypeError("Invalid type for input data.")
    elif input_type == "vector":
        for val in inputs:
            if not utils_gdal._check_is_vector(val):
                raise TypeError("Invalid type for input data.")
    else:
        for val in inputs:
            if not utils_gdal._check_is_raster_or_vector(val):
                raise TypeError("Invalid type for input data.")

    inputs = [utils_path._get_unix_path(path) for path in inputs]

    return inputs



def _get_output_paths(
    inputs: Union[str, gdal.Dataset, ogr.DataSource, List[Union[str, gdal.Dataset, ogr.DataSource]]],
    output_path: Optional[str] = None,
    *,
    prefix: str = "",
    suffix: str = "",
    change_ext: Optional[str] = None,
    add_uuid: bool = False,
    add_timestamp: bool = False,
    overwrite: bool = True,
) -> List[str]:
    """
    Get the output path for a file using the input path and the output path.
    The output path can be None, in which case the created path will be in memory.
    Augmentations can be made to the output.

    Parameters
    ----------
    input_path: str
        The path to the input file.

    output_path: str or None. Optional.
        The path to the output file. If None, the output will be in memory.
        Default: None.

    prefix: str. Optional.
        The prefix to add to the path. Default: "".

    suffix: str. Optional.
        The suffix to add to the path. Default: "".

    change_ext: str. Optional.
        The extension to change the file to. Default: None.

    add_uuid: bool. Optional.
        If True, add a uuid the path. Default: False.

    add_timestamp: bool. Optional.
        If True, add a timestamp the path. Default: False.

    overwrite: bool. Optional.
        If True, if the output path exists and overwrite is False, an error will be thrown.
        Default: True.

    Returns
    -------
    List[str]
        The output path(s).
    """
    assert isinstance(inputs, (str, gdal.Dataset, ogr.DataSource, list)), "input_path must be a string or a list."
    assert isinstance(prefix, str), "prefix must be a string."
    assert isinstance(suffix, str), "suffix must be a string."
    assert isinstance(overwrite, bool), "overwrite must be a bool."
    assert isinstance(add_uuid, bool), "add_uuid must be a bool."
    assert change_ext is None or isinstance(change_ext, str), "change_ext must be a string."

    if output_path is not None:
        assert isinstance(output_path, (str, list)), "output_path must be a string or list."
        if isinstance(output_path, str):
            assert len(output_path) > 0, "output_path must not be empty string."
        elif isinstance(output_path, list):
            assert len(output_path) > 0, "output_path must not be empty list."
            assert all([isinstance(val, str) for val in output_path]), "output_path must be a list of strings."
            assert all([len(val) > 0 for val in output_path]), "output_path must not be a list of empty strings."

    inputs = _get_input_paths(inputs, input_type="mixed")
    outputs = []

    # Output is None - Memory is used.
    if output_path is None:
        for path in inputs:
            aug_path = utils_path._get_temp_filepath(
                path,
                prefix=prefix,
                suffix=suffix,
                add_uuid=add_uuid,
                add_timestamp=add_timestamp,
                ext=change_ext,
            )
            outputs.append(aug_path)

    # Output is a file
    elif isinstance(output_path, str) and utils_path._check_is_valid_output_filepath(output_path):
        aug_path = utils_path._get_augmented_path(
            output_path,
            prefix=prefix,
            suffix=suffix,
            add_uuid=add_uuid,
            add_timestamp=add_timestamp,
            change_ext=change_ext,
            folder=None,
        )
        outputs.append(aug_path)

    # Output is a folder
    elif isinstance(output_path, str) and utils_path._check_dir_exists(output_path):
        for path in inputs:
            aug_path = utils_path._get_augmented_path(
                path,
                prefix=prefix,
                suffix=suffix,
                add_uuid=add_uuid,
                add_timestamp=add_timestamp,
                change_ext=change_ext,
                folder=output_path,
            )
            outputs.append(aug_path)

    # Output is a list of files
    elif isinstance(output_path, list) and utils_path._check_is_valid_filepath_list(output_path):
        if len(output_path) != len(inputs):
            raise ValueError("The number of output paths must match the number of input paths.")

        for path in output_path:
            aug_path = utils_path._get_augmented_path(
                path,
                prefix=prefix,
                suffix=suffix,
                add_uuid=add_uuid,
                add_timestamp=add_timestamp,
                change_ext=change_ext,
                folder=None,
            )
            outputs.append(aug_path)

    else:
        raise ValueError("Invalid output_path.")

    return outputs
