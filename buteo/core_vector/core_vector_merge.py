"""### Merge vectors. ###

Merges vectors into a single vector file.
"""

# Standard library
from typing import Union, List, Optional

# External
from osgeo import ogr
from osgeo_utils.ogrmerge import ogrmerge

# Internal
from buteo.utils import (
    utils_base,
    utils_gdal,
    utils_path,
    utils_projection,
)



def vector_merge_features(
    vectors: Union[str, ogr.DataSource, List[Union[str, ogr.DataSource]]],
    out_path: Optional[str] = None,
    projection: Optional[str] = None,
    single_layer: bool = True,
    overwrite: bool = True,
    skip_failures: bool = True,
) -> str:
    """Merge vectors to a single geopackage.

    Parameters
    ----------
    vectors : Union[str, ogr.DataSource, List[Union[str, ogr.DataSource]]]
        The input vectors.
    out_path : Optional[str], optional
        The output path, default: None.
    projection : Optional[str], optional
        The projection of the output vector, default: None.
        If None, the projection of the first input vector is used.
    single_layer : bool, optional
        If True, all layers are merged into a single layer, default: True.
    overwrite : bool, optional
        If True, the output file is overwritten if it exists, default: True.
    skip_failures : bool, optional
        If True, failures are skipped, default: True.

    Returns
    -------
    str
        The output path.

    Raises
    ------
    ValueError
        If input vectors are invalid or empty.
    RuntimeError
        If merge operation fails.
    """
    # Input validation
    utils_base._type_check(vectors, [[str, ogr.DataSource]], "vector")
    utils_base._type_check(out_path, [str, [str], None], "out_path")
    utils_base._type_check(projection, [str, None], "projection")
    utils_base._type_check(single_layer, [bool], "single_layer")
    utils_base._type_check(overwrite, [bool], "overwrite")
    utils_base._type_check(skip_failures, [bool], "skip_failures")

    vector_list = utils_base._get_variable_as_list(vectors)

    if not vector_list:
        raise ValueError("No input vectors provided")

    if not utils_gdal._check_is_vector_list(vector_list):
        raise ValueError("Invalid input vector list")

    # Check if all input vectors exist and are readable
    for vec in vector_list:
        if isinstance(vec, str) and not utils_path._check_file_exists(vec):
            raise ValueError(f"Input vector does not exist: {vec}")

    # Setup output path
    if out_path is None:
        out_path = utils_path._get_temp_filepath(vector_list[0], suffix="_merged")
    elif utils_path._check_file_exists(out_path) and not overwrite:
        raise ValueError(f"Output file already exists: {out_path}")

    # Get driver
    driver_name = utils_gdal._get_driver_name_from_path(out_path)
    if not driver_name:
        raise ValueError(f"Could not determine driver for output path: {out_path}")

    # Setup projection
    try:
        if projection is None:
            target_projection = utils_projection._get_projection_from_vector(vector_list[0]).ExportToWkt()
        else:
            target_projection = utils_projection.parse_projection_wkt(projection)
    except Exception as e:
        raise ValueError(f"Failed to process projection: {str(e)}") from e

    # Perform merge
    try:
        success = ogrmerge(
            vector_list,
            out_path,
            single_layer=single_layer,
            driver_name=driver_name,
            t_srs=target_projection,
            overwrite_ds=overwrite,
            skip_failures=skip_failures,
            src_layer_field_name="og_name",
            src_layer_field_content="{LAYER_NAME}"
        )

        if success is None:
            raise RuntimeError("Vector merge operation failed")

        # Verify output file was created
        if not utils_path._check_file_exists(out_path):
            raise RuntimeError("Output file was not created")

        return out_path

    except Exception as e:
        if utils_path._check_file_exists(out_path):
            try:
                utils_gdal._delete_raster_or_vector(out_path)
            except (OSError, PermissionError):
                pass
        raise RuntimeError(f"Vector merge operation failed: {str(e)}") from e
