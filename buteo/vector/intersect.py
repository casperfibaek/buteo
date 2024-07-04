"""### Calculate intersections ###

Calculate and tests the intersections between geometries.
"""

# Standard library
import sys; sys.path.append("../../")
from typing import Union, List, Optional

# External
from osgeo import ogr, gdal

# Internal
from buteo.utils import (
    utils_io,
    utils_base,
    utils_gdal,
    utils_path,
)
from buteo.vector import core_vector
from buteo.vector.metadata import _vector_to_metadata
from buteo.vector.reproject import _vector_reproject
from buteo.vector.merge import vector_merge_layers


def _vector_intersect(
    vector: Union[str, ogr.DataSource],
    clip_geom: Union[str, ogr.Geometry],
    out_path: Optional[str] = None,
    process_layer: int = 0,
    process_layer_clip: int = 0,
    add_index: bool = True,
    overwrite: bool = True,
    return_bool: bool = False,
) -> Union[str, bool]:
    """Internal."""
    assert isinstance(vector, ogr.DataSource), f"Invalid input vector: {vector}"
    assert utils_gdal._check_is_vector(vector), f"Invalid input vector: {vector}"

    if out_path is None:
        out_path = utils_path._get_temp_filepath(vector, suffix="_intersect")

    assert utils_path._check_is_valid_output_filepath(out_path, overwrite=overwrite), "Invalid output path."

    match_projection = _vector_reproject(clip_geom, vector)
    geometry_to_clip = core_vector._vector_open(match_projection)

    merged = core_vector._vector_open(vector_merge_layers([vector, match_projection]))

    if add_index:
        core_vector.vector_add_index(merged)

    vector_metadata = _vector_to_metadata(vector)
    vector_layername = vector_metadata["layers"][process_layer]["layer_name"]
    vector_geom_col = vector_metadata["layers"][process_layer]["column_geom"]

    clip_geom_metadata = _vector_to_metadata(geometry_to_clip)
    clip_geom_layername = clip_geom_metadata["layers"][process_layer_clip]["layer_name"]
    clip_geom_col = clip_geom_metadata["layers"][process_layer_clip]["column_geom"]

    if return_bool:
        sql = f"SELECT A.* FROM '{vector_layername}' A, '{clip_geom_layername}' B WHERE ST_INTERSECTS(A.{vector_geom_col}, B.{clip_geom_col});"
    else:
        sql = f"SELECT A.* FROM '{vector_layername}' A, '{clip_geom_layername}' B WHERE ST_INTERSECTS(A.{vector_geom_col}, B.{clip_geom_col});"

    result = merged.ExecuteSQL(sql, dialect="SQLITE")

    if return_bool:
        if result.GetFeatureCount() == 0:
            return False
        else:
            return True

    driver = ogr.GetDriverByName(utils_gdal._get_vector_driver_name_from_path(out_path))
    destination = driver.CreateDataSource(out_path)
    destination.CopyLayer(result, vector_layername, ["OVERWRITE=YES"])

    if destination is None:
        raise RuntimeError("Error while running intersect.")

    destination.FlushCache()

    return out_path


def vector_intersect(
    vector: Union[str, ogr.DataSource, List[Union[str, ogr.DataSource]]],
    clip_geom: Union[str, ogr.Geometry, List[Union[str, ogr.Geometry]]],
    out_path: Optional[str] = None,
    process_layer: int = 0,
    process_layer_clip: int = 0,
    add_index: bool = True,
    overwrite: bool = True,
    prefix: str ="",
    suffix: str ="",
    add_uuid: bool = False,
    allow_lists: bool = True,
) -> Union[str, List[str]]:
    """Clips a vector to a geometry.

    Parameters
    ----------
    vector : Union[str, ogr.DataSource, List[Union[str, ogr.DataSource]]]
        The vector(s) to intersect.

    clip_geom : Union[str, ogr.Geometry, List[Union[str, ogr.Geometry]]]
        The geometry to intersect the vector(s) with.

    out_path : Optional[str], optional
        The path(s) to save the clipped vector(s) to. Default: None

    process_layer : int, optional
        The layer to process in the vector(s). Default: 0

    process_layer_clip : int, optional
        The layer to process in the clip geometry. Default: 0

    add_index : bool, optional
        Add a geospatial index to the vector(s). Default: True

    overwrite : bool, optional
        Overwrite the output vector(s) if they already exist. Default: True

    prefix : str, optional
        A prefix to add to the output vector(s). Default: ""

    suffix : str, optional
        A suffix to add to the output vector(s). Default: ""

    add_uuid : bool, optional
        Add a uuid to the output vector(s). Default: False

    allow_lists : bool, optional
        Allow lists as input. Default: True

    Returns
    -------
    Union[str, List[str]]
        The path(s) to the intersected vector(s).
    """
    utils_base._type_check(vector, [ogr.DataSource, str, list], "vector")
    utils_base._type_check(clip_geom, [ogr.DataSource, gdal.Dataset, str, list, tuple], "clip_geom")
    utils_base._type_check(out_path, [str], "out_path", allow_none=True)
    utils_base._type_check(process_layer, [int], "process_layer")
    utils_base._type_check(process_layer_clip, [int], "process_layer_clip")
    utils_base._type_check(add_index, [bool], "add_index")
    utils_base._type_check(overwrite, [bool], "overwrite")
    utils_base._type_check(prefix, [str], "prefix")
    utils_base._type_check(suffix, [str], "suffix")
    utils_base._type_check(add_uuid, [bool], "add_uuid")
    utils_base._type_check(allow_lists, [bool], "allow_lists")

    if not allow_lists and isinstance(vector, list):
        raise ValueError("Lists are not allowed when allow_lists is False.")

    vector_list = utils_base._get_variable_as_list(vector)

    assert utils_gdal._check_is_vector_list(vector_list), f"Invalid input vector: {vector_list}"

    path_list = utils_io._get_output_paths(
        vector_list,
        out_path,
        prefix=prefix,
        suffix=suffix,
        add_uuid=add_uuid,
    )

    assert utils_path._check_is_valid_output_filepath(path_list, overwrite=overwrite), "Invalid output path generated."

    output = []
    for index, in_vector in enumerate(vector_list):
        output.append(
            _vector_intersect(
                in_vector,
                clip_geom,
                out_path=path_list[index],
                process_layer=process_layer,
                process_layer_clip=process_layer_clip,
                add_index=add_index,
                overwrite=overwrite,
                return_bool=False,
            )
        )

    if isinstance(vector, list):
        return output

    return output[0]
