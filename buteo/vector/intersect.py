"""
Calculate intersections

TODO:
    - Improve documentation
    - Remove internal step
    - Add checks: Do geometries intersect?
"""

import sys; sys.path.append("../../") # Path: buteo/vector/intersect.py

from osgeo import ogr, gdal

from buteo.utils.core import type_check
from buteo.utils.gdal_utils import path_to_driver_vector
from buteo.vector.io import (
    open_vector,
    ready_io_vector,
    _vector_to_metadata,
    vector_add_index,
)
from buteo.vector.reproject import _reproject_vector
from buteo.vector.merge import merge_vectors


def _intersect_vector(
    vector,
    clip_geom,
    out_path=None,
    *,
    to_extent=False,
    process_layer=0,
    process_layer_clip=0,
    add_index=True,
    preserve_fid=True,
    overwrite=True,
    return_bool=False,
):
    """Clips a vector to a geometry.

    Returns:
        A clipped ogr.Datasource or the path to one.
    """
    type_check(vector, [ogr.DataSource, str, list], "vector")
    type_check(clip_geom, [ogr.DataSource, gdal.Dataset, str, list, tuple], "clip_geom")
    type_check(out_path, [str], "out_path", allow_none=True)
    type_check(to_extent, [bool], "to_extent")
    type_check(process_layer, [int], "process_layer")
    type_check(process_layer_clip, [int], "process_layer_clip")
    type_check(add_index, [bool], "add_index")
    type_check(preserve_fid, [bool], "preserve_fid")
    type_check(overwrite, [bool], "overwrite")

    _vector_list, path_list = ready_io_vector(vector, out_path, overwrite=overwrite)
    out_name = path_list[0]

    match_projection = _reproject_vector(clip_geom, vector)
    geometry_to_clip = open_vector(match_projection)

    merged = open_vector(merge_vectors([vector, match_projection]))

    if add_index:
        vector_add_index(merged)

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

    driver = ogr.GetDriverByName(path_to_driver_vector(out_name))
    destination: ogr.DataSource = driver.CreateDataSource(out_name)
    destination.CopyLayer(result, vector_layername, ["OVERWRITE=YES"])

    if destination is None:
        raise Exception("Error while running intersect.")

    destination.FlushCache()

    return out_name


def intersect_vector(
    vector,
    clip_geom,
    out_path=None,
    *,
    to_extent=False,
    process_layer=0,
    process_layer_clip=0,
    add_index=True,
    preserve_fid=True,
    overwrite=True,
):
    """Clips a vector to a geometry."""
    type_check(vector, [ogr.DataSource, str, list], "vector")
    type_check(clip_geom, [ogr.DataSource, gdal.Dataset, str, list, tuple], "clip_geom")
    type_check(out_path, [str], "out_path", allow_none=True)
    type_check(to_extent, [bool], "to_extent")
    type_check(process_layer, [int], "process_layer")
    type_check(process_layer_clip, [int], "process_layer_clip")
    type_check(add_index, [bool], "add_index")
    type_check(preserve_fid, [bool], "preserve_fid")
    type_check(overwrite, [bool], "overwrite")

    vector_list, path_list = ready_io_vector(vector, out_path, overwrite=overwrite)

    output = []
    for index, in_vector in enumerate(vector_list):
        output.append(
            _intersect_vector(
                in_vector,
                clip_geom,
                out_path=path_list[index],
                to_extent=to_extent,
                process_layer=process_layer,
                process_layer_clip=process_layer_clip,
                add_index=add_index,
                preserve_fid=preserve_fid,
                overwrite=overwrite,
            )
        )

    if isinstance(vector, list):
        return output

    return output[0]
