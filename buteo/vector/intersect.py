"""
### Calculate intersections ###

Calculate and tests the intersections between geometries.

TODO:
    * Add boolean checks for intersections.
"""

# Standard library
import sys; sys.path.append("../../")

# External
from osgeo import ogr, gdal

# Internal
from buteo.utils import core_utils, gdal_utils
from buteo.vector import core_vector
from buteo.vector.reproject import _reproject_vector
from buteo.vector.merge import merge_vectors



def _intersect_vector(
    vector,
    clip_geom,
    out_path=None,
    *,
    process_layer=0,
    process_layer_clip=0,
    add_index=True,
    overwrite=True,
    return_bool=False,
):
    """ Internal. """
    assert isinstance(vector, ogr.DataSource), f"Invalid input vector: {vector}"
    assert gdal_utils.is_vector(vector), f"Invalid input vector: {vector}"

    if out_path is None:
        out_path = gdal_utils.create_memory_path(
            gdal_utils.get_path_from_dataset(vector),
            add_uuid=True,
        )

    assert core_utils.is_valid_output_path(out_path, overwrite=overwrite), "Invalid output path."

    match_projection = _reproject_vector(clip_geom, vector)
    geometry_to_clip = core_vector._open_vector(match_projection)

    merged = core_vector._open_vector(merge_vectors([vector, match_projection]))

    if add_index:
        core_vector.vector_add_index(merged)

    vector_metadata = core_vector._vector_to_metadata(vector)
    vector_layername = vector_metadata["layers"][process_layer]["layer_name"]
    vector_geom_col = vector_metadata["layers"][process_layer]["column_geom"]

    clip_geom_metadata = core_vector._vector_to_metadata(geometry_to_clip)
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

    driver = ogr.GetDriverByName(gdal_utils.path_to_driver_vector(out_path))
    destination = driver.CreateDataSource(out_path)
    destination.CopyLayer(result, vector_layername, ["OVERWRITE=YES"])

    if destination is None:
        raise Exception("Error while running intersect.")

    destination.FlushCache()

    return out_path


def intersect_vector(
    vector,
    clip_geom,
    out_path=None,
    *,
    process_layer=0,
    process_layer_clip=0,
    add_index=True,
    overwrite=True,
    prefix="",
    suffix="",
    add_uuid=False,
    allow_lists=True,
):
    """
    Clips a vector to a geometry.

    ## Args:
    `vector` (_str_/_ogr.DataSource_/_list_): The vector(s) to intersect. </br>
    `clip_geom` (_str_/_ogr.Geometry_): The geometry to intersect the vector(s) with. </br>

    ## Kwargs:
    `out_path` (_str_/_list_/_None_): The path(s) to save the clipped vector(s) to. (Default: **None**) </br>
    `process_layer` (_int_): The layer to process in the vector(s). (Default: **0**) </br>
    `process_layer_clip` (_int_): The layer to process in the clip geometry. (Default: **0**) </br>
    `add_index` (_bool_): Add a geospatial index to the vector(s). (Default: **True**) </br>
    `overwrite` (_bool_): Overwrite the output vector(s) if they already exist. (Default: **True**) </br>
    `prefix` (_str_): A prefix to add to the output vector(s). (Default: **""**) </br>
    `suffix` (_str_): A suffix to add to the output vector(s). (Default: **""**) </br>
    `add_uuid` (_bool_): Add a UUID to the output vector(s). (Default: **False**) </br>
    `allow_lists` (_bool_): Allow the input to be a list of vectors. (Default: **True**) </br>

    ## Returns:
    (_str_/_list_): The path(s) to the clipped vector(s).
    """
    core_utils.type_check(vector, [ogr.DataSource, str, list], "vector")
    core_utils.type_check(clip_geom, [ogr.DataSource, gdal.Dataset, str, list, tuple], "clip_geom")
    core_utils.type_check(out_path, [str], "out_path", allow_none=True)
    core_utils.type_check(process_layer, [int], "process_layer")
    core_utils.type_check(process_layer_clip, [int], "process_layer_clip")
    core_utils.type_check(add_index, [bool], "add_index")
    core_utils.type_check(overwrite, [bool], "overwrite")
    core_utils.type_check(prefix, [str], "prefix")
    core_utils.type_check(suffix, [str], "suffix")
    core_utils.type_check(add_uuid, [bool], "add_uuid")
    core_utils.type_check(allow_lists, [bool], "allow_lists")

    if not allow_lists and isinstance(vector, list):
        raise ValueError("Lists are not allowed when allow_lists is False.")

    vector_list = core_utils.ensure_list(vector)

    assert gdal_utils.is_vector_list(vector_list), f"Invalid input vector: {vector_list}"

    path_list = gdal_utils.create_output_path_list(
        vector_list,
        out_path=out_path,
        prefix=prefix,
        suffix=suffix,
        add_uuid=add_uuid,
    )

    assert core_utils.is_valid_output_paths(path_list, overwrite=overwrite), "Invalid output path generated."

    output = []
    for index, in_vector in enumerate(vector_list):
        output.append(
            _intersect_vector(
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
