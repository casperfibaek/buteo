"""
### Clip vectors to other geometries ###

Clip vector files with other geometries. Can come from rasters or vectors.
"""

# Standard library
import sys; sys.path.append("../../")

# External
from osgeo import ogr

# Internal
from buteo.utils import gdal_utils, core_utils
from buteo.vector import core_vector


def _buffer_vector(vector, distance, out_path=None):
    """ Internal. """
    if out_path is None:
        out_path = gdal_utils.create_memory_path(
            gdal_utils.get_path_from_dataset(vector),
            prefix="",
            suffix="_buffer",
            add_uuid=True,
        )

    assert core_utils.is_valid_output_path(out_path), "Invalid output path"

    read = core_vector.open_vector(vector)

    driver = ogr.GetDriverByName(gdal_utils.path_to_driver_vector(out_path))
    destination = driver.CreateDataSource(out_path)

    vector_metadata = core_vector._vector_to_metadata(read)

    for layer_idx in range(0, len(vector_metadata["layers"])):
        layer_name = vector_metadata["layers"][layer_idx]["layer_name"]
        vector_layer = read.GetLayer(layer_name)
        destination.CopyLayer(vector_layer, layer_name, ["OVERWRITE=YES"])

        field_names = vector_metadata["layers"][layer_idx]["field_names"]
        if isinstance(distance, str):
            if distance not in field_names:
                raise ValueError(f"Attribute to buffer by not in vector field names: {distance} not in {field_names}")

            sql = f"update {layer_name} set geom=ST_Buffer(geom, {layer_name}.{distance})"
        else:
            sql = f"update {layer_name} set geom=ST_Buffer(geom, {distance})"

        destination.ExecuteSQL(sql, dialect="SQLITE")

    if destination is None:
        raise Exception("Error while running intersect.")

    destination.FlushCache()

    return out_path


def buffer_vector(
    vector,
    distance,
    out_path=None,
    *,
    prefix="",
    suffix="",
    add_uuid=False,
    allow_lists=True,
    overwrite=True,
):
    """
    Buffers a vector with a fixed distance or an attribute.

    ## Args:
    `vector` (_str_/_ogr.DataSource_/_list_): Vector(s) to buffer. </br>
    `distance` (_int_/_float_/_str_): The distance to buffer with. If string, uses the attribute of that name. </br>

    ## Kwargs:
    `out_path` (_str_/_None_): Output path. If None, memory vectors are created. (Default: **None**) </br>
    `prefix` (_str_): Prefix to add to the output path. (Default: **""**) </br>
    `suffix` (_str_): Suffix to add to the output path. (Default: **""**) </br>
    `add_uuid` (_bool_): Add UUID to the output path. (Default: **False**) </br>
    `allow_lists` (_bool_): Allow lists of vectors as input. (Default: **True**) </br>
    `overwrite` (_bool_): Overwrite output if it already exists. (Default: **True**) </br>

    ## Returns:
    (_str_/_list_): Output path(s) of clipped vector(s).
    """
    core_utils.type_check(vector, [str, ogr.DataSource, [str, ogr.DataSource]], "vector")
    core_utils.type_check(distance, [int, float, str], "distance")
    core_utils.type_check(out_path, [str, None], "out_path")
    core_utils.type_check(prefix, [str], "prefix")
    core_utils.type_check(suffix, [str], "suffix")
    core_utils.type_check(add_uuid, [bool], "add_uuid")
    core_utils.type_check(allow_lists, [bool], "allow_lists")

    if not allow_lists and isinstance(vector, (list, tuple)):
        raise ValueError("Lists are not allowed for vector.")

    vector_list = core_utils.ensure_list(vector)

    assert gdal_utils.is_vector_list(vector_list), f"Invalid vector in list: {vector_list}"

    path_list = gdal_utils.create_output_path_list(
        vector_list,
        out_path=out_path,
        prefix=prefix,
        suffix=suffix,
        add_uuid=add_uuid,
        overwrite=overwrite,
    )

    output = []
    for index, in_vector in enumerate(vector_list):
        output.append(
            _buffer_vector(
                in_vector,
                distance,
                out_path=path_list[index],
            )
        )

    if isinstance(vector, list):
        return output

    return output[0]
