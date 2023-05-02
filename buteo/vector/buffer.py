"""
### Clip vectors to other geometries ###

Clip vector files with other geometries. Can come from rasters or vectors.
"""

# Standard library
import sys; sys.path.append("../../")
from typing import Union, Optional, List

# External
from osgeo import ogr

# Internal
from buteo.utils import utils_gdal, utils_base, utils_path
from buteo.vector import core_vector


def _vector_buffer(
    vector: Union[str, ogr.DataSource],
    distance: Union[int, float],
    out_path: Optional[str] = None,
) -> str:
    """ Internal. """
    if out_path is None:
        out_path = utils_path._get_output_path(
            utils_gdal._get_path_from_dataset(vector),
            prefix="",
            suffix="_buffer",
            add_uuid=True,
        )

    assert utils_path._check_is_valid_filepath(out_path), "Invalid output path"

    read = core_vector.vector_open(vector)

    driver = ogr.GetDriverByName(utils_gdal._get_vector_driver_from_path(out_path))
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
        raise RuntimeError("Error while running intersect.")

    destination.FlushCache()

    return out_path


def vector_buffer(
    vector: Union[str, ogr.DataSource, List[Union[str, ogr.DataSource]]],
    distance: Union[int, float],
    out_path: Optional[str] = None,
    prefix: str = "",
    suffix: str = "",
    add_uuid: bool = False,
    allow_lists: bool = True,
    overwrite: bool = True,
) -> Union[str, List[str]]:
    """
    Buffers a vector with a fixed distance or an attribute.

    Parameters
    ----------
    vector : Union[str, ogr.DataSource, List[str, ogr.DataSource]]
        Vector(s) to buffer.

    distance : Union[int, float, str]
        The distance to buffer with. If string, uses the attribute of that name.

    out_path : Optional[str], optional
        Output path. If None, memory vectors are created. Default: None

    prefix : str, optional
        Prefix to add to the output path. Default: ""

    suffix : str, optional
        Suffix to add to the output path. Default: ""

    add_uuid : bool, optional
        Add UUID to the output path. Default: False

    allow_lists : bool, optional
        Allow lists of vectors as input. Default: True

    overwrite : bool, optional
        Overwrite output if it already exists. Default: True

    Returns
    -------
    Union[str, List[str]]
        Output path(s) of clipped vector(s).
    """
    utils_base.type_check(vector, [str, ogr.DataSource, List[Union[str, ogr.DataSource]]], "vector")
    utils_base.type_check(distance, [int, float, str], "distance")
    utils_base.type_check(out_path, [str, None], "out_path")
    utils_base.type_check(prefix, [str], "prefix")
    utils_base.type_check(suffix, [str], "suffix")
    utils_base.type_check(add_uuid, [bool], "add_uuid")
    utils_base.type_check(allow_lists, [bool], "allow_lists")

    if not allow_lists and isinstance(vector, (list, tuple)):
        raise ValueError("Lists are not allowed for vector.")

    vector_list = utils_base._get_variable_as_list(vector)

    assert utils_gdal._check_is_vector_list(vector_list), f"Invalid vector in list: {vector_list}"

    path_list = utils_gdal._parse_output_data(
        vector_list,
        output_data=out_path,
        prefix=prefix,
        suffix=suffix,
        add_uuid=add_uuid,
        overwrite=overwrite,
    )

    output = []
    for index, in_vector in enumerate(vector_list):
        output.append(
            _vector_buffer(
                in_vector,
                distance,
                out_path=path_list[index],
            )
        )

    if isinstance(vector, list):
        return output

    return output[0]
