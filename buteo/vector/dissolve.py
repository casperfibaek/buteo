"""
### Dissolve vector geometries. ###

Dissolve vectors by attributes or geometry.
"""

# Standard library
import sys; sys.path.append("../../")
from typing import Union, Optional, List

# External
from osgeo import ogr

# Internal
from buteo.utils import (
    utils_io,
    utils_base,
    utils_gdal,
    utils_path,
)
from buteo.vector import core_vector



def _vector_dissolve(
    vector: Union[str, ogr.DataSource],
    attribute: Optional[str] = None,
    out_path: Optional[str] = None,
    overwrite: bool = True,
    add_index: bool = True,
    process_layer: int = -1,
) -> Union[str, ogr.DataSource]:
    """ Internal. """
    assert isinstance(vector, ogr.DataSource), "Invalid input vector"

    vector_list = utils_base._get_variable_as_list(vector)

    if out_path is None:
        out_path = utils_path._get_temp_filepath("dissolve.shp", suffix="_dissolve")

    assert utils_path._check_is_valid_output_filepath(out_path, overwrite=overwrite), "Invalid output path"

    out_format = utils_gdal._get_vector_driver_name_from_path(out_path)

    driver = ogr.GetDriverByName(out_format)

    ref = core_vector._open_vector(vector_list[0])
    metadata = core_vector._vector_to_metadata(ref)

    layers = []

    if process_layer == -1:
        for index in range(len(metadata["layers"])):
            layers.append(
                {
                    "name": metadata["layers"][index]["layer_name"],
                    "geom": metadata["layers"][index]["column_geom"],
                    "fields": metadata["layers"][index]["field_names"],
                }
            )
    else:
        layers.append(
            {
                "name": metadata["layers"][process_layer]["layer_name"],
                "geom": metadata["layers"][process_layer]["column_geom"],
                "fields": metadata["layers"][process_layer]["field_names"],
            }
        )

    utils_path._delete_if_required(out_path, overwrite=overwrite)

    destination = driver.CreateDataSource(out_path)

    # Check if attribute table is valid
    for index in range(len(metadata["layers"])):
        layer = layers[index]
        if attribute is not None and attribute not in layer["fields"]:
            layer_fields = layer["fields"]
            raise ValueError(
                f"Invalid attribute for layer. Layers has the following fields: {layer_fields}"
            )

        geom_col = layer["geom"]
        name = layer["name"]

        sql = None
        if attribute is None:
            sql = f"SELECT ST_Union({geom_col}) AS geom FROM {name};"
        else:
            sql = f"SELECT {attribute}, ST_Union({geom_col}) AS geom FROM {name} GROUP BY {attribute};"

        result = ref.ExecuteSQL(sql, dialect="SQLITE")
        destination.CopyLayer(result, name, ["OVERWRITE=YES"])

    if add_index:
        core_vector.vector_add_index(destination)

    destination.FlushCache()

    return out_path


def vector_dissolve(
    vector: Union[str, ogr.DataSource, List[Union[str, ogr.DataSource]]],
    attribute: Optional[str] = None,
    out_path: Optional[str] = None,
    add_index: bool = True,
    process_layer: int = -1,
    prefix: str = "",
    suffix: str = "",
    add_uuid: bool = False,
    overwrite: bool = True,
    allow_lists: bool = True,
):
    """
    Clips a vector to a geometry.

    Parameters
    ----------
    vector : Union[str, ogr.DataSource, List[Union[str, ogr.DataSource]]]
        The vector(s) to clip.

    attribute : Optional[str], optional
        The attribute to use for the dissolve, default: None
    
    out_path : Optional[str], optional
        The output path, default: None

    add_index : bool, optional
        Add a spatial index to the output, default: True

    process_layer : int, optional
        The layer to process, default: -1

    prefix : str, optional
        The prefix to add to the output path, default: ""

    suffix : str, optional
        The suffix to add to the output path, default: ""

    add_uuid : bool, optional
        Add a uuid to the output path, default: False

    overwrite : bool, optional
        Overwrite the output, default: True

    allow_lists : bool, optional
        Allow lists as input, default: True

    Returns
    -------
    Union[str, ogr.DataSource]
        The output path or ogr.DataSource
    """
    utils_base._type_check(vector, [ogr.DataSource, str, [str, ogr.DataSource]], "vector")
    utils_base._type_check(attribute, [str, None], "attribute")
    utils_base._type_check(out_path, [str, [str], None], "out_path")
    utils_base._type_check(add_index, [bool], "add_index")
    utils_base._type_check(process_layer, [int], "process_layer")
    utils_base._type_check(prefix, [str], "prefix")
    utils_base._type_check(suffix, [str], "suffix")
    utils_base._type_check(add_uuid, [bool], "add_uuid")
    utils_base._type_check(overwrite, [bool], "overwrite")
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
            _vector_dissolve(
                in_vector,
                attribute=attribute,
                out_path=path_list[index],
                overwrite=overwrite,
                add_index=add_index,
                process_layer=process_layer,
            )
        )

    if isinstance(vector, list):
        return output

    return output[0]
