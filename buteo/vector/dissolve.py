"""
### Dissolve vector geometries. ###

Dissolve vectors by attributes or geometry.
"""

# Standard library
import sys; sys.path.append("../../")

# External
from osgeo import ogr

# Internal
from buteo.utils import core_utils, gdal_utils
from buteo.vector import core_vector



def _dissolve_vector(
    vector,
    attribute=None,
    out_path=None,
    *,
    overwrite=True,
    add_index=True,
    process_layer=-1,
):
    """ Internal. """
    assert isinstance(vector, ogr.DataSource), "Invalid input vector"

    vector_list = core_utils.ensure_list(vector)

    if out_path is None:
        out_path = gdal_utils.create_memory_path(
            gdal_utils.get_path_from_dataset(vector),
            prefix="",
            suffix="_dissolve",
            add_uuid=True,
        )

    assert core_utils.is_valid_output_path(out_path, overwrite=overwrite), "Invalid output path"

    out_format = gdal_utils.path_to_driver_vector(out_path)

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

    core_utils.remove_if_required(out_path, overwrite=overwrite)

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


def dissolve_vector(
    vector,
    attribute=None,
    out_path=None,
    *,
    add_index=True,
    process_layer=-1,
    prefix="",
    suffix="",
    add_uuid=False,
    overwrite=True,
    allow_lists=True,
):
    """Clips a vector to a geometry.
    Args:
        vector (list of vectors/path/vector): The vectors(s) to clip.

        clip_geom (list of geom/path/vector/rasters): The geometry to use
        for the clipping

    **kwargs:


    Returns:
        A clipped ogr.Datasource or the path to one.
    """
    core_utils.type_check(vector, [ogr.DataSource, str, [str, ogr.DataSource]], "vector")
    core_utils.type_check(attribute, [str, None], "attribute")
    core_utils.type_check(out_path, [str, [str], None], "out_path")
    core_utils.type_check(add_index, [bool], "add_index")
    core_utils.type_check(process_layer, [int], "process_layer")
    core_utils.type_check(prefix, [str], "prefix")
    core_utils.type_check(suffix, [str], "suffix")
    core_utils.type_check(add_uuid, [bool], "add_uuid")
    core_utils.type_check(overwrite, [bool], "overwrite")
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
            _dissolve_vector(
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
