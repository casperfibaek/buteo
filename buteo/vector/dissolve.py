"""
### Dissolve vector geometries. ###

Dissolve vectors by attributes or geometry.
"""

# Standard library
import sys; sys.path.append("../../")

# External
from osgeo import ogr

# Internal
from buteo.utils import utils_base, utils_gdal
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

    vector_list = utils_base._get_variable_as_list(vector)

    if out_path is None:
        out_path = utils_gdal.create_memory_path(
            utils_gdal._get_path_from_dataset(vector),
            prefix="",
            suffix="_dissolve",
            add_uuid=True,
        )

    assert utils_base.is_valid_output_path(out_path, overwrite=overwrite), "Invalid output path"

    out_format = utils_gdal._get_vector_driver_from_path(out_path)

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
    utils_base.type_check(vector, [ogr.DataSource, str, [str, ogr.DataSource]], "vector")
    utils_base.type_check(attribute, [str, None], "attribute")
    utils_base.type_check(out_path, [str, [str], None], "out_path")
    utils_base.type_check(add_index, [bool], "add_index")
    utils_base.type_check(process_layer, [int], "process_layer")
    utils_base.type_check(prefix, [str], "prefix")
    utils_base.type_check(suffix, [str], "suffix")
    utils_base.type_check(add_uuid, [bool], "add_uuid")
    utils_base.type_check(overwrite, [bool], "overwrite")
    utils_base.type_check(allow_lists, [bool], "allow_lists")

    if not allow_lists and isinstance(vector, list):
        raise ValueError("Lists are not allowed when allow_lists is False.")

    vector_list = utils_base._get_variable_as_list(vector)

    assert utils_gdal._check_is_vector_list(vector_list), f"Invalid input vector: {vector_list}"

    path_list = utils_gdal.create_output_path_list(
        vector_list,
        out_path=out_path,
        prefix=prefix,
        suffix=suffix,
        add_uuid=add_uuid,
    )

    assert utils_base.is_valid_output_paths(path_list, overwrite=overwrite), "Invalid output path generated."

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
