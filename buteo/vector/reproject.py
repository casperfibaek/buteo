"""
### Reproject vectors. ###

Functions to reproject vectors. References can be both vector and raster.
"""

# Standard library
import sys; sys.path.append("../../")

# External
from osgeo import gdal, osr, ogr

# Internal
from buteo.utils import gdal_utils, core_utils


def _reproject_vector(
    vector,
    projection,
    out_path=None,
    copy_if_same=False,
    *,
    prefix="",
    suffix="",
    add_uuid=False,
):
    """ Internal. """
    assert isinstance(vector, (ogr.DataSource, str)), "Invalid vector input"
    assert gdal_utils.is_vector(vector), "Invalid vector input"

    # The input is already in the correct projection.
    if not copy_if_same:
        original_projection = gdal_utils.parse_projection(vector)

        if gdal_utils.projections_match(original_projection, projection):
            return gdal_utils.get_path_from_dataset(vector)

    in_path = gdal_utils.get_path_from_dataset(vector)

    if out_path is None:
        out_path = gdal_utils.create_memory_path(
            gdal_utils.get_path_from_dataset(vector),
            prefix=prefix,
            suffix=suffix,
            add_uuid=add_uuid,
        )

    options = []
    wkt = gdal_utils.parse_projection(projection, return_wkt=True).replace(" ", "\\")

    options.append(f'-t_srs "{wkt}"')

    success = gdal.VectorTranslate(
        out_path,
        in_path,
        format=gdal_utils.path_to_driver_vector(out_path),
        options=" ".join(options),
    )

    if success != 0:
        return out_path
    else:
        raise Exception("Error while clipping geometry.")


def reproject_vector(
    vector,
    projection,
    out_path=None,
    *,
    copy_if_same=False,
    prefix="",
    suffix="",
    add_uuid=False,
    allow_lists=True,
    overwrite=True,
):
    """Reprojects a vector given a target projection.

    Args:
        vector (_path_/_vector_): The vector to reproject.

        projection (_str_/_int_/_vector_/_raster_): The projection is infered from
        the input. The input can be: WKT proj, EPSG proj, Proj, or read from a
        vector or raster datasource either from path or in-memory.

    **kwargs:
        out_path (_path_/_None_): The destination to save to. If None then
        the output is an in-memory raster.

        copy_if_same (_bool_): Create a copy, even if the projections are the same.

        overwite (_bool_): Is it possible to overwrite the out_path if it exists.

    Returns:
        An in-memory vector. If an out_path is given, the output is a string containing
        the path to the newly created vecotr.
    """
    core_utils.type_check(vector, [str, ogr.DataSource], "vector")
    core_utils.type_check(projection, [str, int, ogr.DataSource, gdal.Dataset, osr.SpatialReference], "projection")
    core_utils.type_check(out_path, [str, [str], None], "out_path")
    core_utils.type_check(copy_if_same, [bool], "copy_if_same")
    core_utils.type_check(prefix, [str], "prefix")
    core_utils.type_check(suffix, [str], "suffix")
    core_utils.type_check(add_uuid, [bool], "add_uuid")
    core_utils.type_check(allow_lists, [bool], "allow_lists")
    core_utils.type_check(overwrite, [bool], "overwrite")

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
        overwrite=overwrite,
    )

    assert core_utils.is_valid_output_path_list(path_list, overwrite=overwrite), "Invalid output path generated."

    output = []
    for index, in_vector in enumerate(vector_list):
        output.append(
            _reproject_vector(
                in_vector,
                projection,
                out_path=path_list[index],
                prefix=prefix,
                suffix=suffix,
                add_uuid=add_uuid,
                copy_if_same=copy_if_same,
            )
        )

    if isinstance(vector, list):
        return output

    return output[0]
