"""
### Clip vectors to other geometries ###

Clip vector files with other geometries. Can come from rasters or vectors.
"""

# Standard library
import sys; sys.path.append("../../")

# External
from osgeo import ogr, gdal, osr

# Internal
from buteo.utils import utils_base, utils_gdal
from buteo.raster import core_raster
from buteo.vector import core_vector
from buteo.vector.reproject import _reproject_vector



def _clip_vector(
    vector,
    clip_geom,
    out_path=None,
    *,
    to_extent=False,
    target_projection=None,
    preserve_fid=True,
    promote_to_multi=True,
):
    """ Internal. """

    input_path = utils_gdal._get_path_from_dataset(vector)

    if out_path is None:
        out_path = utils_gdal.create_memory_path(
            input_path,
            prefix="",
            suffix="_clip",
            add_uuid=True,
        )

    assert utils_base.is_valid_output_path(out_path), "Invalid output path"

    options = []

    clear_memory = False
    geometry_to_clip = None
    if utils_gdal._check_is_vector(clip_geom):
        if to_extent:
            extent = core_vector._vector_to_metadata(clip_geom)["get_bbox_vector"]() # pylint: disable=not-callable
            geometry_to_clip = extent
            clear_memory = True
        else:
            geometry_to_clip = core_vector._open_vector(clip_geom)
    elif utils_gdal._check_is_raster(clip_geom):
        extent = core_raster._raster_to_metadata(clip_geom)["get_bbox_vector"]() # pylint: disable=not-callable
        geometry_to_clip = extent
        clear_memory = True
    else:
        raise ValueError(f"Invalid input in clip_geom, unable to parse: {clip_geom}")

    clip_vector_path = utils_gdal._get_path_from_dataset(geometry_to_clip)
    clip_vector_reprojected = _reproject_vector(clip_vector_path, vector)

    if clear_memory:
        utils_gdal._delete_dataset_if_in_memory(clip_vector_path)

    x_min, x_max, y_min, y_max = core_vector._vector_to_metadata(clip_vector_reprojected)["extent"]

    options.append(f"-spat {x_min} {y_min} {x_max} {y_max}")

    options.append(f'-clipsrc "{clip_vector_reprojected}"')

    if promote_to_multi:
        options.append("-nlt PROMOTE_TO_MULTI")

    if preserve_fid:
        options.append("-preserve_fid")
    else:
        options.append("-unsetFid")

    if target_projection is not None:
        wkt = utils_gdal.parse_projection(target_projection, return_wkt=True).replace(" ", "\\")

        options.append(f'-t_srs "{wkt}"')

    # dst  # src
    success = gdal.VectorTranslate(
        out_path,
        input_path,
        format=utils_gdal._get_vector_driver_from_path(out_path),
        options=" ".join(options),
    )

    utils_gdal._delete_dataset_if_in_memory(clip_vector_reprojected)

    if success != 0:
        return out_path
    else:
        raise Exception("Error while clipping geometry.")


def clip_vector(
    vector,
    clip_geom,
    out_path=None,
    *,
    to_extent=False,
    target_projection=None,
    preserve_fid=True,
    prefix="",
    suffix="",
    add_uuid=False,
    allow_lists=True,
    overwrite=True,
    promote_to_multi=True,
):
    """
    Clips a vector to a geometry.

    ## Args:
    `vector` (_str_/_ogr.DataSource_/_list_): Vector(s) to clip. </br>
    `clip_geom` (_str_/_ogr.Geometry_): Vector to clip with. </br>

    ## Kwargs:
    `out_path` (_str_/_None_): Output path. If None, memory vectors are created. (Default: **None**) </br>
    `to_extent` (_bool_): Clip to extent. (Default: **False**) </br>
    `target_projection` (_str_/_ogr.DataSource_/_gdal.Dataset_/_osr.SpatialReference_/_int_/_None_): Target projection. (Default: **None**) </br>
    `preserve_fid` (_bool_): Preserve fid. (Default: **True**) </br>
    `prefix` (_str_): Prefix to add to the output path. (Default: **""**) </br>
    `suffix` (_str_): Suffix to add to the output path. (Default: **""**) </br>
    `add_uuid` (_bool_): Add UUID to the output path. (Default: **False**) </br>
    `allow_lists` (_bool_): Allow lists of vectors as input. (Default: **True**) </br>
    `overwrite` (_bool_): Overwrite output if it already exists. (Default: **True**) </br>
    `promote_to_multi` (_bool_): Should POLYGON by promoted to MULTIPOLYGON.. (Default: **True**) </br>

    ## Returns:
    (_str_/_list_): Output path(s) of clipped vector(s).
    """
    utils_base.type_check(vector, [str, ogr.DataSource, [str, ogr.DataSource]], "vector")
    utils_base.type_check(clip_geom, [ogr.DataSource, gdal.Dataset, str, list, tuple], "clip_geom")
    utils_base.type_check(out_path, [str, None], "out_path")
    utils_base.type_check(to_extent, [bool], "to_extent")
    utils_base.type_check(target_projection, [str, ogr.DataSource, gdal.Dataset, osr.SpatialReference, int, None], "target_projection")
    utils_base.type_check(preserve_fid, [bool], "preserve_fid")
    utils_base.type_check(prefix, [str], "prefix")
    utils_base.type_check(suffix, [str], "suffix")
    utils_base.type_check(add_uuid, [bool], "add_uuid")
    utils_base.type_check(allow_lists, [bool], "allow_lists")

    if not allow_lists and isinstance(vector, (list, tuple)):
        raise ValueError("Lists are not allowed for vector.")

    vector_list = utils_base._get_variable_as_list(vector)

    assert utils_gdal._check_is_vector_list(vector_list), f"Invalid vector in list: {vector_list}"

    path_list = utils_gdal.create_output_path_list(
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
            _clip_vector(
                in_vector,
                clip_geom,
                out_path=path_list[index],
                to_extent=to_extent,
                target_projection=target_projection,
                preserve_fid=preserve_fid,
                promote_to_multi=promote_to_multi,
            )
        )

    if isinstance(vector, list):
        return output

    return output[0]
