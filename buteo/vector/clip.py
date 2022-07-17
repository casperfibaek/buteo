"""
Clip vector files with other geometries. Can come from rasters or vectors.

TODO:
    - Improve documentation
    - Remove internal step
"""

import sys; sys.path.append("../../") # Path: buteo/vector/attributes.py
from uuid import uuid4

from osgeo import ogr, osr, gdal

from buteo.utils.gdal_utils import (
    is_vector,
    is_raster,
    parse_projection,
    path_to_driver_vector,
)
from buteo.utils.core_utils import type_check
from buteo.raster.core_raster import _raster_to_metadata
from buteo.vector.core_vector import (
    get_vector_path,
    _vector_to_metadata,
    open_vector,
    ready_io_vector,
)


def _clip_vector(
    vector,
    clip_geom,
    out_path=None,
    *,
    process_layer=0,
    process_layer_clip=0,
    to_extent=False,
    target_projection=None,
    preserve_fid=True,
):
    """Clips a vector to a geometry.

    Returns:
        A clipped ogr.Datasource or the path to one.
    """
    type_check(vector, [str, ogr.DataSource], "vector")
    type_check(clip_geom, [ogr.DataSource, gdal.Dataset, str, list, tuple], "clip_geom")
    type_check(out_path, [str], "out_path", allow_none=True)
    type_check(process_layer, [int], "process_layer")
    type_check(process_layer_clip, [int], "process_layer_clip")
    type_check(to_extent, [bool], "to_extent")
    type_check(
        target_projection,
        [str, ogr.DataSource, gdal.Dataset, osr.SpatialReference, int],
        "target_projection",
        allow_none=True,
    )
    type_check(preserve_fid, [bool], "preserve_fid")

    out_format = ".gpkg"
    out_target = f"/vsimem/clipped_{uuid4().int}{out_format}"

    if out_path is not None:
        out_target = out_path
        out_format = path_to_driver_vector(out_path)

    options = []

    geometry_to_clip = None
    if is_vector(clip_geom):
        if to_extent:
            extent = _vector_to_metadata(clip_geom)["extent_datasource"]() # pylint: disable=not-callable
            geometry_to_clip = extent
        else:
            geometry_to_clip = open_vector(clip_geom, layer=process_layer_clip)
    elif is_raster(clip_geom):
        extent = _raster_to_metadata(clip_geom)["extent_datasource"]() # pylint: disable=not-callable
        geometry_to_clip = extent
    else:
        raise ValueError(f"Invalid input in clip_geom, unable to parse: {clip_geom}")

    clip_vector_path = _vector_to_metadata(geometry_to_clip)["path"]
    options.append(f"-clipsrc {clip_vector_path}")

    if preserve_fid:
        options.append("-preserve_fid")
    else:
        options.append("-unsetFid")

    out_projection = None
    if target_projection is not None:
        out_projection = parse_projection(target_projection, return_wkt=True)
        options.append(f"-t_srs {out_projection}")

    origin = open_vector(vector, layer=process_layer)

    # dst  # src
    success = gdal.VectorTranslate(
        out_target,
        get_vector_path(origin),
        format=out_format,
        options=" ".join(options),
    )

    if success != 0:
        return out_target
    else:
        raise Exception("Error while clipping geometry.")


def clip_vector(
    vector,
    clip_geom,
    out_path=None,
    *,
    process_layer=0,
    process_layer_clip=0,
    to_extent=False,
    target_projection=None,
    preserve_fid=True,
    prefix="",
    postfix="",
    add_uuid=False,
):
    """Clips a vector to a geometry.

    Returns:
        A clipped ogr.Datasource or the path to one.
    """
    type_check(vector, [list, str, ogr.DataSource], "vector")
    type_check(clip_geom, [ogr.DataSource, gdal.Dataset, str, list, tuple], "clip_geom")
    type_check(out_path, [str], "out_path", allow_none=True)
    type_check(process_layer, [int], "process_layer")
    type_check(process_layer_clip, [int], "process_layer_clip")
    type_check(to_extent, [bool], "to_extent")
    type_check(
        target_projection,
        [str, ogr.DataSource, gdal.Dataset, osr.SpatialReference, int],
        "target_projection",
        allow_none=True,
    )
    type_check(preserve_fid, [bool], "preserve_fid")

    vector_list, path_list = ready_io_vector(
        vector, out_path, prefix=prefix, postfix=postfix, add_uuid=add_uuid
    )

    output = []
    for index, in_vector in enumerate(vector_list):
        output.append(
            _clip_vector(
                in_vector,
                clip_geom,
                out_path=path_list[index],
                process_layer=process_layer,
                process_layer_clip=process_layer_clip,
                to_extent=to_extent,
                target_projection=target_projection,
                preserve_fid=preserve_fid,
            )
        )

    if isinstance(vector, list):
        return output

    return output[0]
