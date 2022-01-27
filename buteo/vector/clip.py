import sys
from uuid import uuid4
from typing import Union, Optional, List
from osgeo import ogr, osr, gdal

sys.path.append("../../")

from buteo.gdal_utils import (
    is_vector,
    is_raster,
    parse_projection,
    path_to_driver_vector,
)
from buteo.utils import type_check
from buteo.raster.io import raster_to_metadata
from buteo.vector.io import (
    get_vector_path,
    internal_vector_to_memory,
    internal_vector_to_metadata,
    open_vector,
    ready_io_vector,
)


def internal_clip_vector(
    vector: Union[str, ogr.DataSource],
    clip_geom: Union[str, ogr.DataSource, gdal.Dataset],
    out_path: Optional[str] = None,
    process_layer: int = 0,
    process_layer_clip: int = 0,
    to_extent: bool = False,
    target_projection: Optional[
        Union[str, ogr.DataSource, gdal.Dataset, osr.SpatialReference, int]
    ] = None,
    preserve_fid: bool = True,
) -> str:
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
            extent = internal_vector_to_metadata(clip_geom, create_geometry=True)[
                "extent_datasource"
            ]
            geometry_to_clip = internal_vector_to_memory(extent)
        else:
            geometry_to_clip = open_vector(clip_geom, layer=process_layer_clip)
    elif is_raster(clip_geom):
        extent = raster_to_metadata(clip_geom, create_geometry=True)[
            "extent_datasource"
        ]
        geometry_to_clip = internal_vector_to_memory(extent)
    else:
        raise ValueError(f"Invalid input in clip_geom, unable to parse: {clip_geom}")

    clip_vector_path = internal_vector_to_metadata(geometry_to_clip)["path"]
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
    vector: Union[List[Union[str, ogr.DataSource]], Union[str, ogr.DataSource]],
    clip_geom: Union[str, ogr.DataSource, gdal.Dataset],
    out_path: str = None,
    process_layer: int = 0,
    process_layer_clip: int = 0,
    to_extent: bool = False,
    target_projection: Union[
        str, ogr.DataSource, gdal.Dataset, osr.SpatialReference, int
    ] = None,
    preserve_fid: bool = True,
    prefix: str = "",
    postfix: str = "",
    add_uuid: bool = False,
) -> Union[List[str], str]:
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

    output: List[str] = []
    for index, in_vector in enumerate(vector_list):
        output.append(
            internal_clip_vector(
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
