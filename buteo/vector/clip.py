import sys

sys.path.append("../../")
from uuid import uuid4
from typing import Union, Optional, List
from osgeo import ogr, osr, gdal

from buteo.gdal_utils import (
    is_vector,
    parse_projection,
    path_to_driver,
    is_raster,
)
from buteo.utils import type_check
from buteo.raster.io import internal_raster_to_metadata
from buteo.vector.io import (
    internal_vector_to_memory,
    internal_vector_to_metadata,
    ready_io_vector,
)


def internal_clip_vector(
    vector: Union[str, ogr.DataSource],
    clip_geom: Union[str, ogr.DataSource],
    out_path: Optional[str] = None,
    to_extent: bool = False,
    target_projection: Optional[
        Union[str, ogr.DataSource, gdal.Dataset, osr.SpatialReference, int]
    ] = None,
    preserve_fid: bool = True,
) -> str:
    """ Clips a vector to a geometry.

    Returns:
        A clipped ogr.Datasource or the path to one.
    """
    type_check(vector, [str, ogr.DataSource], "vector")
    type_check(clip_geom, [ogr.DataSource, gdal.Dataset, str, list, tuple], "clip_geom")
    type_check(out_path, [str], "out_path", allow_none=True)
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
        out_format = path_to_driver(out_path)

    options = []

    geometry_to_clip = None
    if is_vector(clip_geom):
        if to_extent:
            extent = internal_vector_to_metadata(clip_geom, create_geometry=True)[
                "extent_datasource"
            ]
            geometry_to_clip = internal_vector_to_memory(extent)
        else:
            geometry_to_clip = internal_vector_to_memory(clip_geom)
    elif is_raster(clip_geom):
        extent = internal_raster_to_metadata(clip_geom, create_geometry=True)[
            "extent_datasource"
        ]
        geometry_to_clip = internal_vector_to_memory(extent)
    else:
        raise ValueError(f"Invalid input in clip_geom, unable to parse: {clip_geom}")

    options.append(f"-clipsrc {geometry_to_clip}")

    if preserve_fid:
        options.append("-preserve_fid")
    else:
        options.append("-unsetFid")

    out_projection = None
    if target_projection is not None:
        out_projection = parse_projection(target_projection, return_wkt=True)
        options.append(f"-t_srs {out_projection}")

    # dst  # src
    success = gdal.VectorTranslate(
        out_target, vector, format=out_format, options=" ".join(options),
    )

    if success != 0:
        return out_target
    else:
        raise Exception("Error while clipping geometry.")


def clip_vector(
    vector: Union[List[Union[str, ogr.DataSource]], Union[str, ogr.DataSource]],
    clip_geom: Union[ogr.DataSource, str, list, tuple],
    out_path: str = None,
    to_extent: bool = False,
    target_projection: Union[
        str, ogr.DataSource, gdal.Dataset, osr.SpatialReference, int
    ] = None,
    preserve_fid: bool = True,
) -> Union[List[str], str]:
    """ Clips a vector to a geometry.

    Returns:
        A clipped ogr.Datasource or the path to one.
    """
    type_check(vector, [list, str, ogr.DataSource], "vector")
    type_check(clip_geom, [ogr.DataSource, gdal.Dataset, str, list, tuple], "clip_geom")
    type_check(out_path, [str], "out_path", allow_none=True)
    type_check(to_extent, [bool], "to_extent")
    type_check(
        target_projection,
        [str, ogr.DataSource, gdal.Dataset, osr.SpatialReference, int],
        "target_projection",
        allow_none=True,
    )
    type_check(preserve_fid, [bool], "preserve_fid")

    vector_list, path_list = ready_io_vector(vector, out_path)

    output: List[str] = []
    for index, in_vector in enumerate(vector_list):
        output.append(
            internal_clip_vector(
                in_vector,
                clip_geom,
                out_path=path_list[index],
                to_extent=to_extent,
                target_projection=target_projection,
                preserve_fid=preserve_fid,
            )
        )

    if isinstance(vector, list):
        return output

    return output[0]
