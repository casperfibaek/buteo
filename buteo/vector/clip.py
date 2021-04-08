import sys; sys.path.append('../../')
from uuid import uuid4
from typing import Union
from osgeo import ogr, osr, gdal

from buteo.gdal_utils import (
    is_vector,
    parse_projection,
    path_to_driver,
    is_raster,
)
from buteo.utils import type_check
from buteo.raster.io import raster_to_metadata
from buteo.vector.io import vector_to_path, vector_to_metadata


def clip_vector(
    vector: Union[ogr.DataSource, str, list],
    clip_geom: Union[ogr.DataSource, str, list, tuple],
    out_path: str=None,
    to_extent: bool=False,
    target_projection: Union[str, ogr.DataSource, gdal.Dataset, osr.SpatialReference, int]=None,
    preserve_fid: bool=True,
):
    """ Clips a vector to a geometry.
    Args:
        vector (list of vectors | path | vector): The vectors(s) to clip.

        clip_geom (list of geom | path | vector | rasters): The geometry to use
        for the clipping

    **kwargs:


    Returns:
        A clipped ogr.Datasource or the path to one.
    """
    type_check(vector, [ogr.DataSource, str, list], "vector")
    type_check(clip_geom, [ogr.DataSource, gdal.Dataset, str, list, tuple], "clip_geom")
    type_check(out_path, [str], "out_path", allow_none=True)
    type_check(to_extent, [bool], "to_extent")
    type_check(target_projection, [str, ogr.DataSource, gdal.Dataset, osr.SpatialReference, int], "target_projection", allow_none=True)
    type_check(preserve_fid, [bool], "preserve_fid")

    out_format = '.gpkg'
    out_target = f"/vsimem/clipped_{uuid4().int}{out_format}"

    if out_path is not None:
        out_target = out_path
        out_format = path_to_driver(out_path)

    options = []

    geometry_to_clip = None
    if is_vector(clip_geom):
        if to_extent:
            extent = vector_to_metadata(clip_geom, simple=False)["extent_datasource"]
            geometry_to_clip = vector_to_path(extent)
        else:
            geometry_to_clip = clip_geom
    elif is_raster(clip_geom):
        extent = raster_to_metadata(clip_geom, simple=False)["extent_datasource"]
        geometry_to_clip = vector_to_path(extent)
    else:
        raise ValueError(f"Invalid input in clip_geom, unable to parse: {clip_geom}")      

    options.append(f'-clipsrc {vector_to_path(geometry_to_clip)}')

    if preserve_fid:
        options.append("-preserve_fid")
    else:
        options.append("-unsetFid")

    out_projection = None
    if target_projection is not None:
        out_projection = parse_projection(target_projection, return_wkt=True)
        options.append(f"-t_srs {out_projection}")

    success = gdal.VectorTranslate(
        out_target,     # dst
        vector,         # src
        format=out_format,
        options=" ".join(options),
    )

    if success != 0:
        return out_target
    else:
        raise Exception("Error while clipping geometry.")

