import sys; sys.path.append('../../')
from uuid import uuid4
from typing import Union
from osgeo import ogr, osr, gdal

from buteo.gdal_utils import (
    is_vector,
    path_to_driver,
    is_raster,
)
from buteo.utils import type_check
from buteo.raster.io import raster_to_metadata
from buteo.vector.io import vector_to_path, vector_to_metadata, vector_add_index
from buteo.vector.merge import merge_vectors


def intersect_vector(
    vector: Union[ogr.DataSource, str, list],
    clip_geom: Union[ogr.DataSource, str, list, tuple],
    out_path: str=None,
    to_extent: bool=False,
    vector_idx: int=0,
    clip_idx: int=0,
    target_projection: Union[str, ogr.DataSource, gdal.Dataset, osr.SpatialReference, int]=None,
    preserve_fid: bool=True,
) -> str:
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

    geometry_to_clip = None
    if is_vector(clip_geom):
        if to_extent:
            extent = vector_to_metadata(clip_geom)["extent_ogr"]
            geometry_to_clip = vector_to_path(extent)
        else:
            geometry_to_clip = clip_geom
    elif is_raster(clip_geom):
        extent = raster_to_metadata(clip_geom)["extent_ogr"]
        geometry_to_clip = vector_to_path(extent)
    else:
        raise ValueError(f"Invalid input in clip_geom, unable to parse: {clip_geom}")      

    driver = ogr.GetDriverByName("GPKG")

    geometry_to_clip = None
    if is_vector(clip_geom):
        if to_extent:
            extent = vector_to_metadata(clip_geom, simple=False)["extent_ogr"]
            geometry_to_clip = vector_to_path(extent)
        else:
            geometry_to_clip = clip_geom
    elif is_raster(clip_geom):
        extent = raster_to_metadata(clip_geom, simple=False)["extent_ogr"]
        geometry_to_clip = vector_to_path(extent)
    else:
        raise ValueError(f"Invalid input in clip_geom, unable to parse: {clip_geom}")      

    driver = ogr.GetDriverByName(out_format)

    merged = merge_vectors([vector, geometry_to_clip], opened=True)

    vector_add_index(merged)

    vector_metadata = vector_to_metadata(vector)
    vector_layername = vector_metadata["layers"][vector_idx]["layer_name"]
    
    clip_geom_metadata = vector_to_metadata(clip_geom)
    clip_geom_layername = clip_geom_metadata["layers"][clip_idx]["layer_name"]

    sql = f"SELECT A.* FROM {vector_layername} A, {clip_geom_layername} B WHERE ST_INTERSECTS(A.geom, B.geom);"
    result = merged.ExecuteSQL(sql, dialect="SQLITE")

    destination = driver.CreateDataSource(out_target)

    destination.CopyLayer(result, vector_layername, ["OVERWRITE=YES"])


if __name__ == "__main__":
    folder = "C:/Users/caspe/Desktop/test/"

    vector = folder + "odense_grid.gpkg"
    clip_geom = folder + "havnen.gpkg"
    out_dir = folder + "out/"

    intersect_vector(vector, clip_geom, out_path=out_dir + "intersected.gpkg")

    import pdb; pdb.set_trace()