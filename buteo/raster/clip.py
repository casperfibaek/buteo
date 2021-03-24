import sys; sys.path.append('../../')
import os
from osgeo import gdal, ogr
from typing import Union
from buteo.raster.io import raster_to_metadata
from buteo.vector.io import vector_to_metadata, vector_to_memory
from buteo.utils import remove_if_overwrite
from buteo.gdal_utils import (
    raster_to_reference,
    vector_to_reference,
    is_raster,
    is_vector,
    path_to_driver,
    default_options,
    translate_resample_method,
    gdal_nodata_value_from_type,
)


def clip_raster(
    raster: Union[str, gdal.Dataset],
    clip_geom: Union[str, ogr.DataSource],
    out_path: Union[str, None]=None,
    resample_alg: str="nearest",
    crop_to_geom: bool=True,
    all_touch: bool=False,
    align_pixels: bool=False,
    overwrite: bool=True,
    creation_options: list=[],
    dst_nodata: Union[str, int, float]="infer",
) -> Union[gdal.Dataset, str]:

    # Verify inputs
    ref = raster_to_reference(raster)
    metadata = raster_to_metadata(ref)
    
    # Verify geom
    clip_ref = None
    meta_ref = None
    if is_raster(clip_geom):
        meta_ref = raster_to_metadata(clip_geom)
        clip_ref = meta_ref["extent_ogr"]
    elif is_vector(clip_geom):
        clip_ref = vector_to_reference(clip_geom)
        meta_ref = vector_to_metadata(clip_ref)
    else:
        raise ValueError(f"The clip_geom is invalid: {clip_geom}")

    remove_if_overwrite(out_path, overwrite)

    # Fast check: Does the extent of the two inputs overlap?
    intersection = metadata["extent_ogr_geom"].Intersection(meta_ref["extent_ogr_geom"])
    if intersection is None or intersection.Area() == 0.0:
        print("WARNING: Geometries did not intersect. Returning empty layer.")

    out_clip_geom = None
    if isinstance(clip_geom, str):
        out_clip_geom = clip_geom
    else:
        out_clip_geom = vector_to_memory(clip_ref, "/vsimem/clip_geom.gpkg")

    warp_options = []
    if all_touch:
        warp_options.append("CUTLINE_ALL_TOUCHED=TRUE")

    out_name = None
    out_format = None
    out_creation_options = None
    if out_path is None:
        out_name = metadata["name"]
        out_format = "MEM"
        out_creation_options = []
    else:
        out_name = out_path
        out_format = path_to_driver(out_path)
        out_creation_options = default_options(creation_options)

    src_nodata = metadata["nodata_value"]
    out_nodata = None
    if src_nodata is not None:
        out_nodata = src_nodata
    else:
        if dst_nodata == "infer":
            out_nodata = gdal_nodata_value_from_type(metadata["dtype_gdal_raw"])
        else:
            out_nodata = dst_nodata


    clipped = gdal.Warp(
        out_name,
        ref,
        format=out_format,
        creationOptions=out_creation_options,
        resampleAlg=translate_resample_method(resample_alg),
        targetAlignedPixels=align_pixels,
        cutlineDSName=out_clip_geom,
        cropToCutline=crop_to_geom,
        warpOptions=warp_options,
        srcNodata=metadata["nodata_value"],
        dstNodata=out_nodata,
        multithread=True,
    )

    if clipped is None:
        print("WARNING: Geometries did not intersect. Returning empty layer.")

    if out_path is not None:
        return out_path
    else:
        return clipped
