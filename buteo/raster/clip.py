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
    align_pixels: bool=True,
    overwrite: bool=True,
    creation_options: list=[],
    dst_nodata: Union[str, int, float]="infer",
) -> Union[gdal.Dataset, str]:
    """ Clips a raster using a vector geometry or the extents of
        a raster.

    Args:
        raster (path | raster): The raster to clip.
        
        clip_geom (path | vector | raster): The geometry to use to clip
        the raster

    **kwargs:
        out_path (path | None): The destination to save to. If None then
        the output is an in-memory raster.

        resample_alg (str): The algorithm to resample the raster. The following
        are available:
            'nearest', 'bilinear', 'cubic', 'cubicSpline', 'lanczos', 'average',
            'mode', 'max', 'min', 'median', 'q1', 'q3', 'sum', 'rms'.
        
        crop_to_geom (bool): Should the extent of the raster be clipped
        to the extent of the clipping geometry.

        all_touch (bool): Should all the pixels touched by the clipped 
        geometry be included or only those which centre lie within the
        geometry.

        align_pixels (bool): Should the resulting pixels be aligned with the
        original pixels.

        overwite (bool): Is it possible to overwrite the out_path if it exists.

        creation_options (list): A list of options for the GDAL creation. Only
        used if an outpath is specified. Defaults are:
            "TILED=YES"
            "NUM_THREADS=ALL_CPUS"
            "BIGG_TIF=YES"
            "COMPRESS=LZW"

        dst_nodata (str | int | float): If dst_nodata is 'infer' the destination nodata
        is the src_nodata if one exists, otherwise it's automatically chosen based
        on the datatype. If an int or a float is given, it is used as the output nodata.

    Returns:
        An in-memory raster. If an out_path is given the output is a string containing
        the path to the newly created raster.
    """

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
