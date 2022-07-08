"""
Clips a raster using a vector geometry or the extents of a raster.

TODO:
    - Combine the internal and external functions.
"""

import sys; sys.path.append("../../") # Path: buteo/raster/clip.py
import os

from osgeo import gdal, ogr

from buteo.raster.io import raster_to_metadata, ready_io_raster, open_raster
from buteo.vector.io import (
    open_vector,
    _vector_to_memory,
    _vector_to_metadata,
)
from buteo.utils.core import (
    file_exists,
    remove_if_overwrite,
    type_check,
)
from buteo.utils.gdal_utils import (
    gdal_bbox_intersects,
    reproject_extent,
    is_raster,
    is_vector,
    path_to_driver_raster,
    default_options,
    translate_resample_method,
    gdal_nodata_value_from_type,
    align_bbox,
)


def _clip_raster(
    raster,
    clip_geom,
    out_path,
    *,
    resample_alg="nearest",
    crop_to_geom=True,
    adjust_bbox=True,
    all_touch=True,
    to_extent=False,
    overwrite=True,
    creation_options=[],
    dst_nodata="infer",
    src_nodata="infer",
    layer_to_clip=0,
    prefix="",
    postfix="",
    verbose=1,
    uuid=False,
    ram=8000,
):
    """OBS: Internal. Single output.

    Clips a raster(s) using a vector geometry or the extents of
    a raster.
    """
    _, path_list = ready_io_raster(
        raster, out_path, overwrite=overwrite, prefix=prefix, postfix=postfix, uuid=uuid
    )

    if out_path is not None:
        if "vsimem" not in out_path:
            if not os.path.isdir(os.path.split(os.path.normpath(out_path))[0]):
                raise ValueError(f"out_path folder does not exists: {out_path}")

    # Input is a vector.
    if is_vector(clip_geom):
        clip_ds = open_vector(clip_geom)

        clip_metadata = _vector_to_metadata(
            clip_ds, process_layer=layer_to_clip, create_geometry=True,
        )

        if to_extent:
            clip_ds = clip_metadata["extent_datasource"]
        elif clip_metadata["layer_count"] > 1:
            clip_ds = _vector_to_memory(clip_ds, layer_to_extract=layer_to_clip)

        if isinstance(clip_ds, ogr.DataSource):
            clip_ds = clip_ds.GetName()

    # Input is a raster (use extent)
    elif is_raster(clip_geom):
        clip_metadata = raster_to_metadata(clip_geom, create_geometry=True)
        clip_metadata["layer_count"] = 1
        clip_ds = clip_metadata["extent_datasource"].GetName()
    else:
        if file_exists(clip_geom):
            raise ValueError(f"Unable to parse clip geometry: {clip_geom}")
        else:
            raise ValueError(f"Unable to locate clip geometry {clip_geom}")

    if layer_to_clip > (clip_metadata["layer_count"] - 1):
        raise ValueError("Requested an unable layer_to_clip.")

    if clip_ds is None:
        raise ValueError(f"Unable to parse input clip geom: {clip_geom}")

    clip_projection = clip_metadata["projection_osr"]
    clip_extent = clip_metadata["extent"]

    # options
    warp_options = []
    if all_touch:
        warp_options.append("CUTLINE_ALL_TOUCHED=TRUE")
    else:
        warp_options.append("CUTLINE_ALL_TOUCHED=FALSE")

    origin_layer = open_raster(raster)

    raster_metadata = raster_to_metadata(raster)
    origin_projection = raster_metadata["projection_osr"]
    origin_extent = raster_metadata["extent"]

    # Check if projections match, otherwise reproject target geom.
    if not origin_projection.IsSame(clip_projection):
        clip_metadata["extent"] = reproject_extent(
            clip_metadata["extent"],
            clip_projection,
            origin_projection,
        )

    # Fast check: Does the extent of the two inputs overlap?
    if not gdal_bbox_intersects(origin_extent, clip_extent):
        raise Exception("Geometries did not intersect.")

    output_bounds = raster_metadata["extent_gdal_warp"]

    if crop_to_geom:

        if adjust_bbox:
            output_bounds = align_bbox(
                raster_metadata["extent"],
                clip_metadata["extent"],
                raster_metadata["pixel_width"],
                raster_metadata["pixel_height"],
                warp_format=True,
            )

        else:
            output_bounds = clip_metadata["extent_gdal_warp"]

    # formats
    out_name = path_list[0]
    out_format = path_to_driver_raster(out_name)
    out_creation_options = default_options(creation_options)

    # nodata
    if src_nodata == "infer":
        src_nodata = raster_metadata["nodata_value"]

    out_nodata = None
    if dst_nodata == "infer":
        if src_nodata == "infer" and raster_metadata["nodata_value"] is not None:
            out_nodata = raster_metadata["nodata_value"]
        else:
            out_nodata = gdal_nodata_value_from_type(
                raster_metadata["datatype_gdal_raw"]
            )
    elif dst_nodata is None:
        out_nodata = None
    elif isinstance(dst_nodata, (int, float)):
        out_nodata = dst_nodata
    else:
        raise ValueError(f"Unable to parse nodata_value: {dst_nodata}")

    # Removes file if it exists and overwrite is True.
    remove_if_overwrite(out_path, overwrite)

    if verbose == 0:
        gdal.PushErrorHandler("CPLQuietErrorHandler")

    clipped = gdal.Warp(
        out_name,
        origin_layer,
        format=out_format,
        resampleAlg=translate_resample_method(resample_alg),
        targetAlignedPixels=False,
        outputBounds=output_bounds,
        xRes=raster_metadata["pixel_width"],
        yRes=raster_metadata["pixel_height"],
        cutlineDSName=clip_ds,
        cropToCutline=False,  # GDAL does this incorrectly when targetAlignedPixels is True.
        creationOptions=out_creation_options,
        warpMemoryLimit=ram,
        warpOptions=warp_options,
        srcNodata=src_nodata,
        dstNodata=out_nodata,
        multithread=True,
    )

    if verbose == 0:
        gdal.PopErrorHandler()

    if clipped is None:
        raise Exception("Error while clipping raster.")

    return out_name


def clip_raster(
    raster,
    clip_geom,
    out_path=None,
    *,
    resample_alg="nearest",
    crop_to_geom=True,
    adjust_bbox=True,
    all_touch=True,
    to_extent=False,
    prefix="",
    postfix="",
    overwrite=True,
    creation_options=[],
    dst_nodata="infer",
    src_nodata="infer",
    layer_to_clip=0,
    verbose=1,
    uuid=False,
    ram=8000,
):
    """Clips a raster(s) using a vector geometry or the extents of
        a raster.

    Args:
        raster(s) (list, path | raster): The raster(s) to clip.

        clip_geom (path | vector | raster): The geometry to use to clip
        the raster

    **kwargs:
        out_path (list, path | None): The destination to save to. If None then
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

        layer_to_clip (int): The layer in the input vector to use for clipping.

    Returns:
        An in-memory raster. If an out_path is given the output is a string containing
        the path to the newly created raster.
    """
    type_check(raster, [list, str, gdal.Dataset], "raster")
    type_check(clip_geom, [str, ogr.DataSource, gdal.Dataset], "clip_geom")
    type_check(out_path, [list, str], "out_path", allow_none=True)
    type_check(resample_alg, [str], "resample_alg")
    type_check(crop_to_geom, [bool], "crop_to_geom")
    type_check(adjust_bbox, [bool], "adjust_bbox")
    type_check(all_touch, [bool], "all_touch")
    type_check(to_extent, [bool], "to_extent")
    type_check(dst_nodata, [str, int, float], "dst_nodata", allow_none=True)
    type_check(src_nodata, [str, int, float], "src_nodata", allow_none=True)
    type_check(layer_to_clip, [int], "layer_to_clip")
    type_check(overwrite, [bool], "overwrite")
    type_check(creation_options, [list], "creation_options")
    type_check(prefix, [str], "prefix")
    type_check(postfix, [str], "postfix")
    type_check(verbose, [int], "verbose")
    type_check(uuid, [bool], "uuid")

    raster_list, path_list = ready_io_raster(
        raster,
        out_path,
        overwrite=overwrite,
        prefix=prefix,
        postfix=postfix,
        uuid=uuid,
    )

    output = []
    for index, in_raster in enumerate(raster_list):
        output.append(
            _clip_raster(
                in_raster,
                clip_geom,
                out_path=path_list[index],
                resample_alg=resample_alg,
                crop_to_geom=crop_to_geom,
                adjust_bbox=adjust_bbox,
                all_touch=all_touch,
                to_extent=to_extent,
                dst_nodata=dst_nodata,
                src_nodata=src_nodata,
                layer_to_clip=layer_to_clip,
                overwrite=overwrite,
                creation_options=creation_options,
                prefix=prefix,
                postfix=postfix,
                verbose=verbose,
                ram=ram,
            )
        )

    if isinstance(raster, list):
        return output

    return output[0]
