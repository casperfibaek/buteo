"""
### Clip rasters ###

Clips a raster using a vector geometry or the extents of a raster.
"""

# Standard library
import sys; sys.path.append("../../")
import os

# External
from osgeo import gdal, ogr
import numpy as np

# Internal
from buteo.raster import core_raster
from buteo.vector import core_vector
from buteo.utils import gdal_utils, core_utils, bbox_utils, gdal_enums
from buteo.vector.reproject import _reproject_vector


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
    creation_options=None,
    dst_nodata="infer",
    src_nodata="infer",
    layer_to_clip=0,
    prefix="",
    suffix="",
    verbose=1,
    add_uuid=False,
    ram="auto",
):
    """ INTERNAL. """
    path_list = gdal_utils.create_output_path_list(
        core_utils.ensure_list(raster),
        out_path=out_path,
        prefix=prefix,
        suffix=suffix,
        add_uuid=add_uuid,
    )

    if out_path is not None and isinstance(out_path, str):
        if "vsimem" not in out_path:
            if not os.path.isdir(os.path.split(os.path.normpath(out_path))[0]):
                raise ValueError(f"out_path folder does not exist: {out_path}")

    clip_ds = None

    memory_files = []

    # Input is a vector.
    if gdal_utils.is_vector(clip_geom):
        clip_ds = core_vector._open_vector(clip_geom)

        # TODO: Fix potential memory leak
        if clip_ds.GetLayerCount() > 1:
            clip_ds = core_vector.vector_filter_layer(clip_ds, layer_name_or_idx=layer_to_clip, add_uuid=True)
            memory_files.append(clip_ds)

        clip_metadata = core_vector._vector_to_metadata(clip_ds)

        if to_extent:
            clip_ds = bbox_utils.convert_bbox_to_vector(clip_metadata["bbox"], clip_metadata["projection_osr"])
            memory_files.append(clip_ds)

        if isinstance(clip_ds, ogr.DataSource):
            clip_ds = clip_ds.GetName()

    # Input is a raster (use extent)
    elif gdal_utils.is_raster(clip_geom):
        clip_metadata = core_raster._raster_to_metadata(clip_geom)
        clip_ds = bbox_utils.convert_bbox_to_vector(clip_metadata["bbox"], clip_metadata["projection_osr"])
        memory_files.append(clip_ds)
    else:
        if core_utils.file_exists(clip_geom):
            raise ValueError(f"Unable to parse clip geometry: {clip_geom}")
        else:
            raise ValueError(f"Unable to locate clip geometry {clip_geom}")

    if clip_ds is None:
        raise ValueError(f"Unable to parse input clip geom: {clip_geom}")

    # options
    warp_options = []
    if all_touch:
        warp_options.append("CUTLINE_ALL_TOUCHED=TRUE")
    else:
        warp_options.append("CUTLINE_ALL_TOUCHED=FALSE")

    origin_layer = core_raster.open_raster(raster)

    raster_metadata = core_raster._raster_to_metadata(raster)
    origin_projection = raster_metadata["projection_osr"]

    # Fast check: Does the extent of the two inputs overlap?
    has_inf = True in [np.isinf(val) for val in raster_metadata["bbox_latlng"]]
    if not has_inf and not bbox_utils.bboxes_intersect(raster_metadata["bbox_latlng"], clip_metadata["bbox_latlng"]):
        raise ValueError(f"Geometries of {raster} and {clip_geom} did not intersect.")

    if not origin_projection.IsSame(clip_metadata["projection_osr"]):
        clip_ds = _reproject_vector(clip_ds, origin_projection)
        clip_metadata = core_vector._vector_to_metadata(clip_ds)
        memory_files.append(clip_ds)

    output_bounds = raster_metadata["bbox"]

    if crop_to_geom:

        if adjust_bbox:
            output_bounds = bbox_utils.align_bboxes_to_pixel_size(
                raster_metadata["bbox"],
                clip_metadata["bbox"],
                raster_metadata["pixel_width"],
                raster_metadata["pixel_height"],
            )
        else:
            output_bounds = clip_metadata["bbox"]

    # formats
    out_name = path_list[0]
    out_format = gdal_utils.path_to_driver_raster(out_name)
    out_creation_options = gdal_utils.default_creation_options(creation_options)

    # nodata
    if src_nodata == "infer":
        src_nodata = raster_metadata["nodata_value"]
    elif isinstance(src_nodata, (int, float)) or src_nodata is None:
        src_nodata = float(src_nodata)
    else:
        raise ValueError(f"src_nodata must be an int, float or None: {src_nodata}")

    out_nodata = None
    if dst_nodata == "infer":
        if src_nodata == "infer":
            out_nodata = raster_metadata["nodata_value"]
        else:
            out_nodata = gdal_enums.get_default_nodata_value(raster_metadata["datatype_gdal_raw"])
    elif isinstance(dst_nodata, (int, float)) or dst_nodata is None:
        out_nodata = dst_nodata
    else:
        raise ValueError(f"Unable to parse nodata_value: {dst_nodata}")

    # Removes file if it exists and overwrite is True.
    core_utils.remove_if_required(out_path, overwrite)

    if verbose == 0:
        gdal.PushErrorHandler("CPLQuietErrorHandler")

    clipped = gdal.Warp(
        out_name,
        origin_layer,
        format=out_format,
        resampleAlg=gdal_enums.translate_resample_method(resample_alg),
        targetAlignedPixels=False,
        outputBounds=bbox_utils.convert_ogr_bbox_to_gdal_bbox(output_bounds),
        xRes=raster_metadata["pixel_width"],
        yRes=raster_metadata["pixel_height"],
        cutlineDSName=clip_ds,
        cropToCutline=False,
        creationOptions=out_creation_options,
        warpMemoryLimit=gdal_utils.get_gdalwarp_ram_limit(ram),
        warpOptions=warp_options,
        srcNodata=src_nodata,
        dstNodata=out_nodata,
        multithread=True,
    )

    gdal_utils.delete_if_in_memory_list(memory_files)

    if verbose == 0:
        gdal.PopErrorHandler()

    if clipped is None:
        raise ValueError("Error while clipping raster.")

    return out_name


def clip_raster(
    raster,
    clip_geom,
    out_path=None,
    *,
    resample_alg="nearest",
    crop_to_geom=True,
    adjust_bbox=False,
    all_touch=False,
    to_extent=False,
    prefix="",
    suffix="",
    overwrite=True,
    creation_options=None,
    dst_nodata="infer",
    src_nodata="infer",
    layer_to_clip=0,
    verbose=0,
    add_uuid=False,
    ram="auto",
):
    """
    Clips a raster(s) using a vector geometry or the extents of a raster.

    Args:
        raster (list/str/gdal.Dataset): The raster(s) to clip.
        clip_geom (str/ogr.DataSource/gdal.Dataset): The geometry to use to
            clip the raster.

    Keyword Args:
        out_path (str/list/None, default=None): The path(s) to save the
            clipped raster to. If None, a memory raster is created.
        resample_alg (str, default="nearest"): The resampling algorithm to use.
            Options include: nearest, bilinear, cubic, cubicspline, lanczos, average,
                mode, max, min, median, q1, q3, sum, rms.
        crop_to_geom (bool, default=True): If True, the output raster will be
            cropped to the extent of the clip geometry.
        adjust_bbox (bool, default=False): If True, the output raster will have its
            bbox adjusted to match the clip geometry.
        all_touch (bool, default=False): If true, all pixels touching the
            clipping geometry will be included.
        to_extent (bool, default=False): If True, the output raster will be
            cropped to the extent of the clip geometry.
        prefix (str, default=""): The prefix to use for the output raster.
        suffix (str, default=""): The suffix to use for the output raster.
        overwrite (bool, default=True): If True, the output raster will be
            overwritten if it already exists.
        creation_options (list/None, default=None): A list of creation options
            to pass to gdal.
        dst_nodata (int/float/None, default="infer"): The nodata value to use for
            the output raster.
        src_nodata (int/float/None, default="infer"): The nodata value to use for
            the input raster.
        layer_to_clip (int/str, default=0): The layer ID or name in the
            vector to use for clipping.
        verbose (int, default=0): The verbosity level.
        add_uuid (bool, default=False): If True, a UUID will be added to the
            output raster.
        ram (str, default="auto"): The amount of RAM to use for the operation.

    Returns:
        str/list: A string or list of strings representing the path(s) to
            the clipped raster(s).
    """
    core_utils.type_check(raster, [str, gdal.Dataset, [str, gdal.Dataset]], "raster")
    core_utils.type_check(clip_geom, [str, ogr.DataSource, gdal.Dataset], "clip_geom")
    core_utils.type_check(out_path, [[str], str, None], "out_path")
    core_utils.type_check(resample_alg, [str], "resample_alg")
    core_utils.type_check(crop_to_geom, [bool], "crop_to_geom")
    core_utils.type_check(adjust_bbox, [bool], "adjust_bbox")
    core_utils.type_check(all_touch, [bool], "all_touch")
    core_utils.type_check(to_extent, [bool], "to_extent")
    core_utils.type_check(dst_nodata, [str, int, float, None], "dst_nodata")
    core_utils.type_check(src_nodata, [str, int, float, None], "src_nodata")
    core_utils.type_check(layer_to_clip, [int], "layer_to_clip")
    core_utils.type_check(overwrite, [bool], "overwrite")
    core_utils.type_check(creation_options, [[str], None], "creation_options")
    core_utils.type_check(prefix, [str], "prefix")
    core_utils.type_check(suffix, [str], "postfix")
    core_utils.type_check(verbose, [int], "verbose")
    core_utils.type_check(add_uuid, [bool], "uuid")

    raster_list = core_utils.ensure_list(raster)
    path_list = gdal_utils.create_output_path_list(
        raster_list,
        out_path=out_path,
        prefix=prefix,
        suffix=suffix,
        add_uuid=add_uuid,
        overwrite=overwrite,
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
                suffix=suffix,
                verbose=verbose,
                ram=ram,
            )
        )

    if isinstance(raster, list):
        return output

    return output[0]
