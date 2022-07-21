"""
### GDAL Warp functions  ###

Module to wrap the functionality of GDAL's gdalwarp.
"""

# Standard library
import sys; sys.path.append("../../")
from uuid import uuid4

# External
from osgeo import gdal, ogr, osr

# Internal
from buteo.utils import core_utils, gdal_utils, bbox_utils, gdal_enums
from buteo.raster import core_raster
from buteo.vector import core_vector



def _warp_raster(
    raster,
    out_path=None,
    *,
    projection=None,
    clip_geom=None,
    target_size=None,
    target_in_pixels=False,
    resample_alg="nearest",
    crop_to_geom=True,
    all_touch=False,
    adjust_bbox=False,
    src_nodata="infer",
    dst_nodata="infer",
    layer_to_clip=0,
    prefix="",
    suffix="_resampled",
    overwrite=True,
    creation_options=None,
):
    """ Internal. """
    if out_path is None:
        out_path = gdal_utils.create_memory_path(
            out_path,
            prefix=prefix,
            suffix=suffix,
            add_uuid=True,
        )

    origin = core_raster._open_raster(raster)
    raster_metadata = core_raster._raster_to_metadata(origin)

    # options
    warp_options = []
    if all_touch:
        warp_options.append("CUTLINE_ALL_TOUCHED=TRUE")
    else:
        warp_options.append("CUTLINE_ALL_TOUCHED=FALSE")

    origin_projection = raster_metadata["projection_osr"]
    origin_extent = raster_metadata["get_bbox_as_vector_latlng"]() # pylint: disable=not-callable

    target_projection = origin_projection
    if projection is not None:
        target_projection = gdal_utils.parse_projection(projection)

    if clip_geom is not None:
        if gdal_utils.is_raster(clip_geom):
            opened_raster = core_raster._open_raster(clip_geom)
            clip_metadata_raster = core_raster._raster_to_metadata(opened_raster)
            clip_ds = clip_metadata_raster["get_bbox_as_vector"]() # pylint: disable=not-callable
            clip_metadata = core_vector._vector_to_metadata(clip_ds)
        elif gdal_utils.is_vector(clip_geom):
            clip_ds = core_vector.open_vector(clip_geom)
            clip_metadata = core_vector._vector_to_metadata(clip_ds)
        else:
            if core_utils.file_exists(clip_geom):
                raise ValueError(f"Unable to parse clip geometry: {clip_geom}")
            else:
                raise ValueError(f"Unable to find clip geometry {clip_geom}")

        if layer_to_clip > (clip_metadata["layer_count"] - 1):
            raise ValueError("Requested an unable layer_to_clip.")

        clip_projection = clip_metadata["projection_osr"]
        clip_extent = clip_metadata["get_bbox_as_vector_latlng"]() # pylint: disable=not-callable

        # Fast check: Does the extent of the two inputs overlap?
        if not origin_extent.Intersects(clip_extent):
            raise Exception("Clipping geometry did not intersect raster.")

        # Check if projections match, otherwise reproject target geom.
        if not target_projection.IsSame(clip_projection):
            clip_metadata["bbox"] = bbox_utils.reproject_bbox(
                clip_metadata["bbox"],
                clip_projection,
                target_projection,
            )

        # The extent needs to be reprojected to the target.
        # this ensures that adjust_bbox works.
        output_bounds = bbox_utils.reproject_bbox(
            raster_metadata["extent"],
            origin_projection,
            target_projection,
        )

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

        if clip_metadata["layer_count"] > 1:
            clip_ds = core_vector._vector_to_memory(
                clip_ds,
                memory_path=f"clip_geom_{uuid4().int}.gpkg",
                layer_to_extract=layer_to_clip,
            )
        elif not isinstance(clip_ds, str):
            clip_ds = core_vector._vector_to_memory(
                clip_ds,
                memory_path=f"clip_geom_{uuid4().int}.gpkg",
            )

        if clip_ds is None:
            raise ValueError(f"Unable to parse input clip geom: {clip_geom}")

    x_res, y_res, x_pixels, y_pixels = gdal_utils.parse_raster_size(
        target_size, target_in_pixels=target_in_pixels
    )

    out_format = gdal_utils.path_to_driver_raster(out_path)

    # nodata
    out_nodata = None
    if src_nodata is not None:
        out_nodata = raster_metadata["nodata_value"]
    else:
        if dst_nodata == "infer":
            out_nodata = raster_metadata["nodata_value"]
        else:
            out_nodata = dst_nodata

    # Removes file if it exists and overwrite is True.
    core_utils.remove_if_required(out_path, overwrite)

    warped = gdal.Warp(
        out_path,
        origin,
        xRes=x_res,
        yRes=y_res,
        width=x_pixels,
        height=y_pixels,
        cutlineDSName=clip_ds,
        outputBounds=bbox_utils.convert_geom_to_bbox(output_bounds),
        format=out_format,
        srcSRS=origin_projection,
        dstSRS=target_projection,
        resampleAlg=gdal_enums.translate_resample_method(resample_alg),
        creationOptions=gdal_utils.default_creation_options(creation_options),
        warpOptions=warp_options,
        srcNodata=src_nodata,
        dstNodata=out_nodata,
        targetAlignedPixels=False,
        cropToCutline=False,
        multithread=True,
    )

    if warped is None:
        raise Exception(f"Error while warping raster: {raster}")

    return out_path


def warp_raster(
    raster,
    out_path=None,
    *,
    projection=None,
    clip_geom=None,
    target_size=None,
    target_in_pixels=False,
    resample_alg="nearest",
    crop_to_geom=True,
    all_touch=False,
    adjust_bbox=False,
    src_nodata="infer",
    dst_nodata="infer",
    layer_to_clip=0,
    prefix="",
    suffix="_resampled",
    add_uuid=False,
    overwrite=True,
    creation_options=None,
):
    """
    Warps a raster into a target raster. </br>

    Please be aware that all_touch does not work if target_size is set.
    If all_touch is required while resampling. Do it in two steps:
    resample -> warp or resample -> clip.

    ## Args:
    `raster` (_str_/_list_/_gdal.Dataset_): The raster(s) to warp.

    ## Kwargs:
    `out_path` (_str_/_list_/_None_): The path to the output raster(s). If not set, a memory raster is created. (Default: **None**) </br>
    `projection` (_str_/_osr.SpatialReference_/_gdal.Dataset_/_ogr.DataSource_): The projection of the output raster. If not set, the projection of the input raster is used. (Default: **None**) </br>
    `clip_geom` (_str_/_gdal.Dataset_/_ogr.DataSource_): The geometry to clip the raster to. (Default: **None**) </br>
    `target_size` (_tuple_/_None_): The target size of the output raster. If not set, the size of the input raster is used. (Default: **None**) </br>
    `target_in_pixels` (_bool_): If True, the target size is in pixels. If False, the target size is in map units. (Default: **False**) </br>
    `resample_alg` (_str_): The resampling algorithm. (Default: **nearest**) </br>
    `crop_to_geom` (_bool_): If True, the output raster is cropped to the extent of the clip geometry. (Default: **True**) </br>
    `all_touch` (_bool_): If True, all pixels touching the clipping geometry is included. (Default: **False**) </br>
    `adjust_bbox` (_bool_): If True, the bounding box of the output raster is adjusted to align with the clipping geometry. (Default: **False**) </br>
    `src_nodata` (_str_/_int_/_float_/_None_): The nodata value of the input raster. If not set, the nodata value of the input raster is used. (Default: **infer**) </br>
    `dst_nodata` (_str_/_int_/_float_/_None_): The nodata value of the output raster. If not set, the nodata value of the input raster is used. (Default: **infer**) </br>
    `layer_to_clip` (_int_): The layer to clip the raster to. (Default: **0**) </br>
    `prefix` (_str_): The prefix of the output raster. (Default: **""**) </br>
    `suffix` (_str_): The suffix of the output raster. (Default: **"_resampled"**) </br>
    `add_uuid` (_bool_): If True, a unique identifier is added to the output raster. (Default: **False**) </br>
    `overwrite` (_bool_): If True, the output raster is overwritten if it already exists. (Default: **True**) </br>
    `creation_options` (_list_): The creation options of the output raster. (Default: **None**) </br>

    ## Returns:
    (_str_/_list_): The path to the warped output raster(s).
    """
    core_utils.type_check(raster, [str, gdal.Dataset, [str, gdal.Dataset]], "raster")
    core_utils.type_check(out_path, [str, [str], None], "out_path")
    core_utils.type_check(projection, [int, str, gdal.Dataset, ogr.DataSource, osr.SpatialReference, None], "projection")
    core_utils.type_check(clip_geom, [str, ogr.DataSource], "clip_geom")
    core_utils.type_check(target_size, [tuple, list, int, float, str, gdal.Dataset, None], "target_size")
    core_utils.type_check(target_in_pixels, [bool], "target_in_pixels")
    core_utils.type_check(resample_alg, [str], "resample_alg")
    core_utils.type_check(crop_to_geom, [bool], "crop_to_geom")
    core_utils.type_check(all_touch, [bool], "all_touch")
    core_utils.type_check(adjust_bbox, [bool], "adjust_bbox")
    core_utils.type_check(src_nodata, [str, int, float], "src_nodata")
    core_utils.type_check(dst_nodata, [str, int, float], "dst_nodata")
    core_utils.type_check(layer_to_clip, [int], "layer_to_clip")
    core_utils.type_check(prefix, [str], "prefix")
    core_utils.type_check(suffix, [str], "postfix")
    core_utils.type_check(add_uuid, [bool], "add_uuid")
    core_utils.type_check(overwrite, [bool], "overwrite")
    core_utils.type_check(creation_options, [[str], None], "creation_options")

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
            _warp_raster(
                in_raster,
                out_path=path_list[index],
                projection=projection,
                clip_geom=clip_geom,
                target_size=target_size,
                target_in_pixels=target_in_pixels,
                resample_alg=resample_alg,
                crop_to_geom=crop_to_geom,
                all_touch=all_touch,
                adjust_bbox=adjust_bbox,
                overwrite=overwrite,
                creation_options=creation_options,
                src_nodata=src_nodata,
                dst_nodata=dst_nodata,
                layer_to_clip=layer_to_clip,
                prefix=prefix,
                suffix=suffix,
            )
        )

    if isinstance(raster, list):
        return output

    return output[0]
