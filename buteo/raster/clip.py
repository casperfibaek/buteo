"""
### Clip rasters ###

Clips a raster using a vector geometry or the extents of a raster.

TODO:
    * Handle projections
    * Refactor clip_ds part.
"""

# Standard library
import sys; sys.path.append("../../")
import os

# External
from osgeo import gdal, ogr

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

    if out_path is not None:
        if "vsimem" not in out_path:
            if not os.path.isdir(os.path.split(os.path.normpath(out_path))[0]):
                raise ValueError(f"out_path folder does not exists: {out_path}")

    clip_ds = None

    # Input is a vector.
    if gdal_utils.is_vector(clip_geom):
        clip_ds = core_vector._open_vector(clip_geom)

        clip_metadata = core_vector._vector_to_metadata(
            clip_ds, process_layer=layer_to_clip,
        )

        if to_extent:
            clip_ds = clip_metadata["get_bbox_vector"]() # pylint: disable=not-callable

        elif clip_metadata["layer_count"] > 1:
            clip_ds = core_vector._vector_to_memory(clip_ds, layer_to_extract=layer_to_clip)

        if isinstance(clip_ds, ogr.DataSource):
            clip_ds = clip_ds.GetName()

    # Input is a raster (use extent)
    elif gdal_utils.is_raster(clip_geom):
        clip_metadata = core_raster._raster_to_metadata(clip_geom)
        clip_metadata["layer_count"] = 1
        clip_ds = clip_metadata["get_bbox_vector"]() # pylint: disable=not-callable

    else:
        if core_utils.file_exists(clip_geom):
            raise ValueError(f"Unable to parse clip geometry: {clip_geom}")
        else:
            raise ValueError(f"Unable to locate clip geometry {clip_geom}")

    if layer_to_clip > (clip_metadata["layer_count"] - 1):
        raise ValueError("Requested an unable layer_to_clip.")

    if clip_ds is None:
        raise ValueError(f"Unable to parse input clip geom: {clip_geom}")

    clip_projection = clip_metadata["projection_osr"]

    # options
    warp_options = []
    if all_touch:
        warp_options.append("CUTLINE_ALL_TOUCHED=TRUE")
    else:
        warp_options.append("CUTLINE_ALL_TOUCHED=FALSE")

    origin_layer = core_raster.open_raster(raster)

    raster_metadata = core_raster._raster_to_metadata(raster)
    origin_projection = raster_metadata["projection_osr"]

    # Check if projections match, otherwise reproject target geom.
    reprojection_needed = False
    if not origin_projection.IsSame(clip_projection):
        reprojection_needed = True
        clip_metadata["bbox"] = bbox_utils.reproject_bbox(
            clip_metadata["bbox"],
            clip_projection,
            origin_projection,
        )

    # Fast check: Does the extent of the two inputs overlap?
    if not bbox_utils.bboxes_intersect(raster_metadata["bbox"], clip_metadata["bbox"]):
        raise Exception("Geometries did not intersect.")

    if reprojection_needed:
        clip_new = _reproject_vector(clip_ds, origin_projection)
        gdal_utils.delete_if_in_memory(clip_ds)
        clip_ds = clip_new

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
    elif isinstance(src_nodata, (int, float, None)):
        src_nodata = float(src_nodata)
    else:
        raise ValueError(f"src_nodata must be an int, float or None: {src_nodata}")

    out_nodata = None
    if dst_nodata == "infer":
        if src_nodata == "infer":
            out_nodata = raster_metadata["nodata_value"]
        else:
            out_nodata = gdal_enums.get_default_nodata_value(raster_metadata["datatype_gdal_raw"])
    elif isinstance(dst_nodata, (int, float, None)):
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
        outputBounds=output_bounds,
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

    gdal_utils.delete_if_in_memory(clip_ds)

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

    ## Args:
    `raster` (_list_/_str_/_gdal.Dataset_): The raster(s) to clip. </br>
    `clip_geom` (_str_/_ogr.DataSource_/_gdal.Dataset_): The geometry to use to clip the raster </br>

    ## Kwargs:
    `out_path` (_str_/_list_/_None_): The path(s) to save the clipped raster to. If None a memory raster is created. (Default: **None**)</br>
    `resample_alg` (_str_): The resampling algorithm to use. (Default: **nearest**) </br>
    &emsp; • **nearest**: Nearest neighbour. </br>
    &emsp; • **bilinear**: Bilinear. </br>
    &emsp; • **cubic**: Cubic. </br>
    &emsp; • **cubicspline**: Cubic spline. </br>
    &emsp; • **lanczos**: Lanczos. </br>
    &emsp; • **average**: Average. </br>
    &emsp; • **mode**: Mode. </br>
    &emsp; • **max**: Max. </br>
    &emsp; • **min**: Min. </br>
    &emsp; • **median**: Median. </br>
    &emsp; • **q1**: Quartile 1 </br>
    &emsp; • **q3**: Quartile 3 </br>
    &emsp; • **sum**: Sum </br>
    &emsp; • **rms**: Root Mean Squared </br>
    `crop_to_geom` (_bool_): If True, the output raster will be cropped to the extent of the clip geometry. (Default: **True**)</br>
    `adjust_bbox` (_bool_): If True, the output raster have its bbox adjusted to match the clip geometry. (Default: **False**)</br>
    `all_touch` (_bool_): If true all pixels touching the clipping geometry will be included. (Default: **False**)</br>
    `to_extent` (_bool_): If True, the output raster will be cropped to the extent of the clip geometry. (Default: **False**)</br>
    `prefix` (_str_): The prefix to use for the output raster. (Default: **""**)</br>
    `suffix` (_str_): The suffix to use for the output raster. (Default: **""**)</br>
    `overwrite` (_bool_): If True, the output raster will be overwritten if it already exists. (Default: **True**)</br>
    `creation_options` (_list_/_None_): A list of creation options to pass to gdal. (Default: **None**)</br>
    `dst_nodata` (_int_/_float_/_None_): The nodata value to use for the output raster. (Default: **infer**)</br>
    `src_nodata` (_int_/_float_/_None_): The nodata value to use for the input raster. (Default: **infer**)</br>
    `layer_to_clip` (_int_): The layer ID in the vector to use for clipping. (Default: **0**)</br>
    `verbose` (_int_): The verbosity level. (Default: **0**)</br>
    `add_uuid` (_bool_): If True, a UUID will be added to the output raster. (Default: **False**)</br>
    `ram` (_str_): The amount of RAM to use for the operation. (Default: **auto**)</br>

    ## Returns:
    (_str_/_list_): A string or list of strings representing the path(s) to the clipped raster(s).
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
                ram=core_utils.get_dynamic_memory_limit_bytes(ram),
            )
        )

    if isinstance(raster, list):
        return output

    return output[0]
