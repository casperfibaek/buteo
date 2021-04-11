from buteo.project_types import Metadata_raster, Metadata_vector
import sys

sys.path.append("../../")
from osgeo import gdal, ogr
from typing import Union, List
from buteo.raster.io import raster_to_metadata
from buteo.vector.io import (
    vector_in_memory,
    vector_to_memory,
    vector_to_metadata,
    vector_to_path,
)
from buteo.utils import (
    file_exists,
    remove_if_overwrite,
    type_check,
)
from buteo.gdal_utils import (
    gdal_bbox_intersects,
    raster_to_reference,
    reproject_extent,
    ready_io_raster,
    is_raster,
    is_vector,
    path_to_driver,
    default_options,
    translate_resample_method,
    gdal_nodata_value_from_type,
    align_bbox,
)


def clip_raster(
    raster: Union[List[Union[str, gdal.Dataset]], str, gdal.Dataset],
    clip_geom: Union[str, ogr.DataSource],
    out_path: Union[List[str], str, None] = None,
    resample_alg: str = "nearest",
    crop_to_geom: bool = True,
    adjust_bbox: bool = True,
    all_touch: bool = True,
    overwrite: bool = True,
    creation_options: list = [],
    dst_nodata: Union[str, int, float] = "infer",
    layer_to_clip: int = 0,
    opened: bool = False,
    prefix: str = "",
    postfix: str = "_clipped",
    verbose: int = 1,
) -> Union[list, gdal.Dataset, str]:
    """ Clips a raster(s) using a vector geometry or the extents of
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
    type_check(clip_geom, [str, ogr.DataSource], "clip_geom")
    type_check(out_path, [list, str], "out_path", allow_none=True)
    type_check(resample_alg, [str], "resample_alg")
    type_check(crop_to_geom, [bool], "crop_to_geom")
    type_check(adjust_bbox, [bool], "adjust_bbox")
    type_check(all_touch, [bool], "all_touch")
    type_check(dst_nodata, [str, int, float], "dst_nodata")
    type_check(layer_to_clip, [int], "layer_to_clip")
    type_check(overwrite, [bool], "overwrite")
    type_check(creation_options, [list], "creation_options")
    type_check(opened, [bool], "opened")
    type_check(prefix, [str], "prefix")
    type_check(postfix, [str], "postfix")
    type_check(verbose, [int], "verbose")

    raster_list, out_names = ready_io_raster(
        raster, out_path, overwrite, prefix, postfix
    )

    # Input is a vector.
    if is_vector(clip_geom):
        if vector_in_memory(clip_geom):
            clip_ds = clip_geom
        else:
            clip_ds = vector_to_memory(clip_geom)

        clip_metadata = vector_to_metadata(clip_ds, process_layer=layer_to_clip)

        if not isinstance(clip_metadata, dict):
            raise Exception("Error while parsing metadata.")

    # Input is a raster (use extent)
    elif is_raster(clip_geom):
        clip_metadata_raster = raster_to_metadata(clip_geom, simple=False)

        if not isinstance(clip_metadata_raster, dict):
            raise Exception("Error while parsing metadata.")

        clip_ds = vector_to_memory(clip_metadata_raster["extent_datasource"])

        clip_metadata = vector_to_metadata(clip_ds)

        if not isinstance(clip_metadata, dict):
            raise Exception("Error while parsing metadata.")

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

    metadatas = raster_to_metadata(raster_list)

    if not isinstance(metadatas, list):
        raise Exception("Error while parsing metadata.")

    clipped_rasters = []
    for index, in_raster in enumerate(raster_list):
        origin_layer = raster_to_reference(in_raster)

        raster_metadata = metadatas[index]
        origin_projection = raster_metadata["projection_osr"]
        origin_extent = raster_metadata["extent"]

        # Check if projections match, otherwise reproject target geom.
        if not origin_projection.IsSame(clip_projection):
            clip_metadata["extent"] = reproject_extent(
                clip_metadata["extent"], clip_projection, origin_projection,
            )

        # Fast check: Does the extent of the two inputs overlap?
        if not gdal_bbox_intersects(origin_extent, clip_extent):
            print("WARNING: Geometries did not intersect. Returning None.")
            return None

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
        out_name = out_names[index]
        out_format = path_to_driver(out_name)
        out_creation_options = default_options(creation_options)

        # nodata
        src_nodata = raster_metadata["nodata_value"]
        out_nodata = None
        if src_nodata is not None:
            out_nodata = src_nodata
        else:
            if dst_nodata == "infer":
                out_nodata = gdal_nodata_value_from_type(
                    raster_metadata["datatype_gdal_raw"]
                )
            elif dst_nodata == None:
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
            warpOptions=warp_options,
            srcNodata=raster_metadata["nodata_value"],
            dstNodata=out_nodata,
            multithread=True,
        )

        if verbose == 0:
            gdal.PopErrorHandler()

        if clipped is None:
            print("WARNING: Output is None. Returning empty layer.")

        if opened:
            clipped_rasters.append(clipped)
        else:
            clipped_rasters.append(out_name)

    if isinstance(raster, list):
        return clipped_rasters

    return clipped_rasters[0]
