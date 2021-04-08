import sys; sys.path.append('../../')
from typing import Union
from osgeo import gdal, osr, ogr
from uuid import uuid4
from buteo.utils import remove_if_overwrite, file_exists, type_check
from buteo.gdal_utils import (
    parse_projection,
    path_to_driver,
    raster_to_reference,
    default_options,
    translate_resample_method,
    gdal_nodata_value_from_type,
    is_raster,
    is_vector,
    raster_size_from_list,
    align_bbox,
    reproject_extent,
    ready_io_raster,
)
from buteo.raster.io import (
    default_options,
    raster_to_metadata,
)
from buteo.vector.io import (
    vector_to_metadata,
    vector_to_reference,
    vector_to_memory,
)

# TODO: documentation, robustness

def warp_raster(
    raster: Union[list, str, gdal.Dataset],
    out_path: Union[list, str, None]=None,
    projection: Union[int, str, gdal.Dataset, ogr.DataSource, osr.SpatialReference, None]=None,
    clip_geom: Union[str, ogr.DataSource, None]=None,
    target_size: Union[tuple, int, float, None]=None,
    target_in_pixels: bool=False,
    resample_alg: str="nearest",
    crop_to_geom: bool=True,
    all_touch: bool=True,
    adjust_bbox: bool=True,
    overwrite: bool=True,
    creation_options: list=[],
    src_nodata: Union[str, int, float]="infer",
    dst_nodata: Union[str, int, float]="infer",
    layer_to_clip: int=0,
    opened: bool=False,
    prefix: str="",
    postfix: str="_resampled",
) -> Union[gdal.Dataset, str]:
    type_check(raster, [list, str, gdal.Dataset], "raster")
    type_check(out_path, [list, str], "out_path", allow_none=True)
    type_check(projection, [int, str, gdal.Dataset, ogr.DataSource, osr.SpatialReference], "projection", allow_none=True)
    type_check(clip_geom, [str, ogr.DataSource], "clip_geom", allow_none=True)
    type_check(target_size, [tuple, int, float, str, gdal.Dataset], "target_size", allow_none=True)
    type_check(target_in_pixels, [bool], "target_in_pixels")
    type_check(resample_alg, [str], "resample_alg")
    type_check(crop_to_geom, [bool], "crop_to_geom")
    type_check(all_touch, [bool], "all_touch")
    type_check(adjust_bbox, [bool], "adjust_bbox")
    type_check(overwrite, [bool], "overwrite")
    type_check(creation_options, [list], "creation_options")
    type_check(src_nodata, [str, int, float], "src_nodata")
    type_check(dst_nodata, [str, int, float], "dst_nodata")
    type_check(layer_to_clip, [int], "layer_to_clip")
    type_check(opened, [bool], "opened")
    type_check(prefix, [str], "prefix")
    type_check(postfix, [str], "postfix")

    """ Warps a raster into a target raster

        Please be aware that all_touch does not work if target_size is set.
        If all_touch is required while resampling. Do it in two steps: 
        resample -> warp or resample -> clip.
    """

    raster_list, out_names = ready_io_raster(raster, out_path, overwrite, prefix, postfix)

    warped_rasters = []

    # options
    warp_options = []
    if all_touch:
        warp_options.append("CUTLINE_ALL_TOUCHED=TRUE")
    else:
        warp_options.append("CUTLINE_ALL_TOUCHED=FALSE")

    for index, in_raster in enumerate(raster_list):
        origin = raster_to_reference(in_raster)

        raster_metadata = raster_to_metadata(origin)
        origin_projection = raster_metadata["projection_osr"]
        origin_extent = raster_metadata["extent_geom_latlng"]

        target_projection = origin_projection
        if projection is not None:
            target_projection = parse_projection(projection)
        
        # clip placeholders.
        clip_metadata = None
        clip_extent = None
        clip_ds = None
        clip_projection = None
        output_bounds = None

        if clip_geom is not None:
            if is_raster(clip_geom):
                clip_metadata = raster_to_metadata(clip_geom, simple=False)
                clip_ds = clip_metadata["extent_ogr"]
            elif is_vector(clip_geom):
                clip_ds = vector_to_reference(clip_geom)
                clip_metadata = vector_to_metadata(clip_geom, simple=False)
            else:
                if file_exists(clip_geom):
                    raise ValueError(f"Unable to parse clip geometry: {clip_geom}")
                else:
                    raise ValueError(f"Unable to find clip geometry {clip_geom}")

            if layer_to_clip > (clip_metadata["layer_count"] - 1):
                raise ValueError("Requested an unable layer_to_clip.")

            clip_projection = clip_metadata["projection_osr"]
            clip_extent = clip_metadata["extent_geom_latlng"]

            # Fast check: Does the extent of the two inputs overlap?
            if not origin_extent.Intersects(clip_extent):
                print("WARNING: Geometries did not intersect. Returning None.")
                return None

            # Check if projections match, otherwise reproject target geom.
            if not target_projection.IsSame(clip_projection):
                clip_metadata["extent"] = reproject_extent(
                    clip_metadata["extent"],
                    clip_projection,
                    target_projection,
                )
            
            # The extent needs to be reprojected to the target.
            # this ensures that adjust_bbox works.
            x_min_og, y_max_og, x_max_og, y_min_og = reproject_extent(
                raster_metadata["extent"],
                origin_projection,
                target_projection,
            )
            output_bounds = (x_min_og, y_min_og, x_max_og, y_max_og) # gdal_warp format

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
                    x_min_og, y_max_og, x_max_og, y_min_og = clip_metadata["extent"]
                    output_bounds = (x_min_og, y_min_og, x_max_og, y_max_og) # gdal_warp format

            if clip_metadata["layer_count"] > 1:
                clip_ds = vector_to_memory(
                    clip_ds,
                    memory_path=f"clip_geom_{uuid4().int}.gpkg",
                    layer_to_extract=layer_to_clip,
                )
            elif not isinstance(clip_ds, str):
                clip_ds = vector_to_memory(
                    clip_ds,
                    memory_path=f"clip_geom_{uuid4().int}.gpkg",
                )

            if clip_ds is None:
                raise ValueError(f"Unable to parse input clip geom: {clip_geom}")

        x_res, y_res, x_pixels, y_pixels = raster_size_from_list(target_size, target_in_pixels)

        out_name = out_names[index]
        out_format = path_to_driver(out_path)
        out_creation_options = default_options(creation_options)

        # nodata
        src_nodata = raster_metadata["nodata_value"]
        out_nodata = None
        if src_nodata is not None:
            out_nodata = src_nodata
        else:
            if dst_nodata == "infer":
                out_nodata = gdal_nodata_value_from_type(raster_metadata["dtype_gdal_raw"])
            else:
                out_nodata = dst_nodata

        # Removes file if it exists and overwrite is True.
        remove_if_overwrite(out_path, overwrite)

        warped = gdal.Warp(
            out_name,
            origin,
            xRes=x_res,
            yRes=y_res,
            width=x_pixels,
            height=y_pixels,
            cutlineDSName=clip_ds,
            outputBounds=output_bounds,
            format=out_format,
            srcSRS=origin_projection,
            dstSRS=target_projection,
            resampleAlg=translate_resample_method(resample_alg),
            creationOptions=out_creation_options,
            warpOptions=warp_options,
            srcNodata=src_nodata,
            dstNodata=out_nodata,
            targetAlignedPixels=False,
            cropToCutline=False,
            multithread=True,
        )

        if opened:
            warped_rasters.append(warped)
        else:
            warped_rasters.append(out_name)

    if isinstance(raster, list):
        return warped_rasters
    
    return warped_rasters[0]
