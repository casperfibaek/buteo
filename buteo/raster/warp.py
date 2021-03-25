import sys; sys.path.append('../../')
from typing import Union
from osgeo import gdal, osr, ogr
from buteo.utils import remove_if_overwrite
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
)
from buteo.raster.io import (
    default_options,
    raster_to_metadata,
)
from buteo.vector.io import vector_to_metadata, reproject_vector


# TODO: documentation, robustness. handle layer
def warp_raster(
    raster: Union[str, gdal.Dataset],
    out_path: Union[str, None]=None,
    projection: Union[int, str, gdal.Dataset, ogr.DataSource, osr.SpatialReference, None]=None,
    clip_geom: Union[str, ogr.DataSource, None]=None,
    target_size: Union[tuple, int, float, None]=None,
    target_in_pixels: bool=False,
    resample_alg: str="nearest",
    crop_to_geom: bool=True,
    all_touch: bool=False,
    adjust_bbox: bool=True,
    overwrite: bool=True,
    creation_options: list=[],
    src_nodata: Union[str, int, float]="infer",
    dst_nodata: Union[str, int, float]="infer",
) -> Union[gdal.Dataset, str]:
    """ Warps a raster into a target raster

        Please be aware that all_touch does not work if target_size is set.
        If all_touch is required while resampling. Do it in two steps: 
        resample -> warp or resample -> clip.
    """
    ref = raster_to_reference(raster)
    metadata = raster_to_metadata(ref)

    source_projection = metadata["projection_osr"]
    target_projection = source_projection
    if projection is not None:
        target_projection = parse_projection(projection)
    
    out_clip_layer = None
    out_clip_ds = None
    out_clip_meta = None
    output_bounds = None
    if clip_geom is not None:
        if is_raster(clip_geom):
            out_clip_meta = raster_to_metadata(clip_geom)
        elif is_vector(clip_geom):
            out_clip_meta = vector_to_metadata(clip_geom)
        else:
            raise ValueError(f"The clip_geom is invalid: {clip_geom}")

        # Check if projections match, otherwise reproject target geom.
        if not metadata["projection_osr"].IsSame(out_clip_meta["projection_osr"]):
            clip_geom = reproject_vector(clip_geom, metadata["projection_osr"], out_path="/vsimem/clip_geom.gpkg")

        if isinstance(clip_geom, str):
            out_clip_ds = clip_geom
        elif isinstance(clip_geom, ogr.DataSource):
            out_clip_layer = clip_geom.GetLayer()
        elif isinstance(clip_geom, ogr.Layer):
            out_clip_layer = clip_geom
        else:
            raise Exception("Unable to parse clip_geom.")

        og_minX, og_maxY, og_maxX, og_minY = metadata["extent"]
        output_bounds = (og_minX, og_minY, og_maxX, og_maxY)

        if crop_to_geom:
            clip_meta = vector_to_metadata(clip_geom)

            if adjust_bbox:
                output_bounds = align_bbox(
                    metadata["extent"],
                    clip_meta["extent"],
                    metadata["pixel_width"],
                    metadata["pixel_height"],
                )

    x_res, y_res, x_pixels, y_pixels = raster_size_from_list(target_size, target_in_pixels)

    warp_options = []
    if all_touch:
        warp_options.append("CUTLINE_ALL_TOUCHED=TRUE")
    
    if all_touch and target_size is not None:
        print("WARNING: all_touch is disabled when target_size is set. Do a two step approach if all_touch is required.")

    # formats
    out_name = None
    out_format = None
    out_creation_options = [] if creation_options is None else creation_options
    if out_path is None:
        out_name = metadata["name"]
        out_format = "MEM"
    else:
        out_creation_options = default_options(creation_options)
        out_name = out_path
        out_format = path_to_driver(out_path)
    
    # nodata
    if src_nodata is None:
        src_nodata = metadata["nodata_value"]

    out_nodata = None
    if src_nodata is not None:
        out_nodata = src_nodata
    else:
        if dst_nodata == "infer":
            out_nodata = gdal_nodata_value_from_type(metadata["dtype_gdal_raw"])
        else:
            out_nodata = dst_nodata

    remove_if_overwrite(out_path, overwrite)

    warped = gdal.Warp(
        out_name,
        ref,
        xRes=x_res,
        yRes=y_res,
        width=x_pixels,
        height=y_pixels,
        cutlineDSName=out_clip_ds,
        cutlineLayer=out_clip_layer,
        outputBounds=output_bounds,
        outputBoundsSRS=source_projection,
        format=out_format,
        srcSRS=source_projection,
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

    if out_path is not None:
        warped = None
        return out_path
    else:
        return warped
