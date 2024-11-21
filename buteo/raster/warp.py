"""### Vectorize rasters. ###

Module to turn rasters into vector representations.
"""

# Standard library
from typing import Union, Optional, List

# External
from osgeo import gdal, ogr

# Internal
from buteo.utils import (
    utils_gdal,
    utils_path,
    utils_translate,
    utils_projection
)
from buteo.raster import core_raster



def raster_warp(
    raster: Union[str, gdal.Dataset],
    out_path: Optional[str] = None,
    src_projection: Optional[Union[str, gdal.Dataset]] = None,
    dst_projection: Optional[Union[str, gdal.Dataset]] = None,
    resampling_alg: str = "near",
    align_pixels: bool = False,
    dst_extent: Optional[Union[List[Union[int, float]], gdal.Dataset]] = None,
    dst_extent_srs: Optional[Union[str, gdal.Dataset]] = None,
    dst_x_res: Optional[Union[str, gdal.Dataset]] = None,
    dst_y_res: Optional[Union[str, gdal.Dataset]] = None,
    dst_width: Optional[Union[str, gdal.Dataset]] = None,
    dst_height: Optional[Union[str, gdal.Dataset]] = None,
    dst_nodata: Optional[Union[str, gdal.Dataset]] = None,
    clip_geom: Optional[Union[str, ogr.DataSource]]= None,
    overwrite: bool = True,
) -> str:
    """Warp a raster to a new projection, resolution, extent, or size."""

    if out_path is None:
        out_path = utils_path._get_temp_filepath(raster)

    assert utils_path._check_is_valid_output_filepath(out_path, overwrite=overwrite), f"out_path is not a valid output path: {out_path}"

    out_format = utils_gdal._get_driver_name_from_path(out_path)
    input_metadata = core_raster._get_basic_metadata_raster(raster)
    if src_projection is None:
        src_projection = input_metadata["projection_wkt"]
    else:
        src_projection = utils_projection.parse_projection(src_projection, return_wkt=True)

    if dst_projection is None:
        dst_projection = src_projection
    else:
        dst_projection = utils_projection.parse_projection(dst_projection, return_wkt=True)

    if dst_extent_srs is not None:
        dst_extent_srs = utils_projection.parse_projection(dst_extent_srs, return_wkt=True)

    resampling_alg = utils_translate._translate_resample_method(resampling_alg)

    if dst_extent is not None:
        if isinstance(dst_extent, (str, gdal.Dataset)):
            dst_extent = core_raster._get_basic_metadata_raster(dst_extent)["bbox_gdal"]

    if dst_x_res is not None:
        if isinstance(dst_x_res, (str, gdal.Dataset)):
            metadata = core_raster._get_basic_metadata_raster(dst_x_res)
            dst_x_res = metadata["pixel_width"]

    if dst_y_res is not None:
        if isinstance(dst_y_res, (str, gdal.Dataset)):
            metadata = core_raster._get_basic_metadata_raster(dst_y_res)
            dst_y_res = metadata["pixel_height"]

    if dst_width is not None:
        if isinstance(dst_width, (str, gdal.Dataset)):
            metadata = core_raster._get_basic_metadata_raster(dst_width)
            dst_width = metadata["width"]

    if dst_height is not None:
        if isinstance(dst_height, (str, gdal.Dataset)):
            metadata = core_raster._get_basic_metadata_raster(dst_height)
            dst_height = metadata["height"]

    if dst_nodata is not None:
        if isinstance(dst_nodata, (str, gdal.Dataset)):
            metadata = core_raster._get_basic_metadata_raster(dst_nodata)
            dst_nodata = metadata["nodata_value"]

    warp_options = gdal.WarpOptions(
        format=out_format,
        srcSRS=src_projection,
        dstSRS=dst_projection,
        resampleAlg=resampling_alg,
        outputBounds=dst_extent,
        outputBoundsSRS=dst_extent_srs,
        xRes=dst_x_res,
        yRes=dst_y_res,
        width=dst_width,
        height=dst_height,
        dstNodata=dst_nodata,
        multithread=True,
        targetAlignedPixels=align_pixels,
        cutlineDSName=clip_geom,
    )

    success = gdal.Warp(
        out_path,
        raster,
        options=warp_options,
    )

    assert success, f"Failed to warp raster: {raster}"

    return out_path
