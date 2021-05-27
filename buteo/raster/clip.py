import sys

sys.path.append("../../")
from osgeo import gdal, ogr
from typing import Union, List, Optional
from buteo.raster.io import internal_raster_to_metadata, ready_io_raster, open_raster
from buteo.vector.io import (
    get_vector_path,
    open_vector,
    internal_vector_to_memory,
    internal_vector_to_metadata,
)
from buteo.utils import (
    file_exists,
    remove_if_overwrite,
    type_check,
)
from buteo.gdal_utils import (
    gdal_bbox_intersects,
    reproject_extent,
    is_raster,
    is_vector,
    path_to_driver,
    default_options,
    translate_resample_method,
    gdal_nodata_value_from_type,
    align_bbox,
)


def internal_clip_raster(
    raster: Union[str, gdal.Dataset],
    clip_geom: Union[str, ogr.DataSource, gdal.Dataset],
    out_path: Optional[str] = None,
    resample_alg: str = "nearest",
    crop_to_geom: bool = True,
    adjust_bbox: bool = True,
    all_touch: bool = True,
    overwrite: bool = True,
    creation_options: list = [],
    dst_nodata: Union[str, int, float] = "infer",
    layer_to_clip: int = 0,
    prefix: str = "",
    postfix: str = "_clipped",
    verbose: int = 1,
    uuid: bool = False,
    ram: int = 8000,
) -> str:
    """OBS: Internal. Single output.

    Clips a raster(s) using a vector geometry or the extents of
    a raster.
    """
    type_check(raster, [str, gdal.Dataset], "raster")
    type_check(clip_geom, [str, ogr.DataSource, gdal.Dataset], "clip_geom")
    type_check(out_path, [str], "out_path", allow_none=True)
    type_check(resample_alg, [str], "resample_alg")
    type_check(crop_to_geom, [bool], "crop_to_geom")
    type_check(adjust_bbox, [bool], "adjust_bbox")
    type_check(all_touch, [bool], "all_touch")
    type_check(dst_nodata, [str, int, float], "dst_nodata")
    type_check(layer_to_clip, [int], "layer_to_clip")
    type_check(overwrite, [bool], "overwrite")
    type_check(creation_options, [list], "creation_options")
    type_check(prefix, [str], "prefix")
    type_check(postfix, [str], "postfix")
    type_check(verbose, [int], "verbose")
    type_check(uuid, [bool], "uuid")

    _, path_list = ready_io_raster(
        raster, out_path, overwrite=overwrite, prefix=prefix, postfix=postfix, uuid=uuid
    )

    # Input is a vector.
    if is_vector(clip_geom):
        clip_ds = open_vector(clip_geom)

        clip_metadata = internal_vector_to_metadata(
            clip_ds, process_layer=layer_to_clip
        )

        if clip_metadata["layer_count"] > 1:
            clip_ds = internal_vector_to_memory(clip_ds, layer_to_extract=layer_to_clip)

        if isinstance(clip_ds, ogr.DataSource):
            clip_ds = clip_ds.GetName()

    # Input is a raster (use extent)
    elif is_raster(clip_geom):
        clip_metadata = internal_raster_to_metadata(clip_geom, create_geometry=True)
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

    raster_metadata = internal_raster_to_metadata(raster)
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
        warpMemoryLimit=ram,
        warpOptions=warp_options,
        srcNodata=raster_metadata["nodata_value"],
        dstNodata=out_nodata,
        multithread=True,
    )

    if verbose == 0:
        gdal.PopErrorHandler()

    if clipped is None:
        raise Exception("Geometries did not intersect.")

    return out_name


def clip_raster(
    raster: Union[List[Union[str, gdal.Dataset]], str, gdal.Dataset],
    clip_geom: Union[str, ogr.DataSource, gdal.Dataset],
    out_path: Union[List[str], str, None] = None,
    resample_alg: str = "nearest",
    crop_to_geom: bool = True,
    adjust_bbox: bool = True,
    all_touch: bool = True,
    overwrite: bool = True,
    creation_options: list = [],
    dst_nodata: Union[str, int, float] = "infer",
    layer_to_clip: int = 0,
    prefix: str = "",
    postfix: str = "_clipped",
    verbose: int = 1,
    uuid: bool = False,
    ram: int = 8000,
) -> Union[list, gdal.Dataset, str]:
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
    type_check(dst_nodata, [str, int, float], "dst_nodata")
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
            internal_clip_raster(
                in_raster,
                clip_geom,
                out_path=path_list[index],
                resample_alg=resample_alg,
                crop_to_geom=crop_to_geom,
                adjust_bbox=adjust_bbox,
                all_touch=all_touch,
                dst_nodata=dst_nodata,
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


if __name__ == "__main__":
    import os
    from buteo.raster.clip import clip_raster
    from buteo.raster.shift import shift_raster
    from glob import glob

    folder = "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/data/ghana/raster/"
    dst = folder + "tmp/"
    # geom = folder + "ghana_buffered_1280.gpkg"
    geom = folder + "2021_B04_10m.tif"

    files = glob(folder + "clipped/*20m.tif")

    paths = clip_raster(
        files,
        geom,
        postfix="",
        creation_options=["COMPRESS=NONE", "TILED=NO"],
        ram="80%",
        all_touch=False,
        adjust_bbox=False,
    )

    for idx, path in enumerate(paths):
        shift_raster(path, [10.0, 0], dst + os.path.basename(paths[idx]))
