"""
### Clip rasters ###

Clips a raster using a vector geometry or the extents of a raster.
"""

# Standard library
import sys; sys.path.append("../../")
from typing import Union, List, Optional
from warnings import warn

# External
from osgeo import gdal, ogr
import numpy as np

# Internal
from buteo.utils import (
    utils_io,
    utils_gdal,
    utils_base,
    utils_bbox,
    utils_path,
    utils_translate,
)
from buteo.raster import core_raster
from buteo.vector import core_vector
from buteo.vector.reproject import _vector_reproject


def _raster_clip(
    raster: Union[str, List[str], gdal.Dataset, List[gdal.Dataset]],
    clip_geom: Union[str, ogr.DataSource, ogr.Layer, List[ogr.Layer], List[ogr.DataSource]],
    out_path: Optional[Union[str, List[str]]] = None,
    *,
    resample_alg: str = "nearest",
    crop_to_geom: bool = True,
    adjust_bbox: bool = True,
    all_touch: bool = True,
    to_extent: bool = False,
    overwrite: bool = True,
    creation_options: Optional[List] = None,
    dst_nodata: str = "infer",
    src_nodata: str = "infer",
    layer_to_clip: int = 0,
    prefix: str = "",
    suffix: str = "",
    verbose: int = 1,
    ram: float = 0.8,
    ram_max: Optional[int] = None,
    ram_min: Optional[int] = 100,
):
    """
    INTERNAL.
    Clips a raster(s) using a vector geometry or the extents of a raster.
    """
    path_list = utils_io._get_output_paths(
        raster,
        out_path,
        prefix=prefix,
        suffix=suffix,
        add_uuid=out_path is None,
    )
    assert utils_path._check_is_valid_output_path_list(path_list, overwrite), (
        f"Unable to parse out_path: {out_path}"
    )

    clip_ds = None

    memory_files = []

    if isinstance(clip_geom, ogr.Geometry):
        clip_geom = utils_bbox._get_vector_from_geom(clip_geom)
        clip_geom = utils_gdal._get_path_from_dataset(clip_geom)

    # Input is a vector.
    if utils_gdal._check_is_vector(clip_geom):
        clip_ds = core_vector._vector_open(clip_geom)

        # TODO: Fix potential memory leak
        if clip_ds.GetLayerCount() > 1:
            clip_ds = core_vector.vector_filter_layer(clip_ds, layer_name_or_idx=layer_to_clip, add_uuid=True)
            memory_files.append(clip_ds)

        clip_metadata = core_vector._get_basic_metadata_vector(clip_ds)

        if to_extent:
            clip_ds = utils_bbox._get_vector_from_bbox(clip_metadata["bbox"], clip_metadata["projection_osr"])
            memory_files.append(clip_ds)

        if isinstance(clip_ds, ogr.DataSource):
            clip_ds = clip_ds.GetName()

    # Input is a raster (use extent)
    elif utils_gdal._check_is_raster(clip_geom):
        clip_metadata = core_raster._get_basic_metadata_raster(clip_geom)
        clip_ds = utils_bbox._get_vector_from_bbox(clip_metadata["bbox"], clip_metadata["projection_osr"])
        memory_files.append(clip_ds)
    else:
        if utils_path._check_file_exists(clip_geom):
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

    origin_layer = core_raster.raster_open(raster)

    raster_metadata = core_raster._get_basic_metadata_raster(raster)
    origin_projection = raster_metadata["projection_osr"]

    # Fast check: Does the extent of the two inputs overlap?
    has_inf = True in [np.isinf(val) for val in raster_metadata["bbox_latlng"]]
    if not has_inf and not utils_bbox._check_bboxes_intersect(raster_metadata["bbox_latlng"], clip_metadata["bbox_latlng"]):
        raise ValueError(f"Geometries of {raster} and {clip_geom} did not intersect.")

    if not origin_projection.IsSame(clip_metadata["projection_osr"]):
        clip_ds = _vector_reproject(clip_ds, origin_projection)
        clip_metadata = core_vector._get_basic_metadata_vector(clip_ds)
        memory_files.append(clip_ds)

    output_bounds = raster_metadata["bbox"]

    if crop_to_geom:

        if adjust_bbox:
            output_bounds = utils_bbox._get_aligned_bbox_to_pixel_size(
                raster_metadata["bbox"],
                clip_metadata["bbox"],
                raster_metadata["pixel_width"],
                raster_metadata["pixel_height"],
            )
        else:
            output_bounds = clip_metadata["bbox"]

    # formats
    out_name = path_list[0]
    out_format = utils_gdal._get_raster_driver_name_from_path(out_name)
    out_creation_options = utils_gdal._get_default_creation_options(creation_options)

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
            out_nodata = utils_translate._get_default_nodata_value(raster_metadata["dtype_gdal"])
    elif isinstance(dst_nodata, (int, float)) or dst_nodata is None:
        out_nodata = dst_nodata
    else:
        raise ValueError(f"Unable to parse nodata_value: {dst_nodata}")

    if not utils_translate._check_is_value_within_dtype_range(out_nodata, raster_metadata["dtype"]):
        warn("Nodata value is outside of the range of the input raster's dtype. Setting to None.")
        out_nodata = None

    # Removes file if it exists and overwrite is True.
    utils_path._delete_if_required(out_path, overwrite)

    if verbose == 0:
        gdal.PushErrorHandler("CPLQuietErrorHandler")

    options = gdal.WarpOptions(
        format=out_format,
        resampleAlg=utils_translate._translate_resample_method(resample_alg),
        targetAlignedPixels=False,
        outputBounds=utils_bbox._get_gdal_bbox_from_ogr_bbox(output_bounds),
        xRes=raster_metadata["pixel_width"],
        yRes=raster_metadata["pixel_height"],
        cutlineDSName=clip_ds,
        cropToCutline=False,
        creationOptions=out_creation_options,
        warpMemoryLimit=utils_gdal._get_dynamic_memory_limit(ram, min_mb=ram_min, max_mb=ram_max),
        warpOptions=warp_options,
        srcNodata=src_nodata,
        dstNodata=out_nodata,
        multithread=True,
    )

    clipped = gdal.Warp(
        out_name,
        origin_layer,
        options=options,
    )

    utils_gdal.delete_dataset_if_in_memory_list(memory_files)

    if verbose == 0:
        gdal.PopErrorHandler()

    if clipped is None:
        raise ValueError("Error while clipping raster.")

    return out_name


def raster_clip(
    raster: Union[str, gdal.Dataset, List[Union[str, gdal.Dataset]]],
    clip_geom: Union[str, ogr.DataSource, gdal.Dataset, ogr.Geometry],
    out_path: Optional[str] = None,
    *,
    resample_alg: str = "nearest",
    crop_to_geom: bool = True,
    adjust_bbox: bool = False,
    all_touch: bool = False,
    to_extent: bool = False,
    layer_to_clip: int = 0,
    dst_nodata: Union[float, int, str] = "infer",
    src_nodata: Union[float, int, str] = "infer",
    creation_options: Optional[List[str]] = None,
    add_uuid: bool = False,
    add_timestamp: bool = False,
    prefix: str = "",
    suffix: str = "",
    overwrite: bool = True,
    verbose: int = 0,
    ram: float = 0.8,
    ram_max: Optional[int] = None,
    ram_min: Optional[int] = 100,
):
    """
    Clips a raster(s) using a vector geometry or the extents of a raster.

    Parameters
    ----------
    raster : str or gdal.Dataset or list of str/gdal.Dataset
        The raster(s) to clip.

    clip_geom : str or ogr.DataSource, gdal.Dataset or ogr.Geometry
        The geometry to use to clip the raster.

    out_path : str or list or None, optional
        The path(s) to save the clipped raster to. If None, a memory raster is created. Default: None.

    resample_alg : str, optional
        The resampling algorithm to use. Options include: nearest, bilinear, cubic, cubicspline, lanczos, 
        average, mode, max, min, median, q1, q3, sum, rms. Default: "nearest".

    crop_to_geom : bool, optional
        If True, the output raster will be cropped to the extent of the clip geometry. Default: True.

    adjust_bbox : bool, optional
        If True, the output raster will have its bbox adjusted to match the clip geometry. Default: False.

    all_touch : bool, optional
        If true, all pixels touching the clipping geometry will be included. Default: False.

    to_extent : bool, optional
        If True, the output raster will be cropped to the extent of the clip geometry. Default: False.

    prefix : str, optional
        The prefix to use for the output raster. Default: "".

    suffix : str, optional
        The suffix to use for the output raster. Default: "".

    overwrite : bool, optional
        If True, the output raster will be overwritten if it already exists. Default: True.

    creation_options : list or None, optional
        A list of creation options to pass to gdal. Default: None.

    dst_nodata : int or float or None, optional
        The nodata value to use for the output raster. Default: "infer".

    src_nodata : int or float or None, optional
        The nodata value to use for the input raster. Default: "infer".

    layer_to_clip : int or str, optional
        The layer ID or name in the vector to use for clipping. Default: 0.

    verbose : int, optional
        The verbosity level. Default: 0.

    add_uuid : bool, optional
        If True, a UUID will be added to the output raster. Default: False.

    add_timestamp : bool, optional
        If True, a timestamp will be added to the output raster. Default: False.

    ram : float, optional
        The proportion of total ram to allow usage of. Default: 0.8.
    
    ram_max: int, optional
        The maximum amount of ram to use in MB. Default: None.
    
    ram_min: int, optional
        The minimum amount of ram to use in MB. Default: 100.

    Returns
    -------
    str or list
        A string or list of strings representing the path(s) to the clipped raster(s).
    """
    utils_base._type_check(raster, [str, gdal.Dataset, [str, gdal.Dataset]], "raster")
    utils_base._type_check(clip_geom, [str, ogr.DataSource, gdal.Dataset, ogr.Geometry], "clip_geom")
    utils_base._type_check(out_path, [[str], str, None], "out_path")
    utils_base._type_check(resample_alg, [str], "resample_alg")
    utils_base._type_check(crop_to_geom, [bool], "crop_to_geom")
    utils_base._type_check(adjust_bbox, [bool], "adjust_bbox")
    utils_base._type_check(all_touch, [bool], "all_touch")
    utils_base._type_check(to_extent, [bool], "to_extent")
    utils_base._type_check(dst_nodata, [str, int, float, None], "dst_nodata")
    utils_base._type_check(src_nodata, [str, int, float, None], "src_nodata")
    utils_base._type_check(layer_to_clip, [int], "layer_to_clip")
    utils_base._type_check(overwrite, [bool], "overwrite")
    utils_base._type_check(creation_options, [[str], None], "creation_options")
    utils_base._type_check(prefix, [str], "prefix")
    utils_base._type_check(suffix, [str], "postfix")
    utils_base._type_check(verbose, [int], "verbose")
    utils_base._type_check(add_uuid, [bool], "uuid")
    utils_base._type_check(add_timestamp, [bool], "timestamp")
    utils_base._type_check(ram, [float], "ram")
    utils_base._type_check(ram_max, [int, None], "ram_max")
    utils_base._type_check(ram_min, [int, None], "ram_min")

    input_is_list = isinstance(raster, list)
    input_rasters = utils_io._get_input_paths(raster, "raster")
    out_path_list = utils_io._get_output_paths(
        raster,
        out_path,
        prefix=prefix,
        suffix=suffix,
        add_uuid=add_uuid,
        add_timestamp=add_uuid,
        change_ext="tif",
        overwrite=overwrite,
    )
    utils_path._delete_if_required_list(out_path_list, overwrite)

    output = []
    for index, in_raster in enumerate(input_rasters):
        output.append(
            _raster_clip(
                in_raster,
                clip_geom,
                out_path=out_path_list[index],
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
                ram_max=ram_max,
                ram_min=ram_min,
            )
        )

    if input_is_list:
        return output

    return output[0]
