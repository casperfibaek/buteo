"""### Clip rasters. ###

Clips a raster using a vector geometry or the extents of a raster.
"""

# Standard library
import os
from typing import Union, List, Optional, Sequence, Tuple, Dict, Any
from warnings import warn

# External
from osgeo import gdal, ogr
import numpy as np

# Internal
from buteo.utils import (
    utils_io,
    utils_gdal,
    utils_base,
    utils_path,
    utils_translate,
)
# Import necessary bbox functions from their new locations
from buteo.bbox.conversion import _get_vector_from_geom, _get_vector_from_bbox
from buteo.bbox.validation import _check_bboxes_intersect, _check_is_valid_bbox
from buteo.bbox.operations import _get_aligned_bbox_to_pixel_size, _get_gdal_bbox_from_ogr_bbox

from buteo.core_raster.core_raster_info import get_metadata_raster
from buteo.core_raster.core_raster_read import open_raster, _open_raster
from buteo.core_raster.core_raster_extent import raster_to_vector_extent as raster_to_extent
from buteo.core_vector.core_vector_read import _open_vector
from buteo.core_vector.core_vector_filter import vector_filter_layer
from buteo.core_vector.core_vector_info import _get_basic_info_vector
from buteo.vector.reproject import _vector_reproject

# Type Aliases
VectorLayer = Union[str, ogr.DataSource, ogr.Layer]
Vector = Union[str, ogr.DataSource]
Raster = Union[str, gdal.Dataset]
VectorOrRaster = Union[str, gdal.Dataset, ogr.DataSource]
ClipGeomType = Union[str, ogr.DataSource, gdal.Dataset, ogr.Geometry]
BboxType = Sequence[Union[int, float]]
RasterList = List[Raster] # More specific type for lists


def _raster_clip(
    raster: Raster,
    clip_geom_processed_path: str,
    clip_metadata: Dict[str, Any],
    out_path: str,
    *,
    resample_alg: str = "nearest",
    crop_to_geom: bool = True,
    adjust_bbox: bool = True,
    all_touch: bool = True,
    creation_options: Optional[List[str]] = None,
    dst_nodata: Optional[Union[float, int, str]] = "infer",
    src_nodata: Optional[Union[float, int, str]] = "infer",
    verbose: int = 1,
    ram: float = 0.8,
    ram_max: Optional[int] = None,
    ram_min: int = 100,
) -> str:
    """INTERNAL. Clips a single raster using a processed vector path."""

    warp_options = []
    if all_touch:
        warp_options.append("CUTLINE_ALL_TOUCHED=TRUE")
    else:
        warp_options.append("CUTLINE_ALL_TOUCHED=FALSE")

    origin_layer = open_raster(raster)
    raster_metadata = get_metadata_raster(raster)

    output_bounds: BboxType = raster_metadata["bbox"]

    if crop_to_geom:
        if adjust_bbox:
            output_bounds = _get_aligned_bbox_to_pixel_size(
                raster_metadata["bbox"],
                clip_metadata["bbox"],
                raster_metadata["pixel_width"],
                raster_metadata["pixel_height"],
            )
        else:
            output_bounds = clip_metadata["bbox"]

    out_format = utils_gdal._get_raster_driver_name_from_path(out_path)
    out_creation_options = utils_gdal._get_default_creation_options(creation_options)

    resolved_src_nodata: Optional[Union[float, int]] = None
    if src_nodata == "infer":
        resolved_src_nodata = raster_metadata["nodata_value"]
    elif isinstance(src_nodata, (int, float)):
        resolved_src_nodata = src_nodata
    elif src_nodata is not None:
        raise ValueError(f"Invalid src_nodata value: {src_nodata}")

    resolved_dst_nodata: Optional[Union[float, int]] = None
    if dst_nodata == "infer":
        resolved_dst_nodata = resolved_src_nodata if resolved_src_nodata is not None else utils_translate._get_default_nodata_value(raster_metadata["dtype_gdal"])
    elif isinstance(dst_nodata, (int, float)):
        resolved_dst_nodata = dst_nodata
    elif dst_nodata is not None:
        raise ValueError(f"Invalid dst_nodata value: {dst_nodata}")

    if resolved_dst_nodata is not None and not utils_translate._check_is_value_within_dtype_range(resolved_dst_nodata, raster_metadata["dtype"]):
        warn(f"Destination nodata value {resolved_dst_nodata} is outside the range of the input raster's dtype. Setting nodata to None.")
        resolved_dst_nodata = None

    if verbose < 2:
        gdal.PushErrorHandler("CPLQuietErrorHandler")
    error_handler_pushed = verbose < 2

    clipped = None
    try:
        options = gdal.WarpOptions(
            format=out_format,
            resampleAlg=utils_translate._translate_resample_method(resample_alg),
            targetAlignedPixels=False,
            outputBounds=_get_gdal_bbox_from_ogr_bbox(output_bounds),
            xRes=raster_metadata["pixel_width"],
            yRes=raster_metadata["pixel_height"],
            cutlineDSName=clip_geom_processed_path,
            cropToCutline=crop_to_geom,
            creationOptions=out_creation_options,
            warpMemoryLimit=utils_gdal._get_dynamic_memory_limit(ram, min_mb=ram_min, max_mb=ram_max),
            warpOptions=warp_options,
            srcNodata=resolved_src_nodata,
            dstNodata=resolved_dst_nodata,
            multithread=True,
        )

        clipped = gdal.Warp(
            out_path,
            origin_layer,
            options=options,
        )
    finally:
        if error_handler_pushed:
            gdal.PopErrorHandler()

    if clipped is None:
        # Attempt cleanup only if file exists
        if os.path.exists(out_path):
             utils_io._delete_if_required(out_path, True) # Corrected overwrite argument
        raise ValueError("Error while clipping raster.")

    return out_path


def raster_clip(
    raster: Union[Raster, List[Raster]],
    clip_geom: ClipGeomType,
    out_path: Optional[Union[str, List[str]]] = None,
    *,
    resample_alg: str = "nearest",
    crop_to_geom: bool = True,
    adjust_bbox: bool = False,
    all_touch: bool = False,
    to_extent: bool = False,
    layer_to_clip: Union[int, str] = 0,
    dst_nodata: Optional[Union[float, int, str]] = "infer",
    src_nodata: Optional[Union[float, int, str]] = "infer",
    creation_options: Optional[List[str]] = None,
    add_uuid: bool = False,
    add_timestamp: bool = False,
    prefix: str = "",
    suffix: str = "",
    overwrite: bool = True,
    verbose: int = 0,
    ram: float = 0.8,
    ram_max: Optional[int] = None,
    ram_min: int = 100,
) -> Union[str, List[str]]:
    """Clips a raster(s) using a vector geometry or the extents of a raster.

    Parameters
    ----------
    raster : Raster | List[Raster]
        The raster(s) to clip.

    clip_geom : ClipGeomType (str | ogr.DataSource | gdal.Dataset | ogr.Geometry)
        The geometry to use to clip the raster. Can be a path to a vector/raster,
        an opened ogr.DataSource/gdal.Dataset, or an ogr.Geometry object.

    out_path : str | list | None, optional
        The path(s) to save the clipped raster to. If None, a memory raster is created. Default: None.

    resample_alg : str, optional
        The resampling algorithm to use. Options include: nearest, bilinear, cubic, cubicspline, lanczos,
        average, mode, max, min, median, q1, q3, sum, rms. Default: "nearest".

    crop_to_geom : bool, optional
        If True, the output raster will be cropped to the extent of the clip geometry. Default: True.

    adjust_bbox : bool, optional
        If True, the output raster will have its bbox adjusted to match the clip geometry's pixel grid. Default: False.

    all_touch : bool, optional
        If true, all pixels touching the clipping geometry will be included. Default: False.

    to_extent : bool, optional
        If True, the clip geometry is converted to its bounding box extent before clipping. Default: False.

    prefix : str, optional
        The prefix to use for the output raster. Default: "".

    suffix : str, optional
        The suffix to use for the output raster. Default: "".

    overwrite : bool, optional
        If True, the output raster will be overwritten if it already exists. Default: True.

    creation_options : list or None, optional
        A list of creation options to pass to gdal. Default: None.

    dst_nodata : int | float | str | None, optional
        The nodata value to use for the output raster. Default: "infer".

    src_nodata : int | float | str | None, optional
        The nodata value to use for the input raster. Default: "infer".

    layer_to_clip : int or str, optional
        The layer ID or name in the vector to use for clipping if clip_geom is a vector dataset. Default: 0.

    verbose : int, optional
        The verbosity level (0=quiet, 1=normal, 2=debug). Default: 0.

    add_uuid : bool, optional
        If True, a UUID will be added to the output raster name. Default: False.

    add_timestamp : bool, optional
        If True, a timestamp will be added to the output raster name. Default: False.

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
    # Type checking using aliases
    utils_base._type_check(raster, [str, gdal.Dataset, list], "raster")
    utils_base._type_check(clip_geom, [str, ogr.DataSource, gdal.Dataset, ogr.Geometry], "clip_geom")
    utils_base._type_check(out_path, [list, str, None], "out_path")
    utils_base._type_check(resample_alg, [str], "resample_alg")
    utils_base._type_check(crop_to_geom, [bool], "crop_to_geom")
    utils_base._type_check(adjust_bbox, [bool], "adjust_bbox")
    utils_base._type_check(all_touch, [bool], "all_touch")
    utils_base._type_check(to_extent, [bool], "to_extent")
    utils_base._type_check(dst_nodata, [str, int, float, None], "dst_nodata")
    utils_base._type_check(src_nodata, [str, int, float, None], "src_nodata")
    utils_base._type_check(layer_to_clip, [int, str], "layer_to_clip")
    utils_base._type_check(overwrite, [bool], "overwrite")
    utils_base._type_check(creation_options, [list, None], "creation_options")
    utils_base._type_check(prefix, [str], "prefix")
    utils_base._type_check(suffix, [str], "suffix")
    utils_base._type_check(verbose, [int], "verbose")
    utils_base._type_check(add_uuid, [bool], "add_uuid")
    utils_base._type_check(add_timestamp, [bool], "add_timestamp")
    utils_base._type_check(ram, [float], "ram")
    utils_base._type_check(ram_max, [int, None], "ram_max")
    utils_base._type_check(ram_min, [int], "ram_min") # Corrected type hint

    input_is_list = isinstance(raster, list)
    # Use more specific type hint for input paths
    in_paths: List[Union[str, gdal.Dataset]] = utils_io._get_input_paths(raster, "raster")
    out_paths: List[str] = utils_io._get_output_paths(
        in_paths, # Pass resolved input paths
        out_path,
        prefix=prefix,
        suffix=suffix,
        add_uuid=add_uuid,
        add_timestamp=add_timestamp,
        change_ext="tif",
    )

    # Check overwrite policy *before* deleting
    utils_io._check_overwrite_policy(out_paths, overwrite)
    utils_io._delete_if_required_list(out_paths, overwrite)

    # Process clip geometry once
    clip_ds_processed_path: Optional[str] = None
    clip_metadata_processed = None
    memory_files_main = [] # Track temp files created here
    opened_clip_vector_ds: Optional[ogr.DataSource] = None # Track opened vector datasource

    try:
        # Handle Geometry input
        if isinstance(clip_geom, ogr.Geometry):
            temp_vec_path = _get_vector_from_geom(clip_geom)
            memory_files_main.append(temp_vec_path)
            opened_clip_vector_ds = _open_vector(temp_vec_path)
            if opened_clip_vector_ds is None: raise ValueError("Could not open temporary vector from geometry.")
            clip_metadata_processed = _get_basic_info_vector(opened_clip_vector_ds)
            clip_ds_processed_path = temp_vec_path

        # Handle Vector path or DataSource input
        elif utils_gdal._check_is_vector(clip_geom):
            # Ensure we have an opened DataSource for metadata/layer checks
            if isinstance(clip_geom, str):
                opened_clip_vector_ds = _open_vector(clip_geom)
                if opened_clip_vector_ds is None: raise ValueError(f"Could not open clip vector: {clip_geom}")
                clip_ds_path_initial: str = clip_geom # Path is string
            elif isinstance(clip_geom, ogr.DataSource):
                opened_clip_vector_ds = clip_geom
                clip_ds_path_initial = opened_clip_vector_ds.GetName() # Get path from DataSource
                if not isinstance(clip_ds_path_initial, str): # Safety check for GetName()
                     raise ValueError("Could not get valid path from clip DataSource.")
            else: # Should not happen based on _check_is_vector
                 raise TypeError(f"Unexpected type for vector clip_geom: {type(clip_geom)}")


            if opened_clip_vector_ds.GetLayerCount() > 1:
                temp_filtered_path = vector_filter_layer(opened_clip_vector_ds, layer_name_or_idx=layer_to_clip, add_uuid=True)
                memory_files_main.append(temp_filtered_path)
                # Close the previous one if it was opened here
                if isinstance(clip_geom, str): opened_clip_vector_ds = None
                opened_clip_vector_ds = _open_vector(temp_filtered_path) # Re-open
                if opened_clip_vector_ds is None: raise ValueError("Could not open filtered vector layer.")
                clip_ds_path_initial = temp_filtered_path # Update path

            clip_metadata_processed = _get_basic_info_vector(opened_clip_vector_ds)

            if to_extent:
                temp_extent_path = _get_vector_from_bbox(clip_metadata_processed["bbox"], clip_metadata_processed["projection_osr"])
                memory_files_main.append(temp_extent_path)
                clip_ds_processed_path = temp_extent_path
            else:
                clip_ds_processed_path = clip_ds_path_initial

        # Handle Raster input (use extent)
        elif utils_gdal._check_is_raster(clip_geom):
            # Ensure clip_geom is str or gdal.Dataset before passing
            if not isinstance(clip_geom, (str, gdal.Dataset)):
                 raise TypeError(f"Expected str or gdal.Dataset for raster clip_geom, got {type(clip_geom)}")
            temp_extent_path = raster_to_extent(clip_geom)
            clip_metadata_processed = get_metadata_raster(clip_geom)
            memory_files_main.append(temp_extent_path)
            clip_ds_processed_path = temp_extent_path
        else:
            if isinstance(clip_geom, str) and not utils_path._check_file_exists(clip_geom):
                raise ValueError(f"Unable to locate clip geometry file: {clip_geom}")
            raise ValueError(f"Unable to parse clip geometry: {clip_geom}")

        if clip_ds_processed_path is None or clip_metadata_processed is None:
             raise ValueError(f"Unable to process clip geometry: {clip_geom}")

        # Reproject clip geometry if necessary (only needs to be done once)
        first_raster_metadata = get_metadata_raster(in_paths[0])
        if not first_raster_metadata["projection_osr"].IsSame(clip_metadata_processed["projection_osr"]):
            # _vector_reproject expects str or ogr.DataSource
            input_for_reproject: Union[str, ogr.DataSource] = opened_clip_vector_ds if opened_clip_vector_ds else clip_ds_processed_path
            if not isinstance(input_for_reproject, (str, ogr.DataSource)):
                 raise TypeError(f"Invalid type for reprojection input: {type(input_for_reproject)}")

            reprojected_clip_path = _vector_reproject(input_for_reproject, first_raster_metadata["projection_osr"])

            if clip_ds_processed_path not in memory_files_main and reprojected_clip_path != clip_ds_processed_path:
                 memory_files_main.append(reprojected_clip_path)
            clip_ds_processed_path = reprojected_clip_path

            # Update metadata after reprojection
            opened_clip_vector_ds_reproj = _open_vector(clip_ds_processed_path)
            if opened_clip_vector_ds_reproj is None: raise ValueError("Could not open reprojected clip vector.")
            clip_metadata_processed = _get_basic_info_vector(opened_clip_vector_ds_reproj)
            opened_clip_vector_ds_reproj = None # Close immediately

        # Clip each raster
        output = []
        for index, in_raster_path in enumerate(in_paths):
            output.append(
                _raster_clip(
                    in_raster_path,
                    clip_ds_processed_path,
                    clip_metadata_processed,
                    out_path=out_paths[index],
                    resample_alg=resample_alg,
                    crop_to_geom=crop_to_geom,
                    adjust_bbox=adjust_bbox,
                    all_touch=all_touch,
                    creation_options=creation_options,
                    dst_nodata=dst_nodata,
                    src_nodata=src_nodata,
                    verbose=verbose,
                    ram=ram,
                    ram_max=ram_max,
                    ram_min=ram_min,
                )
            )

    finally:
        # Clean up all temporary files created in this main function
        utils_gdal.delete_dataset_if_in_memory_list(memory_files_main)
        # Explicitly close any opened vector datasource
        if opened_clip_vector_ds is not None:
            opened_clip_vector_ds = None


    if input_is_list:
        return output

    return output[0]
