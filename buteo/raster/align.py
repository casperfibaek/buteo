"""### Align rasters. ###

Functions to align a series of rasters to a master or a reference.
"""

# TODO: phase_cross_correlation

# Standard library
from typing import List, Union, Optional, Sequence, Tuple, Dict, Any

# External
from osgeo import gdal, ogr

# Internal
from buteo.utils import (
    utils_io,
    utils_path,
    utils_gdal,
    utils_base,
    utils_translate,
    utils_projection,
)
# Import necessary bbox functions from their new locations
from buteo.bbox.operations import _get_aligned_bbox_to_pixel_size, _get_gdal_bbox_from_ogr_bbox
from buteo.bbox.conversion import _get_geom_from_bbox

from buteo.core_raster.core_raster_info import get_metadata_raster
from buteo.core_raster.core_raster_extent import check_rasters_intersect
from buteo.core_raster.core_raster_read import _open_raster, open_raster
from buteo.core_raster.core_raster_write import raster_create_empty
from buteo.raster.reproject import raster_reproject

# Type Aliases
Raster = Union[str, gdal.Dataset]
BboxType = Sequence[Union[int, float]]
RasterList = List[Raster]


def _raster_align_to_reference(
    rasters: Union[Raster, List[Raster]],
    reference: Raster,
    *,
    out_path: Optional[Union[str, List[str]]] = None,
    resample_alg: str = "nearest",
    target_nodata: Optional[Union[int, float]] = None,
    overwrite=True,
    creation_options: Optional[List[str]] = None,
    prefix: str = "",
    suffix: str = "",
    add_uuid: bool = False,
    add_timestamp: bool = False,
    ram: float = 0.8,
    ram_max: Optional[int] = None,
    ram_min: int = 100,
) -> List[str]:
    """Aligns a series of rasters to a reference.

    Parameters
    ----------
    rasters : Raster | List[Raster]
        A list of rasters to align.
    reference : Raster
        Path to the reference raster or vector.
    out_path : str or list of str, optional
        Paths to the output. If not provided, the output will be in-memory rasters.
    resample_alg : str, optional
        Resampling algorithm to use. Default: "nearest".
    target_nodata : int or float, optional
        Nodata value to use for the output rasters.
    overwrite : bool, optional
        Overwrite existing files. Default: True.
    creation_options : list, optional
        List of creation options.
    prefix : str, optional
        Prefix to add to the output file name. Default: "".
    suffix : str, optional
        Suffix to add to the output file name. Default: "".
    add_uuid : bool, optional
        Whether to add a uuid to the output file name. Default: False.
    add_timestamp : bool, optional
        Whether to add a timestamp to the output file name. Default: False.
    ram : float, optional
        The proportion of total ram to allow usage of. Default: 0.8.
    ram_max: int, optional
        The maximum amount of ram to use in MB. Default: None.
    ram_min: int, optional
        The minimum amount of ram to use in MB. Default: 100.

    Returns
    -------
    List[str]
        A list of paths to the aligned rasters.
    """
    utils_base._type_check(rasters, [str, gdal.Dataset, list], "rasters")
    utils_base._type_check(reference, [str, gdal.Dataset], "reference")
    utils_base._type_check(out_path, [str, None, list], "out_path")
    utils_base._type_check(resample_alg, [str], "resample_alg")
    utils_base._type_check(target_nodata, [int, float, None], "target_nodata")
    utils_base._type_check(overwrite, [bool], "overwrite")
    utils_base._type_check(creation_options, [list, None], "creation_options")
    utils_base._type_check(prefix, [str], "prefix")
    utils_base._type_check(suffix, [str], "suffix")
    utils_base._type_check(add_uuid, [bool], "add_uuid")
    utils_base._type_check(add_timestamp, [bool], "add_timestamp")
    utils_base._type_check(ram, [float], "ram")
    utils_base._type_check(ram_max, [int, None], "ram_max")
    utils_base._type_check(ram_min, [int], "ram_min")

    input_rasters: List[Union[str, gdal.Dataset]] = utils_io._get_input_paths(rasters, "raster") # type: ignore

    assert utils_gdal._check_is_raster_or_vector(reference), "Reference must be a raster or vector."

    # Verify that all of the rasters overlap the reference
    for raster in input_rasters:
        assert check_rasters_intersect(raster, reference), (
            f"Raster: {utils_gdal._get_path_from_dataset(raster)} does not intersect reference."
        )

    path_list: List[str] = utils_io._get_output_paths(
        input_rasters, # type: ignore
        out_path,
        prefix=prefix,
        suffix=suffix,
        add_uuid=add_uuid,
        add_timestamp=add_timestamp,
        change_ext="tif",
    )

    # Check overwrite policy *before* deleting
    utils_io._check_overwrite_policy(path_list, overwrite)
    utils_io._delete_if_required_list(path_list, overwrite)

    creation_options = utils_gdal._get_default_creation_options(creation_options)

    reference_metadata = get_metadata_raster(reference)

    target_projection = reference_metadata["projection_osr"]
    reference_bbox = reference_metadata["bbox"]
    x_min, _, _, y_max = reference_bbox

    # Set the target values.
    x_res = reference_metadata["pixel_width"]
    y_res = reference_metadata["pixel_height"]
    x_pixels = reference_metadata["width"]
    y_pixels = reference_metadata["height"]

    # Reproject the rasters to the reference projection.
    for idx, raster in enumerate(input_rasters):
        raster_metadata = get_metadata_raster(raster)
        raster_path = utils_gdal._get_path_from_dataset(raster)

        raster_reprojected = None
        raster_projection = raster_metadata["projection_osr"]
        if not utils_projection._check_projections_match(raster_projection, target_projection):
            raster_reprojected = raster_reproject(
                raster_path,
                target_projection,
                out_path=None,
                resample_alg=resample_alg,
                creation_options=creation_options,
                prefix="tmp_reprojection_",
                add_uuid=True,
            )
            raster_ds = _open_raster(raster_reprojected) # type: ignore
        else:
            raster_ds = _open_raster(raster_path)

        # Ensure raster_ds is opened
        if raster_ds is None:
            raise ValueError(f"Could not open raster: {raster_path or raster_reprojected}")

        destination_ds = raster_create_empty(
            out_path=path_list[idx],
            width=x_pixels,
            height=y_pixels,
            pixel_size=(x_res, y_res), # type: ignore # Pass as tuple
            x_min=x_min,
            y_max=y_max,
            dtype=raster_metadata["dtype"],
            bands=raster_metadata["bands"],
            nodata_value=target_nodata,
            projection=target_projection,
            creation_options=creation_options,
        )

        destination_ds = _open_raster(destination_ds, writeable=True)
        if destination_ds is None:
             raise ValueError(f"Could not create empty destination raster: {path_list[idx]}")

        warp_options = gdal.WarpOptions(
            resampleAlg=utils_translate._translate_resample_method(resample_alg),
            multithread=True,
            outputBounds=_get_gdal_bbox_from_ogr_bbox(reference_bbox),
            warpMemoryLimit=utils_gdal._get_dynamic_memory_limit(ram, min_mb=ram_min, max_mb=ram_max),
        )

        warped = gdal.Warp(destination_ds, raster_ds, options=warp_options)

        if raster_reprojected is not None:
            # Ensure raster_reprojected is a string path before deleting
            if isinstance(raster_reprojected, str):
                 utils_gdal.delete_dataset_if_in_memory(raster_reprojected)
            else: # Should be gdal.Dataset if not None and not str
                 raster_reprojected = None # Dereference to close

        destination_ds.FlushCache()
        destination_ds = None
        raster_ds = None # Close source dataset

        if warped is None:
            raise ValueError("Error while warping rasters.")

    return path_list


def _raster_find_best_align_reference(
    rasters: Union[Raster, List[Raster]],
    method: str,
    out_path: Optional[str] = None,
) -> str:
    """Find the best reference raster for aligning a list of rasters.

    Parameters
    ----------
    rasters : Raster | List[Raster]
        List of rasters to align.
    method : str
        Bounding box method to use for finding the best reference.
        Options include: "reference", "intersection", and "union".
    out_path : str, optional
        Path to the output raster, default: None

    Returns
    -------
    str
        Path to the best reference raster.
    """
    utils_base._type_check(rasters, [str, gdal.Dataset, list], "rasters")
    utils_base._type_check(method, [str], "method")
    utils_base._type_check(out_path, [str, None], "out_path")

    assert method in ["reference", "intersection", "union"], (
        f"Invalid bounding_box: {method}. Options include: reference, intersection, and union."
    )

    input_rasters: List[Union[str, gdal.Dataset]] = utils_io._get_input_paths(rasters, "raster") # type: ignore

    if len(input_rasters) == 1:
        # If only one raster, return its path
        return utils_gdal._get_path_from_dataset(input_rasters[0])

    if out_path is None:
        out_path = utils_path._get_temp_filepath(name="best_reference.tif")
    else:
        if not utils_path._check_is_valid_output_filepath(out_path, overwrite=True):
            raise ValueError(f"Invalid out_path: {out_path}")

    # Count intersections
    most_intersections = 0
    intersections_arr = []
    for idx_i, raster_i in enumerate(input_rasters):
        intersections = 0
        for idx_j, raster_j in enumerate(input_rasters):
            if idx_i == idx_j:
                continue

            # Uses the latlng bounding box to determine if the rasters intersect.
            intersection = check_rasters_intersect(raster_i, raster_j)

            if intersection:
                intersections += 1

        if intersections > most_intersections:
            most_intersections = intersections

        intersections_arr.append(intersections)

    # Find the raster with the most intersections.
    largest_area = 0.0
    largest_area_idx = 0
    for idx, intersection_count in enumerate(intersections_arr):
        if intersection_count == most_intersections:
            raster = input_rasters[idx]
            raster_area = get_metadata_raster(raster)["area_latlng"]

            if raster_area > largest_area:
                largest_area = raster_area
                largest_area_idx = idx

    best_reference = utils_gdal._get_path_from_dataset(input_rasters[largest_area_idx])

    if method == "reference":
        return best_reference

    # Prepare for intersection/union methods
    input_path_list = [utils_gdal._get_path_from_dataset(r) for r in input_rasters]

    all_other_rasters = []
    for raster_path in input_path_list:
        if raster_path == best_reference:
            continue
        all_other_rasters.append(raster_path)

    # Ensure all other rasters intersect the chosen best reference
    for raster in all_other_rasters:
        assert check_rasters_intersect(best_reference, raster), (
            f"Rasters {best_reference} and {raster} do not intersect."
        )

    # Reproject other rasters to match the best reference
    matched_rasters = raster_reproject(
        all_other_rasters,
        best_reference,
        copy_if_same=False,
    )

    best_reference_meta = get_metadata_raster(best_reference)

    # Calculate union or intersection geometry
    # Need to get geometry from metadata, assuming it exists
    best_geom_wkt = best_reference_meta.get("bounds")
    if best_geom_wkt is None: raise ValueError("Could not get bounds WKT from best reference metadata.")
    best_geom = ogr.CreateGeometryFromWkt(best_geom_wkt)
    if best_geom is None: raise ValueError("Could not create geometry from best reference bounds WKT.")

    for raster_path in matched_rasters:
        raster_meta = get_metadata_raster(raster_path)
        raster_geom_wkt = raster_meta.get("bounds")
        if raster_geom_wkt is None: raise ValueError(f"Could not get bounds WKT from {raster_path} metadata.")
        raster_geom = ogr.CreateGeometryFromWkt(raster_geom_wkt)
        if raster_geom is None: raise ValueError(f"Could not create geometry from {raster_path} bounds WKT.")


        if not raster_geom.Intersects(best_geom):
            raise ValueError(
                f"Raster {raster_path} did not intersect. Consider not using bbox='intersection'"
            )

        if method == "union":
            best_geom = raster_geom.Union(best_geom)
        else: # intersection
            best_geom = raster_geom.Intersection(best_geom)

    # Get envelope of the final geometry
    bbox_best_geom: Tuple[float, float, float, float] = best_geom.GetEnvelope()
    # Convert tuple to list for _get_aligned_bbox_to_pixel_size
    bbox_best_geom_list: BboxType = [bbox_best_geom[0], bbox_best_geom[1], bbox_best_geom[2], bbox_best_geom[3]]

    # Clean up temporary reprojected rasters
    for raster_path in matched_rasters:
        if raster_path not in input_path_list: # Only delete if it was created temporarily
            utils_gdal.delete_dataset_if_in_memory(raster_path)

    # Align the final geometry bbox to the pixel grid of the best reference
    outbox_bbox = _get_aligned_bbox_to_pixel_size(
        best_reference_meta["bbox"],
        bbox_best_geom_list,
        best_reference_meta["pixel_width"],
        best_reference_meta["pixel_height"],
    )

    creation_options = utils_gdal._get_default_creation_options()

    width = int(round((outbox_bbox[1] - outbox_bbox[0]) / best_reference_meta["pixel_width"]))
    height = int(round((outbox_bbox[3] - outbox_bbox[2]) / abs(best_reference_meta["pixel_height"]))) # Use abs height

    # Create the empty reference raster based on the calculated aligned bbox
    out_path = raster_create_empty(
        out_path=out_path,
        width=width,
        height=height,
        pixel_size=(best_reference_meta["pixel_width"], best_reference_meta["pixel_height"]), # type: ignore # Corrected keyword usage
        x_min=outbox_bbox[0],
        y_max=outbox_bbox[3],
        projection=best_reference_meta["projection_osr"],
        dtype=best_reference_meta["dtype"],
        bands=1,
        creation_options=creation_options,
    )

    return out_path


def raster_align(
    rasters: Union[Raster, List[Raster]],
    *,
    out_path: Optional[Union[str, List[str]]] = None,
    reference: Optional[Raster] = None,
    method: str = "reference",
    target_nodata: Optional[Union[int, float]] = None,
    resample_alg: str = "nearest",
    overwrite: bool = True,
    creation_options: Optional[List[str]] = None,
    prefix: str = "",
    suffix: str = "",
    add_uuid: bool = False,
    add_timestamp: bool = False,
    ram: float = 0.8,
    ram_min: int = 100,
    ram_max: Optional[int] = None,
) -> List[str]:
    """Aligns rasters either to a reference raster or to each other using one of three methods:
    reference, intersection, or union.

    Parameters
    ----------
    rasters : Raster | List[Raster]
        The rasters to align.
    out_path : List[str] or str, optional
        The output path(s), default: None
    reference : Raster, optional
        The reference raster to align to, default: None
    method : str, optional
        The method to use if no reference is provided, default: "reference"
        [reference, intersection, union]
    resample_alg : str, optional
        The resampling algorithm to use, default: "nearest"
    overwrite : bool, optional
        Whether to overwrite existing files, default: True
    creation_options : Optional[List[str]], optional
        The creation options to use, default: None
    prefix : str, optional
        The prefix to add to the output file name, default: ""
    suffix : str, optional
        The suffix to add to the output file name, default: ""
    add_uuid : bool, optional
        Whether to add a uuid to the output file name, default: False
    add_timestamp : bool, optional
        Whether to add a timestamp to the output file name, default: False
    ram : float, optional
        The proportion of total ram to allow usage of, default: 0.8
    ram_min : int, optional
        The minimum amount of RAM to use in MB, default: 100
    ram_max : int, optional
        The maximum amount of RAM to use in MB, default: None

    Returns
    -------
    List[str]
        The aligned rasters.
    """
    utils_base._type_check(rasters, [str, gdal.Dataset, list], "rasters")
    utils_base._type_check(out_path, [str, None, list], "out_path")
    utils_base._type_check(reference, [str, gdal.Dataset, None], "reference")
    utils_base._type_check(method, [str], "method")
    utils_base._type_check(target_nodata, [int, float, None], "target_nodata")
    utils_base._type_check(resample_alg, [str], "resample_alg")
    utils_base._type_check(overwrite, [bool], "overwrite")
    utils_base._type_check(creation_options, [list, None], "creation_options")
    utils_base._type_check(prefix, [str], "prefix")
    utils_base._type_check(suffix, [str], "suffix")
    utils_base._type_check(add_uuid, [bool], "add_uuid")
    utils_base._type_check(add_timestamp, [bool], "add_timestamp")
    utils_base._type_check(ram, [float], "ram")
    utils_base._type_check(ram_min, [int], "ram_min")
    utils_base._type_check(ram_max, [int, None], "ram_max")
    assert method in ["reference", "intersection", "union"], "method must be one of reference, intersection, or union."

    raster_list: List[Union[str, gdal.Dataset]] = utils_io._get_input_paths(rasters, "raster") # type: ignore

    path_list: List[str] = utils_io._get_output_paths(
        raster_list, # type: ignore
        out_path,
        prefix=prefix,
        suffix=suffix,
        add_uuid=add_uuid,
        add_timestamp=add_timestamp,
        change_ext="tif",
    )

    internal_reference: Optional[str] = None # Track if reference was created internally
    if reference is None:
        internal_reference = _raster_find_best_align_reference(raster_list, method)
        reference_input: Raster = internal_reference # Use the created reference path
    else:
        reference_input = reference # Use the provided reference

    try:
        aligned = _raster_align_to_reference(
            raster_list,
            reference_input,
            out_path=path_list,
            resample_alg=resample_alg,
            target_nodata=target_nodata,
            overwrite=overwrite,
            creation_options=creation_options,
            # suffix=suffix, # Not needed for internal call
            # prefix=prefix, # Not needed for internal call
            ram=ram,
            ram_min=ram_min,
            ram_max=ram_max,
        )
    finally:
        # Clean up internally created reference if it exists
        if internal_reference is not None:
            utils_io._delete_if_required(internal_reference, True)

    return aligned
