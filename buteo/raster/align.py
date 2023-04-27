"""
### Align rasters ###

Functions to align a series of rasters to a master or a reference.
"""

# TODO: phase_cross_correlation

# Standard library
import sys; sys.path.append("../../")
from typing import List, Union, Optional
from uuid import uuid4

# External
from osgeo import gdal

# Internal
from buteo.utils import core_utils, gdal_enums, gdal_utils, bbox_utils
from buteo.raster import core_raster
from buteo.raster.reproject import reproject_raster, match_raster_projections


def align_rasters_to_reference(
    rasters: Union[str, gdal.Dataset, List[Union[str, gdal.Dataset]]],
    reference: Union[str, gdal.Dataset],
    *,
    out_path: Optional[Union[str, List[str]]] = None,
    resample_alg: str = "nearest",
    target_nodata: Optional[Union[int, float]] = None,
    overwrite=True,
    creation_options: Optional[List[str]] = None,
    prefix: str = "",
    suffix: str = "",
    ram: Union[str, float, int] = "auto",
) -> List[str]:
    """
    Aligns a series of rasters to a reference.

    Parameters
    ----------
    rasters : list of str
        A list of rasters to align.

    reference : str or gdal.Dataset
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

    ram : str or int, optional
        Amount of RAM to use in MB. If "auto", the amount of RAM will be determined automatically.

    Returns
    -------
    List[str]
        A list of paths to the aligned rasters.
    """
    core_utils.type_check(rasters, [str, gdal.Dataset, [str, gdal.Dataset]], "rasters")
    core_utils.type_check(reference, [str, gdal.Dataset], "reference")
    core_utils.type_check(out_path, [str, None, [str]], "out_path")
    core_utils.type_check(resample_alg, [str], "resample_alg")
    core_utils.type_check(target_nodata, [int, float, None], "target_nodata")
    core_utils.type_check(overwrite, [bool], "overwrite")
    core_utils.type_check(creation_options, [[str], None], "creation_options")
    core_utils.type_check(prefix, [str], "prefix")
    core_utils.type_check(suffix, [str], "suffix")
    core_utils.type_check(ram, [int, str], "ram")

    rasters_list = core_utils.ensure_list(rasters)
    assert gdal_utils.is_raster_or_vector(reference), "Reference must be a raster or vector."

    add_uuid = True if out_path is None else False

    # Verify that all of the rasters overlap the reference
    for raster in rasters_list:
        assert core_raster.rasters_intersect(raster, reference), (
            f"Raster: {gdal_utils.get_path_from_dataset(raster)} does not intersect reference."
        )

    path_list = gdal_utils.create_output_path_list(
        rasters_list,
        out_path,
        overwrite=overwrite,
        add_uuid=add_uuid,
        ext=".tif",
        prefix=prefix,
        suffix=suffix,
    )

    creation_options = gdal_utils.default_creation_options(creation_options)

    reference_metadata = core_raster.raster_to_metadata(reference)

    target_projection = reference_metadata["projection_osr"]
    reference_bbox = reference_metadata["bbox"]
    x_min, _, _, y_max = reference_bbox

    # Set the target values.
    x_res = reference_metadata["pixel_width"]
    y_res = reference_metadata["pixel_height"]
    x_pixels = reference_metadata["width"]
    y_pixels = reference_metadata["height"]

    # Reproject the rasters to the reference projection.
    for idx, raster in enumerate(rasters_list):
        raster_metadata = core_raster.raster_to_metadata(raster)
        raster_path = gdal_utils.get_path_from_dataset(raster)

        raster_reprojected = None
        raster_projection = raster_metadata["projection_osr"]
        if not gdal_utils.projections_match(raster_projection, target_projection):
            raster_reprojected = reproject_raster(
                raster_path,
                target_projection,
                out_path=None,
                resample_alg=resample_alg,
                creation_options=creation_options,
                prefix="tmp_reprojection_",
                add_uuid=True,
            )
            raster_ds = core_raster.open_raster(raster_reprojected)
        else:
            raster_ds = core_raster.open_raster(raster_path)

        destination_ds = core_raster.create_empty_raster(
            out_path=path_list[idx],
            width=x_pixels,
            height=y_pixels,
            pixel_size=(x_res, y_res),
            x_min=x_min,
            y_max=y_max,
            dtype=raster_metadata["dtype"],
            bands=raster_metadata["band_count"],
            nodata_value=target_nodata,
            projection=target_projection,
            creation_options=creation_options,
            overwrite=overwrite,
        )

        destination_ds = core_raster.open_raster(destination_ds)

        warp_options = gdal.WarpOptions(
            resampleAlg=gdal_enums.translate_resample_method(resample_alg),
            multithread=True,
            outputBounds=bbox_utils.convert_ogr_bbox_to_gdal_bbox(reference_bbox),
            warpMemoryLimit=gdal_utils.get_gdalwarp_ram_limit(ram),
        )

        warped = gdal.Warp(destination_ds, raster_ds, options=warp_options)

        if raster_reprojected is not None:
            gdal_utils.delete_if_in_memory(raster_reprojected)

        destination_ds.FlushCache()
        destination_ds = None

        if warped is None:
            raise ValueError("Error while warping rasters.")

    return path_list


def find_best_reference(
    rasters: Union[str, gdal.Dataset, List[Union[str, gdal.Dataset]]],
    method: str,
) -> str:
    """
    Find the best reference raster for aligning a list of rasters.

    Parameters
    ----------
    rasters : Union[str, gdal.Dataset, List[Union[str, gdal.Dataset]]]
        List of rasters to align.

    method : str
        Bounding box method to use for finding the best reference.
        Options include: "reference", "intersection", and "union".

    Returns
    -------
    str
        Path to the best reference raster.
    """
    core_utils.type_check(rasters, [str, gdal.Dataset, [str, gdal.Dataset]], "rasters")
    core_utils.type_check(method, [str], "method")

    rasters_list = core_utils.ensure_list(rasters)
    assert method in ["reference", "intersection", "union"], (
        f"Invalid bounding_box: {method}. Options include: reference, intersection, and union."
    )

    assert gdal_utils.is_raster_list(rasters_list), "All rasters must be raster datasets."

    if len(rasters_list) == 1:
        return rasters_list[0]

    # Count intersections
    most_intersections = 0
    intersections_arr = []
    for idx_i, raster_i in enumerate(rasters_list):
        intersections = 0
        for idx_j, raster_j in enumerate(rasters_list):
            if idx_i == idx_j:
                continue

            # Uses the latlng bounding box to determine if the rasters intersect.
            intersection = core_raster.rasters_intersect(raster_i, raster_j)

            if intersection:
                intersections += 1

        if intersections > most_intersections:
            most_intersections = intersections

        intersections_arr.append(intersections)

    # Find the raster with the most intersections.
    largest_area = 0.0
    largest_area_idx = 0
    for idx, intersection in enumerate(intersections_arr):
        if intersection == most_intersections:
            raster = rasters_list[idx]
            raster_metadata = core_raster.raster_to_metadata(raster)
            raster_area = raster_metadata["area_latlng"]

            if raster_area > largest_area:
                largest_area = raster_area
                largest_area_idx = idx

    best_reference = gdal_utils.get_path_from_dataset(rasters_list[largest_area_idx])

    input_path_list = []
    for raster in rasters_list:
        input_path_list.append(gdal_utils.get_path_from_dataset(raster))

    all_other_rasters = []
    for raster_path in input_path_list:
        if raster_path == best_reference:
            continue

        all_other_rasters.append(raster_path)

    for raster in all_other_rasters:
        assert core_raster.rasters_intersect(best_reference, raster), (
            f"Rasters {best_reference} and {raster} do not intersect."
        )

    if method == "reference":
        return best_reference

    matched_rasters = match_raster_projections(
        all_other_rasters,
        best_reference,
        copy_if_already_correct=False,
    )

    best_reference_meta = core_raster.raster_to_metadata(best_reference)

    best_geom = best_reference_meta["geom"]
    for raster_path in matched_rasters:
        raster_meta = core_raster.raster_to_metadata(raster_path)
        raster_geom = raster_meta["geom"]

        if not raster_geom.Intersects(best_geom):
            raise ValueError(
                f"Raster {raster_path} did not intersect. Consider not using bbox='intersection'"
            )

        if method == "union":
            best_geom = raster_geom.Union(best_geom)
        else:
            best_geom = raster_geom.Intersection(best_geom)

    bbox_best_geom = best_geom.GetEnvelope()

    for raster_path in matched_rasters:
        if raster_path not in input_path_list:
            gdal_utils.delete_if_in_memory(raster_path)

    outbox_bbox = bbox_utils.align_bboxes_to_pixel_size(
        best_reference_meta["bbox"],
        bbox_best_geom,
        best_reference_meta["pixel_width"],
        best_reference_meta["pixel_height"],
    )

    creation_options = gdal_utils.default_creation_options()

    width = int(round((outbox_bbox[1] - outbox_bbox[0]) / best_reference_meta["pixel_width"]))
    height = int(round((outbox_bbox[3] - outbox_bbox[2]) / best_reference_meta["pixel_height"]))

    out_path = core_raster.create_empty_raster(
        out_path=f"/vsimem/intersection_union_{uuid4().int}.tif",
        width=width,
        height=height,
        pixel_width=best_reference_meta["pixel_width"],
        pixel_height=best_reference_meta["pixel_height"],
        bands=1,
        dtype=best_reference_meta["dtype"],
        x_min=outbox_bbox[0],
        y_max=outbox_bbox[3],
        projection=best_reference_meta["projection"],
        creation_options=creation_options,
    )

    return out_path


def align_rasters(
    rasters: Union[str, gdal.Dataset, List[Union[str, gdal.Dataset]]],
    *,
    out_path: Optional[Union[str, List[str]]] = None,
    reference: Optional[Union[str, gdal.Dataset]] = None,
    method: str = "reference",
    target_nodata: Optional[Union[int, float]] = None,
    resample_alg: str = "nearest",
    overwrite: bool = True,
    creation_options: Optional[List[str]] = None,
    prefix: str = "",
    suffix: str = "",
    ram: Union[int, str] = "auto",
) -> List[str]:
    """
    Aligns rasters either to a reference raster or to each other using one of three methods:
    reference, intersection, or union.

    Parameters
    ----------
    rasters : Union[str, gdal.Dataset, List[Union[str, gdal.Dataset]]]
        The rasters to align.

    out_path : List[str] or str, optional
        The output path(s), default: None

    reference : Str or gdal.Dataset, optional
        The reference raster to align to, default: None

    method : str, optional
        The method to use, default: "reference" [reference, intersection, union]

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

    ram : Union[str, int], optional
        Amount of RAM to use in MB. If "auto", the amount of RAM will be determined automatically.
    
    Returns
    -------
    List[str]
        The aligned rasters.
    """
    core_utils.type_check(rasters, [str, gdal.Dataset, [str, gdal.Dataset]], "rasters")
    core_utils.type_check(out_path, [str, None, [str]], "out_path")
    core_utils.type_check(reference, [str, [str], None], "reference")
    core_utils.type_check(method, [str], "method")
    core_utils.type_check(target_nodata, [int, float, None], "target_nodata")
    core_utils.type_check(resample_alg, [str], "resample_alg")
    core_utils.type_check(overwrite, [bool], "overwrite")
    core_utils.type_check(creation_options, [[str], None], "creation_options")
    core_utils.type_check(prefix, [str], "prefix")
    core_utils.type_check(suffix, [str], "suffix")
    core_utils.type_check(ram, [int, str], "ram")

    raster_list = core_utils.ensure_list(rasters)

    assert gdal_utils.is_raster_list(raster_list), "rasters must be a single raster or a list of rasters."
    assert method in ["reference", "intersection", "union"], "method must be one of reference, intersection, or union."

    add_uuid = True if out_path is None else False

    path_list = gdal_utils.create_output_path_list(
        raster_list,
        out_path,
        overwrite=overwrite,
        add_uuid=add_uuid,
        ext=".tif",
        prefix=prefix,
        suffix=suffix,
    )

    if reference is None:
        reference = find_best_reference(raster_list, method)

    aligned = align_rasters_to_reference(
        raster_list,
        reference,
        out_path=path_list,
        resample_alg=resample_alg,
        target_nodata=target_nodata,
        overwrite=overwrite,
        creation_options=creation_options,
        suffix=suffix,
        prefix=prefix,
        ram=ram,
    )

    return aligned
