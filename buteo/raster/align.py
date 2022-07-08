"""
Functions to align a series of rasters to a master or a reference.

TODO:
    - Improve documentation
    - Phase correlation?
    - Remove the typings
"""

import sys; sys.path.append("../../") # Path: buteo/raster/align.py
import os

import numpy as np
from osgeo import gdal, ogr, osr

from buteo.raster.io import (
    raster_to_metadata,
    rasters_are_aligned,
    ready_io_raster,
)
from buteo.vector.io import _vector_to_metadata
from buteo.raster.reproject import _reproject_raster
from buteo.vector.reproject import _reproject_vector
from buteo.utils.core import (
    remove_if_overwrite,
    type_check,
)
from buteo.utils.gdal_utils import (
    parse_projection,
    raster_size_from_list,
    is_raster,
    is_vector,
    path_to_driver_raster,
    default_options,
    gdal_nodata_value_from_type,
    translate_resample_method,
)


# TODO: Fix if not all a reprojected, paths are incorrect.
def match_projections(
    rasters,
    master,
    out_dir,
    *,
    overwrite=False,
    dst_nodata="infer",
    copy_if_already_correct=True,
):
    target_projection = parse_projection(master)

    created = []

    for raster in rasters:
        metadata = raster_to_metadata(raster)
        outname = out_dir + metadata["name"] + ".tif"
        created.append(outname)

        if os.path.exists(outname):
            if not overwrite:
                continue

        _reproject_raster(
            raster,
            target_projection,
            outname,
            copy_if_already_correct=copy_if_already_correct,
            dst_nodata=dst_nodata,
        )

    return created


# TODO: phase_cross_correlation
# TODO: Ensure that the xmin, ymax is always the same
# https://github.com/scikit-image/scikit-image/blob/main/skimage/registration/_phase_cross_correlation.py#L109-L276
def align_rasters(
    rasters,
    *,
    out_path=None,
    master=None,
    postfix="_aligned",
    bounding_box="intersection",
    resample_alg="nearest",
    target_size=None,
    target_in_pixels=False,
    projection=None,
    overwrite=True,
    creation_options=[],
    src_nodata="infer",
    dst_nodata="infer",
    prefix="",
    ram=8000,
    skip_existing=False,
):
    type_check(rasters, [list, str, gdal.Dataset], "rasters")
    type_check(out_path, [list, str], "out_path", allow_none=True)
    type_check(master, [list, str], "master", allow_none=True)
    type_check(
        bounding_box, [str, gdal.Dataset, ogr.DataSource, list, tuple], "bounding_box"
    )
    type_check(resample_alg, [str], "resample_alg")
    type_check(
        target_size,
        [tuple, list, int, float, str, gdal.Dataset],
        "target_size",
        allow_none=True,
    )
    type_check(
        target_in_pixels,
        [int, str, gdal.Dataset, ogr.DataSource, osr.SpatialReference],
        "target_in_pixels",
        allow_none=True,
    )
    type_check(overwrite, [bool], "overwrite")
    type_check(creation_options, [list], "creation_options")
    type_check(src_nodata, [str, int, float], "src_nodata", allow_none=True)
    type_check(dst_nodata, [str, int, float], "dst_nodata", allow_none=True)
    type_check(prefix, [str], "prefix")
    type_check(postfix, [str], "postfix")

    was_list = False

    if isinstance(rasters, list):
        was_list = True
    elif isinstance(rasters, str) or isinstance(rasters, gdal.Dataset):
        rasters = [rasters]

    raster_list, path_list = ready_io_raster(
        rasters,
        out_path,
        overwrite=overwrite,
        prefix=prefix,
        postfix=postfix,
        uuid=False,
    )

    x_pixels = None
    y_pixels = None
    x_res = None
    y_res = None
    target_projection = None
    target_bounds = None

    reprojected_rasters = []

    # Read the metadata for each raster.
    # Catalogue the used projections, to choose the most common one if necessary.
    used_projections = []
    metadata = []

    for raster in rasters:
        meta = raster_to_metadata(raster)
        metadata.append(meta)
        used_projections.append(meta["projection"])

    # If there is a master layer, copy information from that layer.
    if master is not None:
        master_metadata = raster_to_metadata(master)

        target_projection = master_metadata["projection_osr"]
        x_min, y_max, x_max, y_min = master_metadata["extent"]

        # Set the target values.
        target_bounds = (x_min, y_min, x_max, y_max)
        x_res = master_metadata["pixel_width"]
        y_res = master_metadata["pixel_height"]
        x_pixels = master_metadata["width"]
        y_pixels = master_metadata["height"]
        target_size = (x_res, y_res)

        target_in_pixels = False

    # We allow overwrite of parameters specifically set.
    # Handle projection
    if projection is not None:
        target_projection = parse_projection(projection)

    # If no projection is specified, other from master or parameters. The most common one is chosen.
    elif target_projection is None:

        # Sort and count the projections
        projection_counter = {}
        for proj in used_projections:
            if proj in projection_counter:
                projection_counter[proj] += 1
            else:
                projection_counter[proj] = 1

        # Choose most common projection
        most_common_projection = sorted(
            projection_counter, key=projection_counter.get, reverse=True
        )

        target_projection = parse_projection(most_common_projection[0])

    if target_size is not None:

        # If a raster is input, use it's pixel size as target values.
        if isinstance(target_size, (gdal.Dataset, str)):
            if isinstance(target_size, str) and not is_raster(target_size):
                raise ValueError(
                    f"Unable to parse the raster used for target_size: {target_size}"
                )

            # Reprojection is necessary to ensure the correct pixel_size
            reprojected_target_size = _reproject_raster(
                target_size, target_projection
            )
            target_size_raster = raster_to_metadata(reprojected_target_size)

            # Set the target values.
            x_res = target_size_raster["width"]
            y_res = target_size_raster["height"]
        else:
            # If a list, tuple, int or float is passed. Turn them into target values.
            x_res, y_res, x_pixels, y_pixels = raster_size_from_list(
                target_size, target_in_pixels
            )

    # If nothing has been specified, we will infer the pixel_size based on
    # the median of all input rasters.
    elif x_res is None and y_res is None and x_pixels is None and y_pixels is None:

        # Ready numpy arrays for insertion
        x_res_arr = np.empty(len(raster_list), dtype="float32")
        y_res_arr = np.empty(len(raster_list), dtype="float32")

        for index, raster in enumerate(raster_list):
            # It is necessary to reproject each raster, as pixel height and width
            # might be different after projection.
            reprojected = _reproject_raster(raster, target_projection)
            target_size_raster = raster_to_metadata(reprojected)

            # Add the pixel sizes to the numpy arrays
            x_res_arr[index] = target_size_raster["pixel_width"]
            y_res_arr[index] = target_size_raster["pixel_height"]

            # Keep track of the reprojected arrays so we only reproject rasters once.
            reprojected_rasters.append(reprojected)

        # Use the median values of pixel sizes as target values.
        x_res = np.median(x_res_arr)
        y_res = np.median(y_res_arr)

    if target_bounds is None:

        # If a bounding box is supplied, simply use that one.
        # It must be in the target projection.
        if isinstance(bounding_box, (list, tuple)):
            if len(bounding_box) != 4:
                raise ValueError("bounding_box as a list/tuple must have 4 values.")
            target_bounds = bounding_box

        # If the bounding box is a raster. Take the extent and
        # reproject it to the target projection.
        elif is_raster(bounding_box):
            reprojected_bbox_raster = raster_to_metadata(
                _reproject_raster(bounding_box, target_projection)
            )

            x_min, y_max, x_max, y_min = reprojected_bbox_raster["extent"]

            # add to target values.
            target_bounds = (x_min, y_min, x_max, y_max)

        # If the bounding box is a raster. Take the extent and
        # reproject it to the target projection.
        elif is_vector(bounding_box):
            reprojected_bbox_vector = _vector_to_metadata(
                _reproject_vector(bounding_box, target_projection)
            )

            x_min, y_max, x_max, y_min = reprojected_bbox_vector["extent"]

            # add to target values.
            target_bounds = (x_min, y_min, x_max, y_max)

        # If the bounding box is a string, we either take the union
        # or the intersection of all the
        # bounding boxes of the input rasters.
        elif isinstance(bounding_box, str):
            if bounding_box == "intersection" or bounding_box == "union":
                extents = []

                # If the rasters have not been reprojected, reproject them now.
                if len(reprojected_rasters) != len(raster_list):
                    reprojected_rasters = []

                    for raster in raster_list:
                        raster_metadata = raster_to_metadata(raster)

                        if raster_metadata["projection_osr"].IsSame(target_projection):
                            reprojected_rasters.append(raster)
                        else:
                            reprojected = _reproject_raster(
                                raster, target_projection
                            )
                            reprojected_rasters.append(reprojected)

                # Add the extents of the reprojected rasters to the extents list.
                for reprojected_raster in reprojected_rasters:
                    reprojected_raster_metadata = dict(
                        raster_to_metadata(reprojected_raster)
                    )
                    extents.append(reprojected_raster_metadata["extent"])

                # Placeholder values
                x_min, y_max, x_max, y_min = extents[0]

                # Loop the extents. Narrowing if intersection, expanding if union.
                for index, extent in enumerate(extents):
                    if index == 0:
                        continue

                    if bounding_box == "intersection":
                        if extent[0] > x_min:
                            x_min = extent[0]
                        if extent[1] < y_max:
                            y_max = extent[1]
                        if extent[2] < x_max:
                            x_max = extent[2]
                        if extent[3] > y_min:
                            y_min = extent[3]

                    elif bounding_box == "union":
                        if extent[0] < x_min:
                            x_min = extent[0]
                        if extent[1] > y_max:
                            y_max = extent[1]
                        if extent[2] > x_max:
                            x_max = extent[2]
                        if extent[3] < y_min:
                            y_min = extent[3]

                # Add to target values.
                target_bounds = (x_min, y_min, x_max, y_max)

            else:
                raise ValueError(
                    f"Unable to parse or infer target_bounds: {target_bounds}"
                )
        else:
            raise ValueError(f"Unable to parse or infer target_bounds: {target_bounds}")

    """
        If the rasters have not been reprojected, we reproject them now.
        The reprojection is necessary as warp has to be a two step process
        in order to align the rasters properly. This might not be necessary
        in a future version of gdal.
    """
    if len(reprojected_rasters) != len(raster_list):
        reprojected_rasters = []

        for raster in raster_list:
            raster_metadata = raster_to_metadata(raster)

            # If the raster is already the correct projection, simply append the raster.
            if raster_metadata["projection_osr"].IsSame(target_projection):
                reprojected_rasters.append(raster)
            else:
                reprojected = _reproject_raster(raster, target_projection)
                reprojected_rasters.append(reprojected)

    # If any of the target values are still undefined. Throw an error!
    if target_projection is None or target_bounds is None:
        raise Exception("Error while preparing the target projection or bounds.")

    if x_res is None and y_res is None and x_pixels is None and y_pixels is None:
        raise Exception("Error while preparing the target pixel size.")

    # This is the list of rasters to return. If output is not memory, it's a list of paths.
    return_list = []
    for index, raster in enumerate(reprojected_rasters):
        raster_metadata = raster_to_metadata(raster)

        out_name = path_list[index]
        out_format = path_to_driver_raster(out_name)

        if skip_existing and os.path.exists(out_name):
            return_list.append(out_name)
            continue

        # Handle nodata.
        out_src_nodata = None
        out_dst_nodata = None
        if src_nodata == "infer":
            out_src_nodata = raster_metadata["nodata_value"]

            if out_src_nodata is None:
                out_src_nodata = gdal_nodata_value_from_type(
                    raster_metadata["datatype_gdal_raw"]
                )

        elif src_nodata is None:
            out_src_nodata = None
        elif not isinstance(src_nodata, str):
            out_src_nodata = src_nodata

        if dst_nodata == "infer":
            out_dst_nodata = out_src_nodata
        elif dst_nodata is False or dst_nodata is None:
            out_dst_nodata = None
        elif src_nodata is None:
            out_dst_nodata = None
        elif not isinstance(dst_nodata, str):
            out_dst_nodata = dst_nodata

        # Removes file if it exists and overwrite is True.
        remove_if_overwrite(out_name, overwrite)

        # Hand over to gdal.Warp to do the heavy lifting!
        warped = gdal.Warp(
            out_name,
            raster,
            xRes=x_res,
            yRes=y_res,
            width=x_pixels,
            height=y_pixels,
            dstSRS=target_projection,
            outputBounds=target_bounds,
            format=out_format,
            resampleAlg=translate_resample_method(resample_alg),
            creationOptions=default_options(creation_options),
            srcNodata=out_src_nodata,
            dstNodata=out_dst_nodata,
            targetAlignedPixels=False,
            cropToCutline=False,
            multithread=True,
            warpMemoryLimit=ram,
        )

        if warped is None:
            raise Exception("Error while warping rasters.")

        return_list.append(out_name)

    if not rasters_are_aligned(return_list, same_extent=True):
        raise Exception("Error while aligning rasters. Output is not aligned")

    if not was_list:
        return return_list[0]

    return return_list
