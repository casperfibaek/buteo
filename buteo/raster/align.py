"""
### Align rasters ###

Functions to align a series of rasters to a master or a reference.

TODO:
    * Fix if not all a reprojected, paths are incorrect.
    * phase_cross_correlation
    * Ensure get_pixel_offsets works as planned.
    * Fix RAM limits to be dynamic % of available RAM.
"""

# Standard library
import sys; sys.path.append("../../")

# External
import numpy as np
from osgeo import gdal, ogr, osr

# Internal
from buteo.utils import core_utils, gdal_enums, gdal_utils, bbox_utils
from buteo.raster import core_raster
from buteo.vector import core_vector
from buteo.raster.reproject import _reproject_raster
from buteo.vector.reproject import _reproject_vector


def match_raster_projections(
    rasters,
    master,
    *,
    out_path=None,
    overwrite=False,
    dst_nodata="infer",
    copy_if_already_correct=True,
    creation_options=None,
):
    """
    Match a raster or list of rasters to a master layer. The master can be either
    an **OGR** layer or a **GDAL** layer.

    ## Args:
    `rasters` (_list_): A list of rasters to match. </br>
    `master` (_str_/_gdal.Dataset_/_ogr.DataSource_): Path to the master raster or vector. </br>

    ## Kwargs:
    `out_path` (_str_/_list_): Paths to the output. If not provided, the output will be in-memory rasters. (Default: **None**) </br>
    `overwrite` (_bool_): If True, existing rasters will be overwritten. (Default: **False**) </br>
    `dst_nodata` (_str_): Value to use for no-data pixels. If not provided, the value will be transfered from the original. (Default: **"infer"**) </br>
    `copy_if_already_correct` (_bool_): If True, the raster will be copied if it is already in the correct projection. (Default: **True**) </br>
    `creation_options` (_list_): List of creation options to pass to the output raster. (Default: **None**) </br>

    ## Returns:
    (_list_): A list of reprojected input rasters with the correct projection.
    """
    assert isinstance(rasters, list), "rasters must be a list."
    assert isinstance(master, (str, gdal.Dataset, ogr.DataSource)), "master must be a string, gdal.Dataset, or ogr.DataSource."
    assert gdal_utils.is_raster_list(rasters), "rasters must be a list of rasters."

    try:
        target_projection = gdal_utils.parse_projection(master)
    except Exception:
        raise ValueError(f"Unable to parse projection from master. Received: {master}") from None

    add_uuid = out_path is None

    path_list = gdal_utils.create_output_path_list(rasters, out_path, overwrite=overwrite, add_uuid=add_uuid)

    output = []

    for index, in_raster in enumerate(rasters):
        path = _reproject_raster(
            in_raster,
            target_projection,
            out_path=path_list[index],
            overwrite=overwrite,
            copy_if_same=copy_if_already_correct,
            dst_nodata=dst_nodata,
            creation_options=gdal_utils.default_creation_options(creation_options),
        )

        output.append(path)

    return output



def align_rasters(
    rasters,
    *,
    out_path=None,
    master=None,
    bounding_box="intersection",
    resample_alg="nearest",
    target_size=None,
    target_in_pixels=False,
    projection=None,
    overwrite=True,
    creation_options=None,
    src_nodata="infer",
    dst_nodata="infer",
    prefix="",
    suffix="_aligned",
    ram="auto",
):
    """
    Aligns a series of rasters to a master raster or specified requirements.

    ## Args:
    `rasters` (_list_): A list of rasters to align. </br>

    ## Kwargs:
    `out_path` (_str_/_list_): Paths to the output. If not provided, the output will be in-memory rasters. (Default: **None**) </br>
    `master` (_str_/_gdal.Dataset_/_ogr.DataSource_): Path to the master raster or vector. (Default: **None**) </br>
    `suffix` (_str_): Suffix to append to the output raster. (Default: **"_aligned"**) </br>
    `bounding_box` (_str_): Method to use for aligning the rasters. Can be either "intersection" or "union". (Default: **"intersection"**) </br>
    `resample_alg` (_str_): Resampling algorithm to use. (Default: **nearest**) </br>
    `target_size` (_list_/_gdal.Dataset_/_ogr.DataSource_): Target size of the output raster. (Default: **None**) </br>
    `target_in_pixels` (_bool_): If True, the target size will be in pixels. (Default: **False**) </br>
    `projection` (_str_/_gdal.Dataset_/_ogr.DataSource_): Projection to use for the output raster. (Default: **None**) </br>
    `overwrite` (_bool_): If **True**, existing rasters will be overwritten. (Default: **True**) </br>
    `creation_options` (_list_): List of creation options to pass to the output raster. (Default: **None**) </br>
    `src_nodata` (_str_/_int_/_float_/_None_): The source dataset of the align sets. (Default: **"infer"**) </br>
    `dst_nodata` (_str_/_int_/_float_/_None_): The destination dataset of the align sets. (Default: **"infer"**) </br>
    `prefix`: (_str_): Prefix to add to the output rasters. (Default: **""**) </br>
    `suffix`: (_str_): Suffix to add to the output rasters. (Default: **""**) </br>
    `ram`: (_int_/_str_): The ram available to **GDAL** for the processing in MB or percentage.
    If auto 80% of available ram is allowed. (Default: **auto**) </br>

    ## Return:
    (_list_): A list of paths to the aligned rasters.
    """
    core_utils.type_check(rasters, [str, gdal.Dataset, [str, gdal.Dataset]], "rasters")
    core_utils.type_check(out_path, [str, None, [str]], "out_path")
    core_utils.type_check(master, [str, [str], None], "master")
    core_utils.type_check(bounding_box, [str, gdal.Dataset, ogr.DataSource, list, tuple], "bounding_box")
    core_utils.type_check(resample_alg, [str], "resample_alg")
    core_utils.type_check(target_size, [tuple, list, int, float, str, gdal.Dataset, None], "target_size")
    core_utils.type_check(target_in_pixels, [int, str, gdal.Dataset, ogr.DataSource, osr.SpatialReference, None], "target_in_pixels")
    core_utils.type_check(overwrite, [bool], "overwrite")
    core_utils.type_check(creation_options, [[str], None], "creation_options")
    core_utils.type_check(src_nodata, [str, int, float, None], "src_nodata")
    core_utils.type_check(dst_nodata, [str, int, float, None], "dst_nodata")
    core_utils.type_check(prefix, [str], "prefix")
    core_utils.type_check(suffix, [str], "suffix")

    assert gdal_utils.is_raster_list(rasters), "rasters must be a list of rasters."

    add_uuid = out_path is None

    raster_list = core_utils.ensure_list(rasters)
    path_list = gdal_utils.create_output_path_list(raster_list, out_path, overwrite=overwrite, add_uuid=add_uuid)

    x_pixels = None
    y_pixels = None
    x_res = None
    y_res = None
    target_projection = None
    target_bounds = None

    reprojected_rasters = []
    paths_to_unlink = []

    # Read the metadata for each raster.
    # Catalogue the used projections, to choose the most common one if necessary.
    used_projections = []
    metadata = []

    for raster in rasters:
        meta = core_raster.raster_to_metadata(raster)
        metadata.append(meta)
        used_projections.append(meta["projection_wkt"])

    # If there is a master layer, copy information from that layer.
    if master is not None:
        master_metadata = core_raster.raster_to_metadata(master)

        target_projection = master_metadata["projection_osr"]
        x_min, x_max, y_min, y_max = master_metadata["bbox"]

        # Set the target values.
        target_bounds = (x_min, x_max, y_min, y_max)
        x_res = master_metadata["pixel_width"]
        y_res = master_metadata["pixel_height"]
        x_pixels = master_metadata["width"]
        y_pixels = master_metadata["height"]
        target_size = (x_res, y_res)

        target_in_pixels = False

    # We allow overwrite of parameters specifically set.
    # Handle projection
    if projection is not None:
        target_projection = gdal_utils.parse_projection(projection)

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

        target_projection = gdal_utils.parse_projection(most_common_projection[0])

    if target_size is not None:

        # If a raster is input, use it's pixel size as target values.
        if isinstance(target_size, (gdal.Dataset, str)):
            if isinstance(target_size, str) and not gdal_utils.is_raster(target_size):
                raise ValueError(
                    f"Unable to parse the raster used for target_size: {target_size}"
                )

            # Reprojection is necessary to ensure the correct pixel_size
            reprojected_target_size = _reproject_raster(
                target_size, target_projection
            )
            target_size_raster = core_raster.raster_to_metadata(reprojected_target_size)

            gdal_utils.delete_if_in_memory(reprojected_target_size)
            reprojected_target_size = None

            # Set the target values.
            x_res = target_size_raster["width"]
            y_res = target_size_raster["height"]
        else:
            # If a list, tuple, int or float is passed. Turn them into target values.
            x_res, y_res, x_pixels, y_pixels = bbox_utils.get_pixel_offsets(
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
            reprojected = _reproject_raster(raster, target_projection, copy_if_same=False)
            target_size_raster = core_raster.raster_to_metadata(reprojected)

            paths_to_unlink.append(core_raster.get_raster_path(reprojected))

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
            target_bounds = list(bounding_box)

        # If the bounding box is a raster. Take the extent and
        # reproject it to the target projection.
        elif gdal_utils.is_raster(bounding_box):
            raster_bbox_reproject = _reproject_raster(bounding_box, target_projection, copy_if_same=False)
            reprojected_bbox_raster = core_raster.raster_to_metadata(raster_bbox_reproject)
            gdal_utils.delete_if_in_memory(raster_bbox_reproject)
            raster_bbox_reproject = None

            x_min, x_max, y_min, y_max = reprojected_bbox_raster["bbox"]

            # add to target values.
            target_bounds = [x_min, x_max, y_min, y_max]

        # If the bounding box is a raster. Take the extent and
        # reproject it to the target projection.
        elif gdal_utils.is_vector(bounding_box):
            vector_bbox_reproject = _reproject_vector(bounding_box, target_projection, copy_if_same=False)
            reprojected_bbox_vector = core_vector._vector_to_metadata(vector_bbox_reproject) # pylint: disable=protected-access
            gdal_utils.delete_if_in_memory(vector_bbox_reproject)
            vector_bbox_reproject = None

            x_min, x_max, y_min, y_max = reprojected_bbox_vector["bbox"]

            # add to target values.
            target_bounds = [x_min, x_max, y_min, y_max]

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
                        raster_metadata = core_raster.raster_to_metadata(raster)

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
                        core_raster.raster_to_metadata(reprojected_raster)
                    )
                    extents.append(reprojected_raster_metadata["bbox"])

                # Placeholder values
                x_min, x_max, y_min, y_max = extents[0]

                # Loop the extents. Narrowing if intersection, expanding if union.
                for index, extent in enumerate(extents):
                    if index == 0:
                        continue

                    b_x_min, b_x_max, b_y_min, b_y_max = extent

                    if bounding_box == "intersection":
                        if b_x_min > x_min:
                            x_min = b_x_min
                        if b_x_max < x_max:
                            x_max = b_x_max
                        if b_y_min > y_min:
                            y_min = b_y_min
                        if b_y_max < y_max:
                            y_max = b_y_max

                    elif bounding_box == "union":
                        if b_x_min < x_min:
                            x_min = b_x_min
                        if b_x_max > y_max:
                            y_max = b_x_max
                        if y_min > x_max:
                            x_max = y_min
                        if y_max < y_min:
                            y_min = y_max

                # Add to target values.
                target_bounds = [x_min, x_max, y_min, y_max]

            else:
                raise ValueError(
                    f"Unable to parse or infer target_bounds: {target_bounds}"
                )
        else:
            raise ValueError(f"Unable to parse or infer target_bounds: {target_bounds}")

    # If the rasters have not been reprojected, we reproject them now.
    # The reprojection is necessary as warp has to be a two step process
    # in order to align the rasters properly. This might not be necessary
    # in a future version of gdal.

    if len(reprojected_rasters) != len(raster_list):
        for raster in reprojected_rasters:
            gdal_utils.delete_if_in_memory(raster)

        reprojected_rasters = []

        for raster in raster_list:
            raster_metadata = core_raster.raster_to_metadata(raster)

            # If the raster is already the correct projection, simply append the raster.
            if raster_metadata["projection_osr"].IsSame(target_projection):
                reprojected_rasters.append(raster)
            else:
                reprojected = _reproject_raster(raster, target_projection)
                reprojected_rasters.append(reprojected)
                paths_to_unlink.append(core_raster.get_raster_path(reprojected))

    # If any of the target values are still undefined. Throw an error!
    if target_projection is None or target_bounds is None:
        raise Exception("Error while preparing the target projection or bounds.")

    if x_res is None and y_res is None and x_pixels is None and y_pixels is None:
        raise Exception("Error while preparing the target pixel size.")

    # This is the list of rasters to return. If output is not memory, it's a list of paths.
    return_list = []
    for index, raster in enumerate(reprojected_rasters):
        raster_metadata = core_raster.raster_to_metadata(raster)

        out_name = path_list[index]
        out_format = gdal_utils.path_to_driver_raster(out_name)

        # Handle nodata.
        out_src_nodata = None
        out_dst_nodata = None
        if src_nodata == "infer":
            out_src_nodata = raster_metadata["nodata_value"]

            if out_src_nodata is None:
                out_src_nodata = gdal_enums.translate_gdal_dtype_to_str(
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
        core_utils.remove_if_required(out_name, overwrite)

        # Hand over to gdal.Warp to do the heavy lifting!
        warped = gdal.Warp(
            out_name,
            raster,
            xRes=x_res,
            yRes=y_res,
            width=x_pixels,
            height=y_pixels,
            dstSRS=target_projection,
            outputBounds=bbox_utils.convert_ogr_bbox_to_gdal_bbox(target_bounds),
            format=out_format,
            resampleAlg=gdal_enums.translate_resample_method(resample_alg),
            creationOptions=gdal_utils.default_creation_options(creation_options),
            srcNodata=out_src_nodata,
            dstNodata=out_dst_nodata,
            targetAlignedPixels=False,
            cropToCutline=False,
            multithread=True,
            warpMemoryLimit=gdal_utils.get_gdalwarp_ram_limit(ram),
        )

        if warped is None:
            raise Exception("Error while warping rasters.")

        return_list.append(out_name)

    # Remove the reprojected rasters if they are in memory.
    for mem_path in paths_to_unlink:
        gdal_utils.delete_if_in_memory(mem_path)

    if not core_raster.rasters_are_aligned(return_list, same_extent=True):
        raise Exception("Error while aligning rasters. Output is not aligned")

    if isinstance(rasters, list):
        return return_list

    return return_list[0]
