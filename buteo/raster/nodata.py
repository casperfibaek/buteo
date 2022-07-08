"""
A module to handle the various aspects of NODATA in raster files.

TODO:
    - raster_to_mask
    - raster_invert_nodata
    - Improve documentation
"""

import sys; sys.path.append("../../") # Path: buteo/raster/nodata.py

import numpy as np
from osgeo import gdal

from buteo.utils.core import remove_if_overwrite, type_check
from buteo.utils.gdal_utils import (
    is_raster,
    gdal_nodata_value_from_type,
    raster_to_reference,
)
from buteo.raster.io import (
    raster_to_array,
    raster_to_disk,
    raster_to_memory,
    raster_to_metadata,
    array_to_raster,
    ready_io_raster,
    get_raster_path,
)


def raster_has_nodata_value(raster):
    """Check if a raster or a list of rasters contain nodata values

    Args:
        raster (path | raster | list): The raster(s) to check for nodata values.

    Returns:
        True if input raster has nodata values. If a list is the input, the output
        is a list of booleans indicating if the input raster has nodata values.
    """
    type_check(raster, [list, str, gdal.Dataset], "raster")

    nodata_values = []
    rasters = get_raster_path(raster, return_list=True)

    for internal_raster in rasters:
        if not is_raster(internal_raster):
            raise ValueError(f"Input raster is invalid: {internal_raster}")

        raster_metadata = raster_to_metadata(internal_raster)

        if not isinstance(raster_metadata, dict):
            raise Exception("Metadata is in the wrong format.")

        raster_nodata = raster_metadata["nodata_value"]

        if raster_nodata is not None:
            nodata_values.append(True)
        else:
            nodata_values.append(False)

    if isinstance(raster, list):
        return nodata_values
    else:
        return nodata_values[0]


def raster_get_nodata_value(raster):
    """Get the nodata value of a raster or a from a list of rasters.

    Args:
        raster (path | raster | list): The raster(s) to retrieve nodata values from.

    Returns:
        Returns the nodata value from a raster or a list of rasters
    """
    type_check(raster, [list, str, gdal.Dataset], "raster")

    rasters = get_raster_path(raster, return_list=True)

    nodata_values = []
    for internal_raster in rasters:
        if not is_raster(internal_raster):
            raise ValueError(f"Input raster is invalid: {internal_raster}")

        raster_metadata = raster_to_metadata(internal_raster)

        if not isinstance(raster_metadata, dict):
            raise Exception("Metadata is in the wrong format.")

        raster_nodata = raster_metadata["nodata_value"]

        nodata_values.append(raster_nodata)

    if isinstance(raster, list):
        return nodata_values
    else:
        return nodata_values[0]


def raster_set_nodata(
    raster,
    dst_nodata,
    out_path=None,
    *,
    overwrite=True,
    in_place=False,
    prefix="",
    postfix="_nodata_set",
    opened=False,
    creation_options=[],
):
    """Sets all the nodata from a raster or a list of rasters.

    Args:
        raster (path | raster | list): The raster(s) to retrieve nodata values from.

        dst_nodata (float, int, str, None): The target nodata value. If 'infer' the nodata
        value is set based on the input datatype. A list of nodata values can be based matching
        the amount of input rasters. If multiple nodata values should be set,
        use raster_mask_values.

    **kwargs:
        out_path (path | list | None): The destination of the changed rasters. If out_paths
        are specified, in_place is automatically set to False. The path can be a folder.

        in_place (bool): Should the rasters be changed in_place or copied?

        prefix (str): Prefix to add the the output if a folder is specified in out_path.

        postfix (str): Postfix to add the the output if a folder is specified in out_path.

    Returns:
        Returns the rasters with nodata set. If in_place is True a reference to the
        changed orignal is returned, otherwise a copied memory raster or the path to the
        generated raster is outputted.
    """
    type_check(raster, [list, str, gdal.Dataset], "raster")
    type_check(dst_nodata, [float, int, str, list], "dst_nodata", allow_none=True)
    type_check(out_path, [list, str], "out_path", allow_none=True)
    type_check(overwrite, [bool], "overwrite")
    type_check(prefix, [str], "prefix")
    type_check(postfix, [str], "postfix")
    type_check(opened, [bool], "opened")
    type_check(creation_options, [list], "creation_options")

    rasters, out_names = ready_io_raster(
        raster,
        out_path,
        overwrite=overwrite,
        prefix=prefix,
        postfix=postfix,
    )

    rasters_metadata = []
    internal_dst_nodata = None

    if isinstance(dst_nodata, str) and dst_nodata != "infer":
        raise ValueError(f"Invalid dst_nodata value. {dst_nodata}")

    if isinstance(dst_nodata, list):
        if not isinstance(raster, list) or len(dst_nodata) != len(raster):
            raise ValueError(
                "If dst_nodata is a list, raster must also be a list of equal length."
            )

        for value in dst_nodata:
            if isinstance(value, (float, int, str, None)):
                raise ValueError("Invalid type in dst_nodata list.")

            if isinstance(value, str) and value != "infer":
                raise ValueError("If dst_nodata is a string it must be 'infer'")

    output_rasters = []

    for index, internal_raster in enumerate(rasters):

        raster_metadata = None
        if len(rasters_metadata) == 0:
            raster_metadata = raster_to_metadata(internal_raster)

            if not isinstance(raster_metadata, dict):
                raise Exception("Metadata is in the wrong format.")

            rasters_metadata.append(raster_metadata)
        else:
            raster_metadata = rasters_metadata[index]

        if dst_nodata == "infer":
            internal_dst_nodata = gdal_nodata_value_from_type(
                raster_metadata["dtype_gdal_raw"]
            )
        elif isinstance(dst_nodata, list):
            internal_dst_nodata = dst_nodata[index]
        else:
            internal_dst_nodata = dst_nodata

        if in_place:
            for band in range(raster_metadata["bands"]):
                raster_band = internal_raster.GetRasterBand(band + 1)
                raster_band.SetNodataValue(internal_dst_nodata)
                raster_band = None
        else:
            if out_path is None:
                raster_mem = raster_to_memory(internal_raster)
                raster_mem_ref = raster_to_reference(raster_mem)
            else:
                remove_if_overwrite(out_names[index], overwrite)
                raster_mem = raster_to_disk(internal_raster, out_names[index])
                raster_mem_ref = raster_to_reference(raster_mem)

            for band in range(raster_metadata["bands"]):
                raster_band = raster_mem_ref.GetRasterBand(band + 1)
                raster_band.SetNodataValue(internal_dst_nodata)

    if isinstance(raster, list):
        return output_rasters

    return output_rasters[0]


def raster_remove_nodata(
    raster,
    out_path=None,
    *,
    in_place=False,
    overwrite=True,
    prefix="",
    postfix="_nodata_removed",
    creation_options=[],
):
    """Removes all the nodata from a raster or a list of rasters.

    Args:
        raster (path | raster | list): The raster(s) to retrieve nodata values from.

    **kwargs:
        out_path (path | list | None): The destination of the changed rasters. If out_paths
        are specified, in_place is automatically set to False. The path can be a folder.

        in_place (bool): Should the rasters be changed in_place or copied?

        prefix (str): Prefix to add the the output if a folder is specified in out_path.

        postfix (str): Postfix to add the the output if a folder is specified in out_path.

    Returns:
        Returns the rasters with nodata removed. If in_place is True a reference to the
        changed orignal is returned, otherwise a copied memory raster or the path to the
        generated raster is outputted.
    """

    return raster_set_nodata(
        raster,
        dst_nodata=None,
        out_path=out_path,
        in_place=in_place,
        overwrite=overwrite,
        prefix=prefix,
        postfix=postfix,
        creation_options=creation_options,
    )


def raster_mask_values(
    raster,
    values_to_mask,
    out_path=None,
    *,
    include_original_nodata=True,
    dst_nodata="infer",
    in_place=False,
    overwrite=True,
    opened=False,
    prefix="",
    postfix="_nodata_masked",
    creation_options=[],
):
    """Mask a raster with a list of values.

    Args:
        raster (path | raster | list): The raster(s) to retrieve nodata values from.
        values_to_mask (list): The list of values to mask in the raster(s)

    **kwargs:
        include_original_nodata: (bool): If True, the nodata value of the raster(s) will be
        included in the values to mask.

        dst_nodata (float, int, str, None): The target nodata value. If 'infer' the nodata
        value is set based on the input datatype. A list of nodata values can be based matching
        the amount of input rasters. If multiple nodata values should be set,
        use raster_mask_values.

        out_path (path | list | None): The destination of the changed rasters. If out_paths
        are specified, in_place is automatically set to False. The path can be a folder.

        in_place (bool): Should the rasters be changed in_place or copied?

        prefix (str): Prefix to add the the output if a folder is specified in out_path.

        postfix (str): Postfix to add the the output if a folder is specified in out_path.

    Returns:
        Returns the rasters with nodata removed. If in_place is True a reference to the
        changed orignal is returned, otherwise a copied memory raster or the path to the
        generated raster is outputted.
    """
    type_check(raster, [list, str, gdal.Dataset], "raster")
    type_check(values_to_mask, [list], "values_to_mask")
    type_check(out_path, [list, str], "out_path", allow_none=True)
    type_check(include_original_nodata, [bool], "include_original_nodata")
    type_check(dst_nodata, [float, int, str, list], "dst_nodata", allow_none=True)
    type_check(in_place, [bool], "in_place")
    type_check(overwrite, [bool], "overwrite")
    type_check(prefix, [str], "prefix")
    type_check(postfix, [str], "postfix")
    type_check(opened, [bool], "opened")
    type_check(creation_options, [list], "creation_options")

    rasters_metadata = []
    internal_in_place = in_place if out_path is None else False
    internal_dst_nodata = None

    for value in values_to_mask:
        if not isinstance(value, (int, float)):
            raise ValueError("Values in values_to_mask must be ints or floats")

    if isinstance(dst_nodata, str) and dst_nodata != "infer":
        raise ValueError(f"Invalid dst_nodata value. {dst_nodata}")

    if isinstance(dst_nodata, list):
        if not isinstance(raster, list) or len(dst_nodata) != len(raster):
            raise ValueError(
                "If dst_nodata is a list, raster must also be a list of equal length."
            )

        for value in dst_nodata:
            if isinstance(value, (float, int, str, None)):
                raise ValueError("Invalid type in dst_nodata list.")

            if isinstance(value, str) and value != "infer":
                raise ValueError("If dst_nodata is a string it must be 'infer'")

    raster_list, out_names = ready_io_raster(
        raster, out_path, overwrite=overwrite, prefix=prefix, postfix=postfix
    )

    output_rasters = []

    for index, internal_raster in enumerate(raster_list):

        raster_metadata = None
        if len(rasters_metadata) == 0:
            raster_metadata = raster_to_metadata(internal_raster)
            rasters_metadata.append(raster_metadata)
        else:
            raster_metadata = rasters_metadata[index]

        if dst_nodata == "infer":
            internal_dst_nodata = gdal_nodata_value_from_type(
                raster_metadata["dtype_gdal_raw"]
            )
        elif isinstance(dst_nodata, list):
            internal_dst_nodata = dst_nodata[index]
        else:
            internal_dst_nodata = dst_nodata

        mask_values = list(values_to_mask)
        if include_original_nodata:
            if raster_metadata["nodata_value"] is not None:
                mask_values.append(raster_metadata["nodata_value"])

        arr = raster_to_array(internal_raster, filled=True)

        mask = None
        for idx, mask_value in enumerate(mask_values):
            if idx == 0:
                mask = arr == mask_value
            else:
                mask = mask | arr == mask_value

        arr = np.ma.masked_array(arr, mask=mask, fill_value=internal_dst_nodata)

        if internal_in_place:
            for band in range(raster_metadata["bands"]):
                raster_band = internal_raster.GetRasterBand(band + 1)
                raster_band.WriteArray(arr[:, :, band])
                raster_band = None
        else:
            out_name = out_names[index]
            remove_if_overwrite(out_name, overwrite)

            output_rasters.append(
                array_to_raster(arr, internal_raster, out_path=out_name)
            )

    if isinstance(raster, list):
        return output_rasters

    return output_rasters[0]
