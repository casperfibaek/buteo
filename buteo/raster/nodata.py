"""
### Handle nodata values in rasters. ###

A module to handle the various aspects of NODATA in raster files.

TODO:
    * raster_to_mask
    * raster_invert_nodata
"""

# Standard library
import sys; sys.path.append("../../")

# External
import numpy as np
from osgeo import gdal

# Internal
from buteo.utils import core_utils, gdal_utils, gdal_enums
from buteo.raster import core_raster



def raster_has_nodata_value(raster):
    """
    Check if a raster or a list of rasters contain nodata values

    ## Args:
    `raster` (_str_/_gdal.Dataset_, _list_): The raster to check for nodata values.

    ## Returns:
    (_bool_): **True** if the raster or list of rasters contain nodata values.
    """
    core_utils.type_check(raster, [str, gdal.Dataset, [str, gdal.Dataset]], "raster")

    rasters = gdal_utils.get_path_from_dataset(core_utils.ensure_list(raster))
    assert gdal_utils.is_raster_list(raster), f"List contains invalid rasters. {raster}"

    nodata_values = []
    for internal_raster in rasters:
        raster_metadata = core_raster._raster_to_metadata(internal_raster)

        raster_nodata = raster_metadata["nodata_value"]

        if raster_nodata is not None:
            nodata_values.append(True)
        else:
            nodata_values.append(False)

    if isinstance(raster, list):
        return nodata_values

    return nodata_values[0]


def raster_get_nodata_value(raster):
    """
    Get the nodata value of a raster or a list of rasters.

    ## Args:
    `raster` (_str_/_gdal.Dataset_, _list_): The raster(s) to get nodata values from.

    ## Returns:
    (_float_/_int_/_list_): The nodata value(s) of the raster(s).
    """
    core_utils.type_check(raster, [str, gdal.Dataset, [str, gdal.Dataset]], "raster")

    rasters = gdal_utils.get_path_from_dataset(core_utils.ensure_list(raster))
    assert gdal_utils.is_raster_list(raster), f"List contains invalid rasters. {raster}"

    nodata_values = []
    for internal_raster in rasters:
        raster_metadata = core_raster._raster_to_metadata(internal_raster)

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
    suffix="_nodata_set",
    creation_options=None,
):
    """
    Sets all the nodata for raster(s) to a value.

    ## Args:
    `raster` (_str_/_gdal.Dataset_/_list_): The raster(s) to set nodata values for. </br>
    `dst_nodata` (_float_/_int_/_str_/_None_): The target nodata value. If 'infer' the nodata
        value is set based on the input datatype. A list of nodata values can be based matching
        the amount of input rasters. If multiple nodata values should be set,
        use raster_mask_values. </br>

    ## Kwargs:
    `out_path` (_str_/_list_/_None_): The destination of the changed rasters. (Default: **None**) </br>
    `overwrite` (_bool_): Should the rasters be overwritten if they already exist? (Default: **True**) </br>
    `in_place` (_bool_): Should the rasters be changed in_place or copied? (Default: **False**) </br>
    `prefix` (_str_): Prefix to add to the output. (Default: **""**) </br>
    `suffix` (_str): Suffix to add to the output. (Default: **""**) </br>
    `creation_options` (_list_): Creation options for the output rasters. (Default: **None**) </br>

    ## Returns:
    (_str_/_list_): Returns the rasters with nodata set. If in_place is True a reference to the
    changed orignal is returned, otherwise a copied memory raster or the path to the
    generated raster is outputted.
    """
    core_utils.type_check(raster, [str, gdal.Dataset, [str, gdal.Dataset]], "raster")
    core_utils.type_check(dst_nodata, [float, int, str, list, None], "dst_nodata")
    core_utils.type_check(out_path, [list, str, None], "out_path")
    core_utils.type_check(overwrite, [bool], "overwrite")
    core_utils.type_check(prefix, [str], "prefix")
    core_utils.type_check(suffix, [str], "postfix")
    core_utils.type_check(creation_options, [[str], None], "creation_options")

    raster_list = core_utils.ensure_list(raster)
    path_list = gdal_utils.create_output_path_list(
        raster_list,
        out_path,
        prefix=prefix,
        suffix=suffix,
        overwrite=overwrite,
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

    for index, internal_raster in enumerate(raster_list):

        raster_metadata = None
        if len(rasters_metadata) == 0:
            raster_metadata = core_raster._raster_to_metadata(internal_raster)

            if not isinstance(raster_metadata, dict):
                raise Exception("Metadata is in the wrong format.")

            rasters_metadata.append(raster_metadata)
        else:
            raster_metadata = rasters_metadata[index]

        if dst_nodata == "infer":
            internal_dst_nodata = gdal_enums.translate_gdal_dtype_to_str(
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
            raster_mem = None

            if out_path is None:
                raster_mem = gdal_utils.save_dataset_to_memory(internal_raster)
            else:
                core_utils.remove_if_required(path_list[index], overwrite)
                raster_mem = gdal_utils.save_dataset_to_disk(internal_raster, path_list[index], creation_options=creation_options)

            for band in range(raster_metadata["bands"]):
                raster_mem_ref = core_raster._open_raster(raster_mem)
                raster_band = raster_mem_ref.GetRasterBand(band + 1)
                raster_band.SetNodataValue(internal_dst_nodata)

    if isinstance(raster, list):
        return output_rasters

    return output_rasters[0]


def raster_remove_nodata(
    raster,
    out_path=None,
    *,
    overwrite=True,
    in_place=False,
    prefix="",
    suffix="_nodata_removed",
    creation_options=None,
):
    """
    Removes all the nodata from a raster or a list of rasters.

    ## Args:
    `raster` (_str_/_gdal.Dataset_/_list_): The raster(s) to remove nodata values for. </br>

    ## Kwargs:
    `out_path` (_str_/_list_/_None_): The destination of the changed rasters. (Default: **None**) </br>
    `overwrite` (_bool_): Should the rasters be overwritten if they already exist? (Default: **True**) </br>
    `in_place` (_bool_): Should the rasters be changed in_place or copied? (Default: **False**) </br>
    `prefix` (_str_): Prefix to add to the output. (Default: **""**) </br>
    `suffix` (_str_): Suffix to add to the output. (Default: **"_nodata_removed"**) </br>
    `creation_options` (_list_): Creation options for the output rasters. (Default: **None**) </br>

    ## Returns:
    (_str_/_list_): Returns the rasters with nodata removed. If in_place is True a reference to the
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
        suffix=suffix,
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
    prefix="",
    suffix="",
    add_uuid=False,
    creation_options=[],
):
    """
    Mask a raster with a list of values.

    ## Args:
    `raster` (_str_/_gdal.Dataset_/_list_): The raster(s) to mask. </br>
    `values_to_mask` (_list_): The values to mask. </br>

    ## Kwargs:
    `include_original_nodata` (_bool_): Should the nodata_value of the input raster be added
    to the list of masked values? (Default: **True**) </br>
    `dst_nodata` (_float_/_int_/_str_/_list_): The nodata value to use for the output raster.
    If infer, the nodata_value from the input raster is used. (Default: **"infer"**) </br>
    `out_path` (_str_/_list_/_None_): The destination of the changed rasters.
    If out_paths are specified, in_place is automatically set to False. The path can be a folder. (Default: **None**)</br>
    `in_place` (_bool_): Should the rasters be changed in_place or copied? (Default: **False**) </br>
    `overwrite` (_bool_): If the output path exists already, should it be overwritten? (Default: **True**)</br>
    `prefix` (_str_): Prefix to add to the output. (Default: **""**) </br>
    `suffix` (_str_): Suffix to add to the output. (Default: **""**) </br>
    `add_uuid` (_bool_): Should a uuid be added to the output path? (Default: **False**) </br>
    `creation_options` (_list_/_None_): The GDAL creation options to be passed. (Default: **None**) </br>

    Returns:
    (_str_/_list_): Returns the rasters with nodata removed. If in_place is True a reference to the
    changed orignal is returned, otherwise a copied memory raster or the path to the
    generated raster is outputted.
    """
    core_utils.type_check(raster, [str, gdal.Dataset, [str, gdal.Dataset]], "raster")
    core_utils.type_check(values_to_mask, [[int, float, None]], "values_to_mask")
    core_utils.type_check(out_path, [[str], str, None], "out_path")
    core_utils.type_check(include_original_nodata, [bool], "include_original_nodata")
    core_utils.type_check(dst_nodata, [float, int, str, [float, int, str, None], None], "dst_nodata")
    core_utils.type_check(in_place, [bool], "in_place")
    core_utils.type_check(overwrite, [bool], "overwrite")
    core_utils.type_check(prefix, [str], "prefix")
    core_utils.type_check(suffix, [str], "postfix")
    core_utils.type_check(creation_options, [[str], None], "creation_options")

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

    raster_list = core_utils.ensure_list(raster)
    out_paths = gdal_utils.create_output_path_list(
        raster_list,
        out_path=out_path,
        overwrite=overwrite,
        prefix=prefix,
        suffix=suffix,
        add_uuid=add_uuid,
    )


    output_rasters = []

    for index, internal_raster in enumerate(raster_list):

        raster_metadata = None
        if len(rasters_metadata) == 0:
            raster_metadata = core_raster._raster_to_metadata(internal_raster)
            rasters_metadata.append(raster_metadata)
        else:
            raster_metadata = rasters_metadata[index]

        if dst_nodata == "infer":
            internal_dst_nodata = gdal_enums.translate_gdal_dtype_to_str(
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

        arr = core_raster.raster_to_array(internal_raster, filled=True)

        mask = None
        for idx, mask_value in enumerate(mask_values):
            if idx == 0:
                mask = arr == mask_value
            else:
                mask = mask/arr == mask_value

        arr = np.ma.masked_array(arr, mask=mask, fill_value=internal_dst_nodata)

        if internal_in_place:
            for band in range(raster_metadata["bands"]):
                raster_band = internal_raster.GetRasterBand(band + 1)
                raster_band.WriteArray(arr[:, :, band])
                raster_band = None
        else:
            out_name = out_paths[index]
            core_utils.remove_if_required(out_name, overwrite)

            output_rasters.append(
                core_raster.array_to_raster(arr, internal_raster, out_path=out_name)
            )

    if isinstance(raster, list):
        return output_rasters

    return output_rasters[0]
