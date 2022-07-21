"""
### Handle and create borders on rasters. ###

Functions to add or remove borders from rasters.
Useful for warped satellite images and for proximity searching.

TODO:
    * Remove near black borders
    * Add functions for rasters. (e.g. add_border_to_array)
    * Support multiple rasters (create internal version)
"""

# Standard library
import sys; sys.path.append("../../")

# External
import numpy as np
from osgeo import gdal

# Internal
from buteo.raster import core_raster
from buteo.utils import gdal_utils, core_utils, gdal_enums


def _add_border_to_raster(
    raster,
    out_path=None,
    border_size=100,
    *,
    border_size_unit="px",
    border_value=0,
    overwrite=True,
    creation_options=None,
):
    """ Internal. """
    in_raster = core_raster.open_raster(raster)
    metadata = core_raster.raster_to_metadata(in_raster)

    # Parse the driver
    driver_name = "GTiff" if out_path is None else gdal_utils.path_to_driver_raster(out_path)
    if driver_name is None:
        raise ValueError(f"Unable to parse filetype from path: {out_path}")

    driver = gdal.GetDriverByName(driver_name)
    if driver is None:
        raise ValueError(f"Error while creating driver from extension: {out_path}")

    output_name = None
    if out_path is None:
        output_name = gdal_utils.create_memory_path("raster_proximity.tif", add_uuid=True)
    else:
        output_name = out_path

    in_arr = core_raster.raster_to_array(in_raster)

    if border_size_unit == "px":
        border_size_y = border_size
        border_size_x = border_size
        new_shape = (
            in_arr.shape[0] + (2 * border_size_y),
            in_arr.shape[1] + (2 * border_size_x),
            in_arr.shape[2],
        )
    else:
        border_size_y = round(border_size / metadata["pixel_height"])
        border_size_x = round(border_size / metadata["pixel_width"])
        new_shape = (
            in_arr.shape[0] + (2 * border_size_y),
            in_arr.shape[1] + (2 * border_size_x),
            in_arr.shape[2],
        )

    new_arr = np.full(new_shape, border_value, dtype=in_arr.dtype)
    new_arr[border_size_y:-border_size_y, border_size_x:-border_size_x, :] = in_arr

    if isinstance(in_arr, np.ma.MaskedArray):
        mask = np.zeros(new_shape, dtype=bool)
        mask[
            border_size_y:-border_size_y, border_size_x:-border_size_x, :
        ] = in_arr.mask
        new_arr = np.ma.array(new_arr, mask=mask)
        new_arr.fill_value = in_arr.fill_value

    core_utils.remove_if_required(out_path, overwrite)

    dest_raster = driver.Create(
        output_name,
        new_shape[1],
        new_shape[0],
        metadata["band_count"],
        gdal_enums.translate_str_to_gdal_dtype(in_arr.dtype),
        gdal_utils.default_creation_options(creation_options),
    )

    og_transform = in_raster.GetGeoTransform()

    new_transform = []
    for i in og_transform:
        new_transform.append(i)

    new_transform[0] -= border_size_x * og_transform[1]
    new_transform[3] -= border_size_y * og_transform[5]

    dest_raster.SetGeoTransform(new_transform)
    dest_raster.SetProjection(in_raster.GetProjectionRef())

    for band_num in range(1, metadata["band_count"] + 1):
        dst_band = dest_raster.GetRasterBand(band_num)
        dst_band.WriteArray(new_arr[:, :, band_num - 1])

        if metadata["has_nodata"]:
            dst_band.SetNoDataValue(metadata["nodata_value"])

    return output_name


def add_border_to_raster(
    raster,
    out_path=None,
    border_size=100,
    *,
    border_size_unit="px",
    border_value=0,
    allow_lists=True,
    overwrite=True,
    creation_options=None,
):
    """
    Add a border to a raster.

    ## Args:
    `input_raster` (_str_/_gdal.DataSet_): The input raster.

    ## Kwargs:
    `out_path` (_str_/_None_): The output path. If **None** the output will be a memory raster. </br>
    `border_size` (_int_): The size of the border. </br>
    `border_size_unit` (_str_): The unit of the border size. </br>
    `border_value` (_int_): The value of the border. </br>
    `overwrite` (_bool_): If **True**, the output raster will be overwritten. </br>
    `creation_options` (_list_/_None_): Creation options for the output raster. </br>

    ## Returns:
    `out_raster` (_str_/_gdal.DataSet_): The output raster with added borders.
    """
    core_utils.type_check(raster, [str, gdal.Dataset, [str, gdal.Dataset]], "raster")
    core_utils.type_check(out_path, [str, None], "out_path")
    core_utils.type_check(border_size, [int], "border_size")
    core_utils.type_check(border_size_unit, [str], "border_size_unit")
    core_utils.type_check(border_value, [int, float], "border_value")
    core_utils.type_check(overwrite, [bool], "overwrite")
    core_utils.type_check(creation_options, [list, None], "creation_options")

    if not allow_lists:
        if isinstance(raster, list):
            raise ValueError("Lists are not allowed as input.")

        return _add_border_to_raster(
            raster,
            out_path=out_path,
            border_size=border_size,
            border_size_unit=border_size_unit,
            border_value=border_value,
            overwrite=overwrite,
            creation_options=creation_options,
        )

    raster_list = core_utils.ensure_list(raster)

    if out_path is None:
        out_path = gdal_utils.create_memory_path("raster_proximity.tif", add_uuid=True)

    for raster_path in raster_list:
        _add_border_to_raster(
            raster_path,
            out_path=out_path,
            border_size=border_size,
            border_size_unit=border_size_unit,
            border_value=border_value,
            overwrite=overwrite,
            creation_options=creation_options,
        )

    return out_path
