"""### Handle and create borders on rasters. ###

Functions to add or remove borders from rasters.
Useful for warped satellite images and for proximity searching.
"""

# TODO: Remove near black borders.

# Standard library
import sys; sys.path.append("../../")
from typing import Union, List, Optional, Any

# External
import numpy as np
from osgeo import gdal

# Internal
from buteo.utils import (
    utils_io,
    utils_gdal,
    utils_base,
    utils_path,
    utils_translate,
)
from buteo.raster import core_raster, core_raster_io


def _raster_add_border(
    raster: Union[str, gdal.Dataset],
    out_path: Optional[str] = None,
    border_size: int = 100,
    *,
    border_size_unit: str = "px",
    border_value: Union[int, float] = 0,
    overwrite: bool = True,
    creation_options: Optional[List[str]] = None,
):
    """Internal.
    Add a border to a raster.
    """
    in_raster = core_raster._raster_open(raster)
    metadata = core_raster._get_basic_metadata_raster(in_raster)

    # Parse the driver
    driver_name = "GTiff" if out_path is None else utils_gdal._get_raster_driver_name_from_path(out_path)
    if driver_name is None:
        raise ValueError(f"Unable to parse filetype from path: {out_path}")

    driver = gdal.GetDriverByName(driver_name)
    if driver is None:
        raise ValueError(f"Error while creating driver from extension: {out_path}")

    output_name = None
    if out_path is None:
        output_name = utils_path._get_temp_filepath("raster_proximity.tif", add_uuid=True, add_timestamp=True)
    else:
        assert utils_path._check_is_valid_output_filepath(out_path, overwrite=overwrite), (
            f"Output path is not valid: {out_path}"
        )
        output_name = out_path

    in_arr = core_raster_io.raster_to_array(in_raster)

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

    utils_path._delete_if_required(output_name, overwrite)

    dest_raster = driver.Create(
        output_name,
        new_shape[1],
        new_shape[0],
        metadata["bands"],
        utils_translate._translate_dtype_numpy_to_gdal(in_arr.dtype),
        utils_gdal._get_default_creation_options(creation_options),
    )

    og_transform = in_raster.GetGeoTransform()

    new_transform = []
    for i in og_transform:
        new_transform.append(i)

    new_transform[0] -= border_size_x * og_transform[1]
    new_transform[3] -= border_size_y * og_transform[5]

    dest_raster.SetGeoTransform(new_transform)
    dest_raster.SetProjection(in_raster.GetProjectionRef())

    for band_num in range(1, metadata["bands"] + 1):
        dst_band = dest_raster.GetRasterBand(band_num)
        dst_band.WriteArray(new_arr[:, :, band_num - 1])

        if metadata["nodata"]:
            dst_band.SetNoDataValue(metadata["nodata_value"])

    return output_name


def raster_add_border(
    raster: Union[str, gdal.Dataset, List[Union[str, gdal.Dataset]]],
    out_path: Optional[str] = None,
    border_size: int = 100,
    border_size_unit: str = "px",
    border_value: int = 0,
    prefix: str = "",
    suffix: str = "",
    add_uuid: bool = False,
    add_timestamp: bool = False,
    overwrite: bool = True,
    creation_options: Optional[List[str]] = None,
    ) -> Union[str, gdal.Dataset]:
    """Add a border to a raster.

    Parameters
    ----------
    raster : Union[str, gdal.Dataset, List[Union[str, gdal.Dataset]]]
        The input raster.

    out_path : Optional[str], optional
        The output path. If None, the output will be a memory raster.
        Default: None

    border_size : int, optional
        The size of the border. Default: 100

    border_size_unit : str, optional
        The unit of the border size. Default: 'px'

    border_value : int, optional
        The value of the border. Default: 0

    prefix : str, optional
        The prefix to add to the output file name, default: ""

    suffix : str, optional
        The suffix to add to the output file name, default: ""

    add_uuid : bool, optional
        Whether to add a uuid to the output file name, default: False

    add_timestamp : bool, optional
        Whether to add a timestamp to the output file name, default: False

    overwrite : bool, optional
        If True, the output raster will be overwritten. Default: True

    creation_options : Optional[List[str]], optional
        Creation options for the output raster. Default is None

    Returns
    -------
    Union[str, gdal.Dataset]
        The output raster with added borders.
    """
    utils_base._type_check(raster, [str, gdal.Dataset, [str, gdal.Dataset]], "raster")
    utils_base._type_check(out_path, [str, None], "out_path")
    utils_base._type_check(border_size, [int], "border_size")
    utils_base._type_check(border_size_unit, [str], "border_size_unit")
    utils_base._type_check(border_value, [int, float], "border_value")
    utils_base._type_check(prefix, [str], "prefix")
    utils_base._type_check(suffix, [str], "suffix")
    utils_base._type_check(add_uuid, [bool], "add_uuid")
    utils_base._type_check(add_timestamp, [bool], "add_timestamp")
    utils_base._type_check(overwrite, [bool], "overwrite")
    utils_base._type_check(creation_options, [list, None], "creation_options")

    input_is_list = isinstance(raster, list)

    input_rasters = utils_io._get_input_paths(raster, "raster")
    out_path = utils_io._get_output_paths(
        input_rasters,
        out_path,
        prefix=prefix,
        suffix=suffix,
        add_uuid=add_uuid,
        add_timestamp=add_timestamp,
        change_ext="tif",
        overwrite=overwrite,
    )

    utils_path._delete_if_required_list(out_path, overwrite)

    for idx, raster in enumerate(input_rasters):
        _raster_add_border(
            raster,
            out_path=out_path[idx],
            border_size=border_size,
            border_size_unit=border_size_unit,
            border_value=border_value,
            overwrite=overwrite,
            creation_options=creation_options,
        )

    if input_is_list:
        return out_path

    return out_path[0]
