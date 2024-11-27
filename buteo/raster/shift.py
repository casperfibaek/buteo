"""### Shift rasters. ###

Module to shift the location of rasters in geographic coordinates.
"""

# Standard library
from typing import Union, Optional, List

# External
import numpy as np
from osgeo import gdal

# Internal
from buteo.utils import (
    utils_base,
    utils_gdal,
    utils_path,
    utils_io,
)
from buteo.raster import core_raster, core_raster_io
from buteo.array.convolution import convolve_array_simple
from buteo.array.convolution_kernels import kernel_shift



def _raster_shift(
    raster: Union[str, gdal.Dataset],
    shift_list: List[Union[int, float]],
    out_path: Optional[str] = None,
    *,
    overwrite: bool = True,
    creation_options: Optional[List[str]] = None,
) -> Union[str, gdal.Dataset]:
    """Internal."""
    assert isinstance(shift_list, (list, tuple)), f"shift_list must be a list or a tuple. {shift_list}"
    assert len(shift_list) == 2, f"shift_list must be a list or tuple with len 2 (x_shift, y_shift): {shift_list}"

    for shift in shift_list:
        assert isinstance(shift, (int, float)), f"shift must be an int or a float: {shift}"

    ref = core_raster._open_raster(raster)
    metadata = core_raster.get_metadata_raster(ref)

    x_shift, y_shift = shift_list

    if out_path is None:
        out_path = utils_path._get_temp_filepath("shifted_raster.tif")
    else:
        if not utils_path._check_is_valid_output_filepath(out_path, overwrite=overwrite):
            raise ValueError(f"out_path is not a valid output path: {out_path}")

    utils_io._delete_if_required(out_path, overwrite)

    driver = gdal.GetDriverByName(utils_gdal._get_raster_driver_name_from_path(out_path))

    shifted = driver.Create(
        out_path,  # Location of the saved raster, ignored if driver is memory.
        metadata["width"],  # Dataframe width in pixels (e.g. 1920px).
        metadata["height"],  # Dataframe height in pixels (e.g. 1280px).
        metadata["bands"],  # The number of bands required.
        metadata["dtype_gdal"],  # Datatype of the destination.
        utils_gdal._get_default_creation_options(creation_options),
    )

    new_transform = list(metadata["geotransform"])
    new_transform[0] += x_shift
    new_transform[3] += y_shift

    shifted.SetGeoTransform(new_transform)
    shifted.SetProjection(metadata["projection_wkt"])

    src_nodata = metadata["nodata_value"]

    for band in range(metadata["bands"]):
        origin_raster_band = ref.GetRasterBand(band + 1)
        target_raster_band = shifted.GetRasterBand(band + 1)

        target_raster_band.WriteArray(origin_raster_band.ReadAsArray())

        if src_nodata is not None:
            if utils_base._check_variable_is_int(src_nodata):
                target_raster_band.SetNoDataValue(int(src_nodata))
            else:
                target_raster_band.SetNoDataValue(float(src_nodata))

    shifted.FlushCache()
    shifted = None
    ref = None

    return out_path


def raster_shift(
    raster: Union[str, gdal.Dataset, List],
    shift_list: List[Union[int, float]],
    out_path: Optional[str] = None,
    *,
    overwrite: bool = True,
    prefix: str = "",
    suffix: str = "",
    add_uuid: bool = False,
    add_timestamp: bool = False,
    creation_options: Optional[List[str]] = None,
) -> Union[str, List[str], gdal.Dataset]:
    """Shifts a raster in a given direction. (The frame is shifted)

    Parameters
    ----------
    raster : Union[str, List, gdal.Dataset]
        The raster(s) to be shifted.

    shift_list : List[Union[int, float]]
        The shift in x and y direction.

    out_path : Optional[str], optional
        The path to the output raster. If None, the raster is created in memory., default: None

    overwrite : bool, optional
        If True, the output raster will be overwritten if it already exists., default: True

    prefix : str, optional
        The prefix to be added to the output raster name., default: ""

    suffix : str, optional
        The suffix to be added to the output raster name., default: ""

    add_uuid : bool, optional
        If True, a unique identifier will be added to the output raster name., default: False

    add_timestamp : bool, optional
        If True, a timestamp will be added to the output raster name., default: False

    creation_options : Optional[List[str]], optional
        The creation options to be used when creating the output., default: None

    Returns
    -------
    Union[str, List[str]]
        The path(s) to the shifted raster(s).
    """
    utils_base._type_check(raster, [str, gdal.Dataset, [str, gdal.Dataset]], "raster")
    utils_base._type_check(shift_list, [[int, float], tuple], "shift_list")
    utils_base._type_check(out_path, [str, [str], None], "out_path")
    utils_base._type_check(overwrite, [bool], "overwrite")
    utils_base._type_check(creation_options, [[str], None], "creation_options")
    utils_base._type_check(prefix, [str], "prefix")
    utils_base._type_check(suffix, [str], "suffix")
    utils_base._type_check(add_uuid, [bool], "add_uuid")
    utils_base._type_check(add_timestamp, [bool], "add_timestamp")

    input_is_list = isinstance(raster, list)

    in_paths = utils_io._get_input_paths(raster, "raster")
    out_paths = utils_io._get_output_paths(
        in_paths,
        out_path,
        overwrite=overwrite,
        prefix=prefix,
        suffix=suffix,
        add_uuid=add_uuid,
        add_timestamp=add_timestamp,
        change_ext="tif",
    )

    utils_io._check_overwrite_policy(out_paths, overwrite)
    utils_io._delete_if_required_list(out_paths, overwrite)

    shifted_rasters = []
    for idx, in_raster in enumerate(in_paths):
        shifted_rasters.append(
            _raster_shift(
                in_raster,
                shift_list,
                out_path=out_paths[idx],
                overwrite=overwrite,
                creation_options=creation_options,
            )
        )

    if input_is_list:
        return shifted_rasters

    return shifted_rasters[0]


def raster_shift_pixel(
    raster: Union[str, gdal.Dataset],
    shift_list: List[Union[int, float]],
    out_path: Optional[str] = None,
) -> str:
    """Shifts a raster in a given direction. (The pixels are shifted, not the frame)

    Parameters
    ----------
    raster : Union[str, gdal.Dataset]
        The raster to be shifted.

    shift_list : List[Union[int, float]]
        The shift in x and y direction.

    out_path : Optional[str], optional
        The path to the output raster. If None, the raster is created in memory., default: None

    Returns
    -------
    str
        The path to the shifted raster.
    """
    utils_base._type_check(raster, [str, gdal.Dataset], "raster")
    utils_base._type_check(shift_list, [[int, float], tuple], "shift_list")
    utils_base._type_check(out_path, [str, None], "out_path")

    arr = core_raster_io.raster_to_array(raster)

    offsets, weights = kernel_shift(shift_list[0], shift_list[1])

    arr_float32 = arr.astype(np.float32, copy=False)

    for channel in range(arr.shape[2]):
        arr[:, :, channel] = convolve_array_simple(
            arr_float32[:, :, channel], offsets, weights,
        )

    if out_path is None:
        out_path = utils_path._get_temp_filepath("shifted_raster.tif")
    else:
        if not utils_path._check_is_valid_output_filepath(out_path):
            raise ValueError(f"out_path is not a valid output path: {out_path}")

    utils_io._delete_if_required(out_path, overwrite=True)

    return core_raster_io.array_to_raster(
        arr,
        reference=raster,
        out_path=out_path,
    )
