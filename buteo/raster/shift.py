"""
### Shift rasters. ###

Module to shift the location of rasters in geographic coordinates.
"""

# Standard library
import sys; sys.path.append("../../")
from typing import Union, Optional, List

# External
from osgeo import gdal
import numpy as np

# Internal
from buteo.utils import utils_base, utils_gdal, utils_path
from buteo.raster import core_raster
from buteo.array.convolution import convolve_array_simple
from buteo.array.convolution_kernels import kernel_shift


def _raster_shift(
    raster: Union[str, gdal.Dataset, List],
    shift_list: List[Union[int, float]],
    out_path: Optional[str] = None,
    *,
    overwrite: bool = True,
    creation_options: Optional[List[str]] = None,
) -> Union[str, gdal.Dataset]:
    """ Internal. """
    assert isinstance(shift_list, (list, tuple)), f"shift_list must be a list or a tuple. {shift_list}"
    assert len(shift_list) == 2, f"shift_list must be a list or tuple with len 2 (x_shift, y_shift): {shift_list}"

    for shift in shift_list:
        assert isinstance(shift, (int, float)), f"shift must be an int or a float: {shift}"

    ref = core_raster._raster_open(raster)
    metadata = core_raster._raster_to_metadata(ref)

    x_shift, y_shift = shift_list

    if out_path is None:
        raster_name = metadata["basename"]
        out_path = utils_path._get_output_path(raster_name, add_uuid=True)
    else:
        if not utils_base.is_valid_output_path(out_path, overwrite=overwrite):
            raise ValueError(f"out_path is not a valid output path: {out_path}")

    utils_path._delete_if_required(out_path, overwrite)

    driver = gdal.GetDriverByName(utils_gdal._get_raster_driver_from_path(out_path))

    shifted = driver.Create(
        out_path,  # Location of the saved raster, ignored if driver is memory.
        metadata["width"],  # Dataframe width in pixels (e.g. 1920px).
        metadata["height"],  # Dataframe height in pixels (e.g. 1280px).
        metadata["band_count"],  # The number of bands required.
        metadata["datatype_gdal_raw"],  # Datatype of the destination.
        utils_gdal._get_default_creation_options(creation_options),
    )

    new_transform = list(metadata["transform"])
    new_transform[0] += x_shift
    new_transform[3] += y_shift

    shifted.SetGeoTransform(new_transform)
    shifted.SetProjection(metadata["projection_wkt"])

    src_nodata = metadata["nodata_value"]

    for band in range(metadata["band_count"]):
        origin_raster_band = ref.GetRasterBand(band + 1)
        target_raster_band = shifted.GetRasterBand(band + 1)

        target_raster_band.WriteArray(origin_raster_band.ReadAsArray())
        target_raster_band.SetNoDataValue(src_nodata)

    if out_path is not None:
        shifted = None
        return out_path
    else:
        return shifted


def raster_shift(
    raster: Union[str, gdal.Dataset, List],
    shift_list: List[Union[int, float]],
    out_path: Optional[str] = None,
    *,
    overwrite: bool = True,
    prefix: str = "",
    suffix: str = "",
    add_uuid: bool = False,
    creation_options: Optional[List[str]] = None,
) -> Union[str, List[str], gdal.Dataset]:
    """
    Shifts a raster in a given direction. (The frame is shifted)

    Parameters
    ----------
    raster : Union[str, List, gdal.Dataset]
        The raster(s) to be shifted.

    shift_list : List[Union[int, float]]
        The shift in x and y direction.

    out_path : Optional[str], optional
        The path to the output raster. If None, the raster is created in memory., by default None

    overwrite : bool, optional
        If True, the output raster will be overwritten if it already exists., by default True

    prefix : str, optional
        The prefix to be added to the output raster name., by default ""

    suffix : str, optional
        The suffix to be added to the output raster name., by default ""

    add_uuid : bool, optional
        If True, a unique identifier will be added to the output raster name., by default False

    creation_options : Optional[List[str]], optional
        The creation options to be used when creating the output., by default None

    Args:
        raster (Union[str, List, gdal.Dataset]): The raster(s) to be shifted.
        shift_list (List[Union[int, float]]): The shift in x and y direction.

    Returns
    -------
    Union[str, List[str]]
        The path(s) to the shifted raster(s).
    """
    utils_base.type_check(raster, [str, gdal.Dataset, [str, gdal.Dataset]], "raster")
    utils_base.type_check(shift_list, [[tuple, list]], "shift_list")
    utils_base.type_check(out_path, [str, [str], None], "out_path")
    utils_base.type_check(overwrite, [bool], "overwrite")
    utils_base.type_check(creation_options, [[str], None], "creation_options")

    raster_list = utils_base._get_variable_as_list(raster)
    assert utils_gdal._check_is_raster_list(raster_list), f"Invalid raster in raster list: {raster_list}"

    path_list = utils_gdal._parse_output_data(
        raster_list,
        output_data=out_path,
        overwrite=overwrite,
        prefix=prefix,
        suffix=suffix,
        add_uuid=add_uuid,
    )

    shifted_rasters = []
    for index, in_raster in enumerate(raster_list):
        shifted_rasters.append(
            _raster_shift(
                in_raster,
                shift_list,
                out_path=path_list[index],
                overwrite=overwrite,
                creation_options=creation_options,
            )
        )

    if isinstance(raster, list):
        return shifted_rasters

    return shifted_rasters[0]


def raster_shift_pixel(
    raster: Union[str, gdal.Dataset],
    shift_list: List[Union[int, float]],
    out_path: Optional[str] = None,
) -> str:
    """
    Shifts a raster in a given direction. (The pixels are shifted, not the frame)

    Parameters
    ----------
    raster : Union[str, gdal.Dataset]
        The raster to be shifted.

    shift_list : List[Union[int, float]]
        The shift in x and y direction.

    out_path : Optional[str], optional
        The path to the output raster. If None, the raster is created in memory., by default None

    Returns
    -------
    str
        The path to the shifted raster.
    """
    utils_base.type_check(raster, [str, gdal.Dataset], "raster")
    utils_base.type_check(shift_list, [[tuple, list]], "shift_list")
    utils_base.type_check(out_path, [str, None], "out_path")

    arr = core_raster.raster_to_array(raster)

    offsets, weights = kernel_shift(shift_list[0], shift_list[1])

    arr_float32 = arr.astype(np.float32)

    for channel in arr.shape[2]:
        arr[:, :, channel] = convolve_array_simple(
            arr_float32[:, :, channel], offsets, weights,
        )

    if out_path is None:
        out_path = utils_path._get_output_path("shifted_raster", add_uuid=True)

    return core_raster.array_to_raster(
        arr,
        reference=raster,
        out_path=out_path,
    )
