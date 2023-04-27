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
from buteo.utils import core_utils, gdal_utils
from buteo.raster import core_raster
from buteo.array.convolution import convolve_array_simple, _simple_shift_kernel_2d


def _shift_raster(
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

    ref = core_raster._open_raster(raster)
    metadata = core_raster._raster_to_metadata(ref)

    x_shift, y_shift = shift_list

    if out_path is None:
        raster_name = metadata["basename"]
        out_path = gdal_utils.create_memory_path(raster_name, add_uuid=True)
    else:
        if not core_utils.is_valid_output_path(out_path, overwrite=overwrite):
            raise ValueError(f"out_path is not a valid output path: {out_path}")

    core_utils.remove_if_required(out_path, overwrite)

    driver = gdal.GetDriverByName(gdal_utils.path_to_driver_raster(out_path))

    shifted = driver.Create(
        out_path,  # Location of the saved raster, ignored if driver is memory.
        metadata["width"],  # Dataframe width in pixels (e.g. 1920px).
        metadata["height"],  # Dataframe height in pixels (e.g. 1280px).
        metadata["band_count"],  # The number of bands required.
        metadata["datatype_gdal_raw"],  # Datatype of the destination.
        gdal_utils.default_creation_options(creation_options),
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


def shift_raster(
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

    Args:
        raster (Union[str, List, gdal.Dataset]): The raster(s) to be shifted.
        shift_list (List[Union[int, float]]): The shift in x and y direction.

    Keyword Args:
        out_path (Optional[str]='None'): The path to the output raster. If None, the raster is
            created in memory.
        overwrite (bool=True): If True, the output raster will be overwritten if it already exists.
        prefix (str=''): The prefix to be added to the output raster name.
        suffix (str=''): The suffix to be added to the output raster name.
        add_uuid (bool=False): If True, a unique identifier will be added to the output raster name.
        creation_options (Optional[List[str]]='None'): The creation options to be used when creating the output.

    Returns:
        Union[str, List[str], gdal.Dataset]: The path(s) to the shifted raster(s).
    """
    core_utils.type_check(raster, [str, gdal.Dataset, [str, gdal.Dataset]], "raster")
    core_utils.type_check(shift_list, [[tuple, list]], "shift_list")
    core_utils.type_check(out_path, [str, [str], None], "out_path")
    core_utils.type_check(overwrite, [bool], "overwrite")
    core_utils.type_check(creation_options, [[str], None], "creation_options")

    raster_list = core_utils.ensure_list(raster)
    assert gdal_utils.is_raster_list(raster_list), f"Invalid raster in raster list: {raster_list}"

    path_list = gdal_utils.create_output_path_list(
        raster_list,
        out_path=out_path,
        overwrite=overwrite,
        prefix=prefix,
        suffix=suffix,
        add_uuid=add_uuid,
    )

    shifted_rasters = []
    for index, in_raster in enumerate(raster_list):
        shifted_rasters.append(
            _shift_raster(
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


def shift_raster_pixel(
    raster: Union[str, gdal.Dataset],
    shift_list: List[Union[int, float]],
    nodata_value: Union[int, float] = -9999.9,
    out_path: Optional[str] = None,
) -> str:
    """
    Shifts a raster in a given direction. (The pixels are shifted, not the frame)

    Args:
        raster (Union[str, gdal.Dataset]): The raster to be shifted.
        shift_list (List[Union[int, float]]): The shift in x and y direction.

    Keyword Args:
        nodata_value (Union[int, float]=-9999.9): The nodata value to use when shifting pixels.
        out_path (Optional[str]='None'): The path to the output raster. If None, the raster is
            created in memory.

    Returns:
        str: The path to the shifted raster.
    """
    core_utils.type_check(raster, [str, gdal.Dataset], "raster")
    core_utils.type_check(shift_list, [[tuple, list]], "shift_list")
    core_utils.type_check(out_path, [str, None], "out_path")

    arr = core_raster.raster_to_array(raster)

    offsets, weights = _simple_shift_kernel_2d(shift_list[0], shift_list[1])

    arr_float32 = arr.astype(np.float32)

    for channel in arr.shape[2]:
        arr[:, :, channel] = convolve_array_simple(
            arr_float32[:, :, channel], offsets, weights, nodata_value,
        )

    if out_path is None:
        out_path = gdal_utils.create_memory_path("shifted_raster", add_uuid=True)

    return core_raster.array_to_raster(
        arr,
        reference=raster,
        out_path=out_path,
    )
