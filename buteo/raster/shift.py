"""
Module to shift a raster

TODO:
    - Remove typings
    - Improve documentations
"""

import sys; sys.path.append("../../") # Path: buteo/raster/shift.py
from uuid import uuid4
from typing import Union, Tuple, List, Optional

from osgeo import gdal

from buteo.utils.project_types import Number
from buteo.utils.core import remove_if_overwrite, is_number, type_check
from buteo.utils.gdal_utils import path_to_driver_raster, default_options
from buteo.raster.io import open_raster, raster_to_metadata


def shift_raster(
    raster: Union[gdal.Dataset, str],
    shift: Union[Number, Tuple[Number, Number], List[Number]],
    out_path: Optional[str] = None,
    overwrite: bool = True,
    creation_options: list = [],
) -> Union[gdal.Dataset, str]:
    """Shifts a raster in a given direction.

    Returns:
        A raster. If an out_path is given the output is a string containing
        the path to the newly created raster.
    """
    type_check(raster, [list, str, gdal.Dataset], "raster")
    type_check(shift, [tuple, list], "shift")
    type_check(out_path, [list, str], "out_path", allow_none=True)
    type_check(overwrite, [bool], "overwrite")
    type_check(creation_options, [list], "creation_options")

    ref = open_raster(raster)
    metadata = raster_to_metadata(ref)

    x_shift: float = 0.0
    y_shift: float = 0.0
    if isinstance(shift, tuple) or isinstance(shift, list):
        if len(shift) == 1:
            if is_number(shift[0]):
                x_shift = float(shift[0])
                y_shift = float(shift[0])
            else:
                raise ValueError("shift is not a number or a list/tuple of numbers.")
        elif len(shift) == 2:
            if is_number(shift[0]) and is_number(shift[1]):
                x_shift = float(shift[0])
                y_shift = float(shift[1])
        else:
            raise ValueError("shift is either empty or larger than 2.")
    elif is_number(shift):
        x_shift = float(shift)
        y_shift = float(shift)
    else:
        raise ValueError("shift is invalid.")

    out_name = None
    out_format = None
    out_creation_options = []
    if out_path is None:
        raster_name = metadata["basename"]
        out_name = f"/vsimem/{raster_name}_{uuid4().int}_resampled.tif"
        out_format = "GTiff"
    else:
        out_creation_options = default_options(creation_options)
        out_name = out_path
        out_format = path_to_driver_raster(out_path)

    remove_if_overwrite(out_path, overwrite)

    driver = gdal.GetDriverByName(out_format)

    shifted = driver.Create(
        out_name,  # Location of the saved raster, ignored if driver is memory.
        metadata["width"],  # Dataframe width in pixels (e.g. 1920px).
        metadata["height"],  # Dataframe height in pixels (e.g. 1280px).
        metadata["band_count"],  # The number of bands required.
        metadata["datatype_gdal_raw"],  # Datatype of the destination.
        out_creation_options,
    )

    new_transform = list(metadata["transform"])
    new_transform[0] += x_shift
    new_transform[3] += y_shift

    shifted.SetGeoTransform(new_transform)
    shifted.SetProjection(metadata["projection"])

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
