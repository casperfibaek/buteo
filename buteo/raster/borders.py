import sys
import numpy as np
from osgeo import gdal
from uuid import uuid4

sys.path.append("../../")
from buteo.raster.io import (
    open_raster,
    path_to_driver_raster,
    raster_to_array,
    raster_to_metadata,
)
from buteo.gdal_utils import (
    numpy_to_gdal_datatype,
    default_options,
)
from buteo.utils import remove_if_overwrite


def add_border_to_raster(
    input_raster,
    out_path=None,
    border_size=1,
    border_size_unit_px=True,
    border_value=0,
    overwrite: bool = True,
    creation_options: list = [],
):
    in_raster = open_raster(input_raster)
    metadata = raster_to_metadata(in_raster)

    # Parse the driver
    driver_name = "GTiff" if out_path is None else path_to_driver_raster(out_path)
    if driver_name is None:
        raise ValueError(f"Unable to parse filetype from path: {out_path}")

    driver = gdal.GetDriverByName(driver_name)
    if driver is None:
        raise ValueError(f"Error while creating driver from extension: {out_path}")

    output_name = None
    if out_path is None:
        output_name = f"/vsimem/raster_proximity_{uuid4().int}.tif"
    else:
        output_name = out_path

    in_arr = raster_to_array(in_raster)

    if border_size_unit_px:
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

    remove_if_overwrite(out_path, overwrite)

    dest_raster = driver.Create(
        output_name,
        new_shape[1],
        new_shape[0],
        metadata["band_count"],
        numpy_to_gdal_datatype(in_arr.dtype),
        default_options(creation_options),
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
