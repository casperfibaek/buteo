import sys
import os
import numpy as np
from osgeo import gdal
from uuid import uuid4

sys.path.append("../../")
from buteo.raster.io import (
    open_raster,
    path_to_driver,
    raster_to_array,
    array_to_raster,
    raster_to_metadata,
    ready_io_raster,
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
    border_value=0,
    overwrite: bool = True,
    creation_options: list = [],
):
    in_raster = open_raster(input_raster)
    metadata = raster_to_metadata(in_raster)

    # Parse the driver
    driver_name = "GTiff" if out_path is None else path_to_driver(out_path)
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

    new_shape = (
        in_arr.shape[0] + (2 * border_size),
        in_arr.shape[1] + (2 * border_size),
        in_arr.shape[2],
    )

    new_arr = np.full(new_shape, border_value, dtype=in_arr.dtype)
    new_arr[border_size:-border_size, border_size:-border_size, :] = in_arr

    if isinstance(in_arr, np.ma.MaskedArray):
        mask = np.zeros(new_shape, dtype=bool)
        mask[border_size:-border_size, border_size:-border_size, :] = in_arr.mask
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

    new_transform[0] -= border_size * og_transform[1]
    new_transform[3] -= border_size * og_transform[5]

    dest_raster.SetGeoTransform(new_transform)
    dest_raster.SetProjection(in_raster.GetProjectionRef())

    for band_num in range(1, metadata["band_count"] + 1):
        dst_band = dest_raster.GetRasterBand(band_num)
        dst_band.WriteArray(new_arr[:, :, band_num - 1])

        if metadata["has_nodata"]:
            dst_band.SetNoDataValue(metadata["nodata_value"])

    return output_name


def calc_proximity(
    input_rasters,
    target_value=1,
    out_path=None,
    max_dist=1000,
    add_border=False,
    weighted=False,
    invert=False,
    return_array=False,
    postfix="_proximity",
    uuid=False,
    overwrite=True,
    skip_existing=False,
):
    """
    Calculate the proximity of input_raster to values
    """
    raster_list, path_list = ready_io_raster(
        input_rasters, out_path, overwrite, postfix=postfix, uuid=uuid
    )

    output = []
    for index, input_raster in enumerate(raster_list):
        out_path = path_list[index]

        if skip_existing and os.path.exists(out_path):
            output.append(out_path)
            continue

        in_arr = raster_to_array(input_raster, filled=True)
        bin_arr = (in_arr != target_value).astype("uint8")
        bin_raster = array_to_raster(bin_arr, reference=input_raster)

        in_raster = open_raster(bin_raster)
        in_raster_path = bin_raster

        if add_border:
            border_size = 1
            border_raster = add_border_to_raster(
                in_raster,
                border_size=border_size,
                border_value=0,
                overwrite=True,
            )

            in_raster = open_raster(border_raster)

            gdal.Unlink(in_raster_path)
            in_raster_path = border_raster

        src_band = in_raster.GetRasterBand(1)

        driver_name = "GTiff" if out_path is None else path_to_driver(out_path)
        if driver_name is None:
            raise ValueError(f"Unable to parse filetype from path: {out_path}")

        driver = gdal.GetDriverByName(driver_name)
        if driver is None:
            raise ValueError(f"Error while creating driver from extension: {out_path}")

        mem_path = f"/vsimem/raster_proximity_tmp_{uuid4().int}.tif"

        dest_raster = driver.Create(
            mem_path,
            in_raster.RasterXSize,
            in_raster.RasterYSize,
            1,
            gdal.GetDataTypeByName("Float32"),
        )

        dest_raster.SetGeoTransform(in_raster.GetGeoTransform())
        dest_raster.SetProjection(in_raster.GetProjectionRef())
        dst_band = dest_raster.GetRasterBand(1)

        gdal.ComputeProximity(
            src_band,
            dst_band,
            [
                f"VALUES='1'",
                "DISTUNITS=GEO",
                f"MAXDIST={max_dist}",
            ],
        )

        dst_arr = dst_band.ReadAsArray()
        gdal.Unlink(mem_path)
        gdal.Unlink(in_raster_path)

        dst_arr = np.where(dst_arr > max_dist, max_dist, dst_arr)

        if invert:
            dst_arr = max_dist - dst_arr

        if weighted:
            dst_arr = dst_arr / max_dist

        if add_border:
            dst_arr = dst_arr[border_size:-border_size, border_size:-border_size]

        src_band = None
        dst_band = None
        in_raster = None
        dest_raster = None

        if return_array:
            output.append(dst_arr)
        else:
            array_to_raster(dst_arr, reference=input_raster, out_path=out_path)
            output.append(out_path)

        dst_arr = None

    if isinstance(input_rasters, list):
        return output

    return output[0]


# folder = "C:/Users/caspe/Desktop/test_area/"

# calc_proximity(
#     folder + "clip_test.tif",
#     -9999.0,
#     folder + "proximity_02.tif",
#     max_dist=5000,
#     invert=False,
#     weighted=True,
#     add_border=True,
# )
