"""
### Calculate distances on a raster. ###

Module to calculate the distance from a pixel value to other pixels.
"""

# Standard library
import sys; sys.path.append("../../")
from uuid import uuid4

# External
import numpy as np
from osgeo import gdal

# Internal
from buteo.utils import core_utils, gdal_utils
from buteo.raster import core_raster
from buteo.raster.borders import add_border_to_raster



def calc_proximity(
    input_rasters,
    target_value=1,
    unit="GEO",
    out_path=None,
    max_dist=1000,
    add_border=False,
    border_value=0,
    weighted=False,
    invert=False,
    return_array=False,
    prefix="",
    suffix="_proximity",
    add_uuid=False,
    overwrite=True,
):
    """
    Calculate the proximity of input_raster to values

    ## Args:
    `input_rasters` (_list_): A list of rasters to use as input. </br>

    ## Kwargs:
    `target_value` (_int_/_float_): The value to use as target. (Default: **1**) </br>
    `unit` (_str_): The unit to use for the distance. GEO or PIXEL. (Default: **"GEO"**) </br>
    `out_path` (_str_/_None_/_list_): The output path. (Default: **None**) </br>
    `max_dist` (_int_/_float_): The maximum distance to use. (Default: **1000**) </br>
    `add_border` (_bool_): If **True**, a border will be added to the raster. (Default: **False**) </br>
    `border_value` (_int_/_float_): The value to use for the border. (Default: **0**) </br>
    `weighted` (_bool_): If **True**, the distance will be divided by the max distance. (Default: **False**) </br>
    `invert` (_bool_): If **True**, the target will be inversed. (Default: **False**) </br>
    `return_array` (_bool_): If **True** a NumPy array will be returned instead of a raster. (Default: **False**) </br>
    `prefix` (_str_): Prefix to add to the output. (Default: **""**) </br>
    `suffix` (_str_): Suffix to add to the output. (Default: **""**) </br>
    `add_uuid` (_bool_): Should a uuid be added to the output path? (Default: **False**) </br>
    `creation_options` (_list_/_None_): The GDAL creation options to be passed. (Default: **None**) </br>
    `overwrite` (_bool_): If the output path exists already, should it be overwritten? (Default: **True**)</br>

    ## Returns:
    (_str_/_np.ndarray_): A path to a raster with the calculate proximity, or a numpy array with the data.
    """
    core_utils.type_check(input_rasters, [str, gdal.Dataset, [str, gdal.Dataset]], "input_rasters")
    core_utils.type_check(target_value, [int, float], "target_value")
    core_utils.type_check(out_path, [str, [str], None], "out_path")
    core_utils.type_check(max_dist, [int, float], "max_dist")
    core_utils.type_check(add_border, [bool], "add_border")
    core_utils.type_check(border_value, [int, float], "border_value")
    core_utils.type_check(weighted, [bool], "weighted")
    core_utils.type_check(invert, [bool], "invert")
    core_utils.type_check(return_array, [bool], "return_array")
    core_utils.type_check(prefix, [str], "prefix")
    core_utils.type_check(suffix, [str], "suffix")
    core_utils.type_check(add_uuid, [bool], "add_uuid")
    core_utils.type_check(overwrite, [bool], "overwrite")

    raster_list = core_utils.ensure_list(input_rasters)
    path_list = gdal_utils.create_output_path_list(
        raster_list,
        out_path=out_path,
        overwrite=overwrite,
        prefix=prefix,
        suffix=suffix,
        add_uuid=add_uuid,
    )

    output = []
    for index, input_raster in enumerate(raster_list):
        out_path = path_list[index]

        in_arr = core_raster.raster_to_array(input_raster, filled=True)
        bin_arr = (in_arr != target_value).astype("uint8")
        bin_raster = core_raster.array_to_raster(bin_arr, reference=input_raster)

        in_raster = core_raster._open_raster(bin_raster)
        in_raster_path = bin_raster

        if add_border:
            border_size = 1
            border_raster = add_border_to_raster(
                in_raster,
                border_size=border_size,
                border_value=border_value,
                overwrite=True,
            )

            in_raster = core_raster._open_raster(border_raster)

            gdal.Unlink(in_raster_path)
            in_raster_path = border_raster

        src_band = in_raster.GetRasterBand(1)

        driver_name = "GTiff" if out_path is None else gdal_utils.path_to_driver_raster(out_path)
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
            options=[
                "VALUES='1'",
                f"DISTUNITS={unit}",
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
            core_raster.array_to_raster(dst_arr, reference=input_raster, out_path=out_path)
            output.append(out_path)

        dst_arr = None

    if isinstance(input_rasters, list):
        return output

    return output[0]
