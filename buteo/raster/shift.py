"""
### Shift rasters. ###

Module to shift the location of rasters in geographic coordinates.
"""

# Standard library
import sys; sys.path.append("../../")

# External
from osgeo import gdal

# Internal
from buteo.utils import core_utils, gdal_utils
from buteo.raster import core_raster



def _shift_raster(
    raster,
    shift_list,
    out_path=None,
    *,
    overwrite=True,
    creation_options=None,
):
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
    raster,
    shift_list,
    out_path=None,
    *,
    overwrite=True,
    prefix="",
    suffix="",
    add_uuid=False,
    creation_options=None,
):
    """
    Shifts a raster in a given direction.

    ## Args:
    `raster` (_str_/_list_/_gdal.Dataset_): The raster(s) to be shifted. </br>
    `shift_list` (_list_/_tuple_): The shift in x and y direction. </br>

    ## Kwargs:
    `out_path` (_str_/_list_/_None_): The path to the output raster. If None, the raster is
    created in memory. (Default: **None**)</br>
    `overwrite` (_bool_): If True, the output raster will be overwritten if it already exists. (Default: **True**) </br>
    `prefix` (_str_): The prefix to be added to the output raster name. (Default: **""**) </br>
    `suffix` (_str_): The suffix to be added to the output raster name. (Default: **""**) </br>
    `add_uuid` (_bool_): If True, a unique identifier will be added to the output raster name. (Default: **False**) </br>
    `creation_options` (_list_/_None_): The creation options to be used when creating the output. (Default: **None**) </br>

    ## Returns:
    (_str_/_list_): The path(s) to the shifted raster(s).
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
