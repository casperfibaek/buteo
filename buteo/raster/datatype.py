"""
### Functions for changing the datatype of a raster. ###
"""

# Standard library
import sys; sys.path.append("../../")
import os
from typing import Union, List, Optional
from warnings import warn

# External
from osgeo import gdal

# Internal
from buteo.utils import (
    utils_gdal,
    utils_base,
    utils_path,
    utils_translate,
)
from buteo.raster.core_raster import _raster_open, _get_basic_metadata_raster


def _raster_set_datatype(
    raster: Union[str, gdal.Dataset],
    dtype_str: str,
    out_path: Optional[str] = None,
    *,
    overwrite: bool = True,
    creation_options: Optional[List[str]] = None,
) -> str:
    """ **INTERNAL**. """
    assert isinstance(raster, (str, gdal.Dataset)), "raster must be a string or a GDAL.Dataset."
    assert isinstance(dtype_str, str), "dtype_str must be a string."
    assert len(dtype_str) > 0, "dtype_str must be a non-empty string."
    assert out_path is None or isinstance(out_path, str), "out_path must be a string."

    if not utils_gdal._check_is_raster(raster):
        raise ValueError(f"Unable to open input raster: {raster}")

    ref = _raster_open(raster)
    metadata = _get_basic_metadata_raster(ref)

    path = ""
    if out_path is None:
        path = utils_path._get_augmented_path_list("set_datatype.tif", add_uuid=True, folder="/vsimem/")

    elif utils_path._check_dir_exists(out_path):
        path = os.path.join(out_path, os.path.basename(utils_gdal._get_path_from_dataset(ref)))

    elif utils_path._check_dir_exists(utils_path._get_dir_from_path(out_path)):
        path = out_path

    elif utils_path._check_is_valid_mem_filepath(out_path):
        path = out_path

    else:
        raise ValueError(f"Unable to find output folder: {out_path}")

    driver_name = utils_gdal._get_raster_driver_from_path(path)
    driver = gdal.GetDriverByName(driver_name)

    if driver is None:
        raise ValueError(f"Unable to get driver for raster: {raster}")

    utils_path._delete_if_required(path, overwrite)

    if isinstance(dtype_str, str):
        dtype_str = dtype_str.lower()

    copy = driver.Create(
        path,
        metadata["width"],
        metadata["height"],
        metadata["bands"],
        utils_translate._translate_str_to_gdal_dtype(dtype_str),
        utils_gdal._get_default_creation_options(creation_options),
    )

    if copy is None:
        raise ValueError(f"Unable to create output raster: {path}")

    copy.SetProjection(metadata["projection_wkt"])
    copy.SetGeoTransform(metadata["transform"])

    for band_idx in range(metadata["bands"]):
        input_band = ref.GetRasterBand(band_idx + 1)
        output_band = copy.GetRasterBand(band_idx + 1)

        # Read the input band data and write it to the output band
        data = input_band.ReadRaster(0, 0, input_band.XSize, input_band.YSize)
        output_band.WriteRaster(0, 0, input_band.XSize, input_band.YSize, data)

        # Set the NoData value for the output band if it exists in the input band
        if input_band.GetNoDataValue() is not None:
            input_nodata = input_band.GetNoDataValue()
            if utils_translate._check_is_value_within_dtype_range(input_nodata, dtype_str):
                output_band.SetNoDataValue(input_nodata)
            else:
                warn("Input NoData value is outside the range of the output datatype. NoData value will not be set.", UserWarning)
                output_band.SetNoDataValue(None)

        # Set the color interpretation for the output band
        output_band.SetColorInterpretation(input_band.GetColorInterpretation())

    copy.FlushCache()

    ref = None
    copy = None

    return path


def raster_set_datatype(
    raster: Union[str, gdal.Dataset, List[Union[str, gdal.Dataset]]],
    dtype: str,
    out_path: Optional[Union[str, List[str]]] = None,
    *,
    overwrite: bool = True,
    allow_lists: bool = True,
    creation_options: Optional[List[str]] = None,
) -> Union[str, List[str]]:
    """
    Converts the datatype of a raster.

    Parameters
    ----------
    raster : str or gdal.Dataset or list
        The input raster(s) for which the datatype will be changed.

    dtype : str
        The target datatype for the output raster(s).

    out_path : path or list, optional
        The output location for the processed raster(s). Default: None.

    overwrite : bool, optional
        Determines whether to overwrite existing files with the same name. Default: True.

    allow_lists : bool, optional
        Allows processing multiple rasters as a list. If set to False, only single rasters are accepted.
        Default: True.

    creation_options : list, optional
        A list of GDAL creation options for the output raster(s). Default is
        ["TILED=YES", "NUM_THREADS=ALL_CPUS", "BIGTIFF=YES", "COMPRESS=LZW"].

    Returns
    -------
    str or list
        The file path(s) of the newly created raster(s) with the specified datatype.
    """
    utils_base.type_check(raster, [str, gdal.Dataset, [str, gdal.Dataset]], "raster")
    utils_base.type_check(dtype, [str], "dtype")
    utils_base.type_check(out_path, [list, str, None], "out_path")
    utils_base.type_check(overwrite, [bool], "overwrite")
    utils_base.type_check(allow_lists, [bool], "allow_lists")
    utils_base.type_check(creation_options, [list, None], "creation_options")

    if not allow_lists:
        if isinstance(raster, list):
            raise ValueError("allow_lists is False, but the input raster is a list.")

        return _raster_set_datatype(
            raster,
            dtype,
            out_path=out_path,
            overwrite=overwrite,
            creation_options=creation_options,
        )

    add_uuid = out_path is None

    raster_list = utils_base._get_variable_as_list(raster)
    path_list = utils_gdal._parse_output_data(
        raster_list,
        output_data=out_path,
        add_uuid=add_uuid,
        overwrite=overwrite,
    )

    output = []
    for index, in_raster in enumerate(raster_list):
        path = _raster_set_datatype(
            in_raster,
            dtype,
            out_path=path_list[index],
            overwrite=overwrite,
            creation_options=utils_gdal._get_default_creation_options(creation_options),
        )

        output.append(path)

    if isinstance(raster, list):
        return output

    return output[0]
