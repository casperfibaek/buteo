"""
### Functions for changing the datatype of a raster. ###
"""

# Standard library
import sys; sys.path.append("../../")
from typing import Union, List, Optional
from warnings import warn

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
from buteo.raster.core_raster import _raster_open, _get_basic_metadata_raster


def _raster_set_datatype(
    raster: Union[str, gdal.Dataset],
    dtype_str: str,
    out_path: Optional[str] = None,
    *,
    overwrite: bool = True,
    creation_options: Optional[List[str]] = None,
) -> str:
    """
    Internal function.

    For internal functions, only basic testing should be done.
    The input should be checked in the public function calling the internal function.
    an input and output raster should be provided.
    """
    assert isinstance(raster, (str, gdal.Dataset)), "raster must be a string or a GDAL.Dataset."
    assert isinstance(dtype_str, (str, int, np.dtype)), "dtype_str must be a string."
    assert isinstance(out_path, (str, None)), "out_path must be a string or None."

    if out_path is None:
        out_path = utils_path._get_temp_filepath("set_datatype.tif")

    ref = _raster_open(raster)
    metadata = _get_basic_metadata_raster(ref)

    driver_name = utils_gdal._get_raster_driver_name_from_path(out_path)
    driver = gdal.GetDriverByName(driver_name)

    if driver is None:
        raise ValueError(f"Unable to get driver for raster: {raster}")

    utils_path._delete_if_required(out_path, overwrite)

    if isinstance(dtype_str, str):
        dtype_str = dtype_str.lower()

    copy = driver.Create(
        out_path,
        metadata["width"],
        metadata["height"],
        metadata["bands"],
        utils_translate._translate_dtype_numpy_to_gdal(dtype_str),
        utils_gdal._get_default_creation_options(creation_options),
    )

    if copy is None:
        raise ValueError(f"Unable to create output raster: {out_path}")

    copy.SetProjection(metadata["projection_wkt"])
    copy.SetGeoTransform(metadata["geotransform"])

    for band_idx in range(metadata["bands"]):
        input_band = ref.GetRasterBand(band_idx + 1)
        output_band = copy.GetRasterBand(band_idx + 1)

        # Read the input band data and write it to the output band
        data = input_band.ReadAsArray(0, 0, input_band.XSize, input_band.YSize).astype(dtype_str)
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

    return out_path


def raster_set_datatype(
    raster: Union[str, gdal.Dataset, List[Union[str, gdal.Dataset]]],
    dtype: str,
    out_path: Optional[Union[str, List[str]]] = None,
    *,
    creation_options: Optional[List[str]] = None,
    add_uuid: bool = False,
    add_timestamp: bool = False,
    prefix: str = "",
    suffix: str = "",
    overwrite: bool = True,
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

    creation_options : list, optional
        A list of GDAL creation options for the output raster(s). Default is
        ["TILED=YES", "NUM_THREADS=ALL_CPUS", "BIGTIFF=YES", "COMPRESS=LZW"].

    add_uuid : bool, optional
        Determines whether to add a UUID to the output path. Default: False.

    add_timestamp : bool, optional
        Determines whether to add a timestamp to the output path. Default: False.

    prefix : str, optional
        A prefix to add to the output path. Default: "".

    suffix : str, optional
        A suffix to add to the output path. Default: "".

    overwrite : bool, optional
        Determines whether to overwrite existing files with the same name. Default: True.

    Returns
    -------
    str or list
        The file path(s) of the newly created raster(s) with the specified datatype.
    """
    utils_base._type_check(raster, [str, gdal.Dataset, [str, gdal.Dataset]], "raster")
    utils_base._type_check(dtype, [str, np.dtype, int, type(np.int8)], "dtype")
    utils_base._type_check(out_path, [list, str, None], "out_path")
    utils_base._type_check(creation_options, [list, None], "creation_options")
    utils_base._type_check(add_uuid, [bool], "add_uuid")
    utils_base._type_check(add_timestamp, [bool], "add_timestamp")
    utils_base._type_check(prefix, [str], "prefix")
    utils_base._type_check(suffix, [str], "suffix")
    utils_base._type_check(overwrite, [bool], "overwrite")

    input_is_list = isinstance(raster, list)

    input_rasters = utils_io._get_input_paths(raster, "raster")
    out_paths = utils_io._get_output_paths(
        input_rasters,
        out_path,
        add_uuid=add_uuid,
        add_timestamp=add_timestamp,
        prefix=prefix,
        suffix=suffix,
    )

    creation_options = utils_gdal._get_default_creation_options(creation_options)

    utils_path._delete_if_required_list(out_paths, overwrite)

    output = []
    for idx, in_raster in enumerate(input_rasters):
        path = _raster_set_datatype(
            in_raster,
            dtype,
            out_path=out_paths[idx],
            overwrite=overwrite,
            creation_options=creation_options,
        )

        output.append(path)

    if input_is_list:
        return output

    return output[0]


def raster_get_datatype(
    raster: Union[str, gdal.Dataset, List[Union[str, gdal.Dataset]]],
) -> Union[str, List[str]]:
    """
    Gets the datatype of a raster.

    Parameters
    ----------
    raster : str or gdal.Dataset or list
        The input raster(s) for which the datatype will be changed.

    Returns
    -------
    str or list
        The datatype of the input raster(s).
    """
    input_is_list = isinstance(raster, list)
    input_rasters = utils_io._get_input_paths(raster, "raster")

    datatypes = []
    for input_raster in input_rasters:
        if not utils_gdal._check_is_raster(input_raster):
            raise ValueError(f"Unable to open input raster: {input_raster}")

        metadata = _get_basic_metadata_raster(input_raster)
        datatypes.append(metadata["dtype_name"])

    if input_is_list:
        return datatypes

    return datatypes[0]
