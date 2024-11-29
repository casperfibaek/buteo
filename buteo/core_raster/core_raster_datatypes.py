"""### Functions for changing the datatype of a raster. ###"""

# Standard library
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
from buteo.core_raster.core_raster_read import _open_raster
from buteo.core_raster.core_raster_info import get_metadata_raster



def raster_get_datatype(
    raster: Union[str, gdal.Dataset, List[Union[str, gdal.Dataset]]],
) -> Union[str, List[str]]:
    """
    Gets the data type of one or more rasters.

    Parameters
    ----------
    raster : str or gdal.Dataset or list
        The input raster(s) to get the data type from.

    Returns
    -------
    str or list of str
        The data type(s) of the input raster(s).

    Raises
    ------
    ValueError
        If the input raster cannot be opened or has no bands.
    """
    utils_base._type_check(raster, [str, gdal.Dataset, [str, gdal.Dataset]], "raster")

    input_is_list = isinstance(raster, list)
    input_rasters = utils_io._get_input_paths(raster, "raster")  # type: ignore

    datatypes = []
    for input_raster in input_rasters:
        dataset = _open_raster(input_raster)
        if dataset is None:
            raise ValueError(f"Unable to open input raster: {input_raster}")

        band = dataset.GetRasterBand(1)
        if band is None:
            raise ValueError(f"Raster {input_raster} has no bands.")

        data_type_name = utils_translate._translate_dtype_gdal_to_numpy(band.DataType)
        datatypes.append(data_type_name.name)

    if input_is_list:
        return datatypes

    return datatypes[0]


def _raster_set_datatype(
    raster: Union[str, gdal.Dataset],
    dtype_str: Union[str, np.dtype],
    out_path: Optional[str] = None,
    *,
    overwrite: bool = True,
    creation_options: Optional[List[str]] = None,
) -> str:
    """
    Sets the data type of a raster and saves it to a new file.

    Parameters
    ----------
    raster : str or gdal.Dataset
        Input raster to change data type.
    dtype_str : str or np.dtype
        Target data type for the output raster.
    out_path : str, optional
        Output file path. If not provided, a temporary file will be created.
    overwrite : bool, optional
        If True, overwrites the existing file.
    creation_options : list of str, optional
        GDAL creation options for the output raster.

    Returns
    -------
    str
        File path to the output raster.

    Raises
    ------
    ValueError
        If unable to open input raster or create output raster.
    TypeError
        If input arguments are of incorrect type.
    FileExistsError
        If output file already exists and overwrite is False.
    """
    utils_base._type_check(raster, [str, gdal.Dataset], "raster")
    utils_base._type_check(dtype_str, [str, np.dtype], "dtype_str")
    utils_base._type_check(out_path, [str, None], "out_path")

    if not utils_gdal._check_is_raster(raster):
        raise ValueError(f"Unable to open input raster: {raster}")

    if out_path is None:
        out_path = utils_path._get_temp_filepath("set_datatype.tif")

    ref = _open_raster(raster)
    metadata = get_metadata_raster(ref)

    driver_name = utils_gdal._get_raster_driver_name_from_path(out_path)
    driver = gdal.GetDriverByName(driver_name)
    if driver is None:
        raise ValueError(f"Unable to get driver for path: {out_path}")

    if not overwrite and utils_path._check_file_exists(out_path):
        raise FileExistsError(f"Output file already exists: {out_path}")

    utils_io._check_overwrite_policy([out_path], overwrite)
    utils_io._delete_if_required(out_path, overwrite)

    if isinstance(dtype_str, np.dtype):
        dtype_str = dtype_str.name.lower()
    elif isinstance(dtype_str, str):
        dtype_str = dtype_str.lower()
    else:
        raise TypeError("dtype_str must be a string or numpy dtype.")

    gdal_dtype = utils_translate._translate_dtype_numpy_to_gdal(dtype_str)
    creation_options = utils_gdal._get_default_creation_options(creation_options)

    copy = driver.Create(
        out_path,
        metadata["width"],
        metadata["height"],
        metadata["bands"],
        gdal_dtype,
        creation_options,
    )
    if copy is None:
        raise ValueError(f"Unable to create output raster: {out_path}")

    copy.SetProjection(metadata["projection_wkt"])
    copy.SetGeoTransform(metadata["geotransform"])

    for band_idx in range(metadata["bands"]):
        input_band = ref.GetRasterBand(band_idx + 1)
        output_band = copy.GetRasterBand(band_idx + 1)

        data = input_band.ReadAsArray().astype(dtype_str)
        output_band.WriteArray(data)

        input_nodata = input_band.GetNoDataValue()
        if input_nodata is not None:
            if utils_translate._check_is_value_within_dtype_range(input_nodata, dtype_str):
                output_band.SetNoDataValue(input_nodata)
            else:
                warn(
                    "Input NoData value is outside the range of the output datatype. NoData value will not be set.",
                    UserWarning,
                )
                output_band.DeleteNoDataValue()

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
    """Converts the datatype of a raster.

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

    input_rasters = utils_io._get_input_paths(raster, "raster") # type: ignore
    out_paths = utils_io._get_output_paths(
        input_rasters, # type: ignore
        out_path,
        add_uuid=add_uuid,
        add_timestamp=add_timestamp,
        prefix=prefix,
        suffix=suffix,
    )

    creation_options = utils_gdal._get_default_creation_options(creation_options)

    utils_io._check_overwrite_policy(out_paths, overwrite)
    utils_io._delete_if_required_list(out_paths, overwrite)

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
