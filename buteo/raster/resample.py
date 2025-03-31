"""### Resample rasters. ###

Module to resample rasters to a target resolution.
Can uses references from vector or other raster datasets.
"""

# Standard library
from typing import Union, Optional, List

# External
import numpy as np
from osgeo import gdal

# Internal
from buteo.utils import (
    utils_io,
    utils_base,
    utils_gdal,
    utils_path,
    utils_translate,
)
from buteo.core_raster.core_raster_read import _open_raster
from buteo.core_raster.core_raster_info import get_metadata_raster
from buteo.core_raster.core_raster_write import raster_create_from_array
from buteo.core_raster.core_raster_array import raster_to_array



def _raster_resample(
    raster: Union[str, gdal.Dataset],
    target_size: Union[List[Union[int, float]], int, float, gdal.Dataset, str],
    out_path: Optional[str] = None,
    *,
    target_in_pixels: bool = False,
    resample_alg: str = "nearest",
    overwrite: bool = True,
    creation_options: Optional[List[str]] = None,
    dtype: Optional[str] = None,
    dst_nodata: Union[float, int, str] = "infer",
    verbose: int = 0,
    ram: float = 0.8,
    ram_max: Optional[int] = None,
    ram_min: Optional[int] = 100,
    add_uuid: bool = False,
    add_timestamp: bool = True,
    prefix: str = "",
    suffix: str = "resampled",
) -> Union[str, gdal.Dataset]:
    """Internal."""
    assert isinstance(raster, (gdal.Dataset, str)), f"The input raster must be in the form of a str or a gdal.Dataset: {raster}"

    if out_path is None:
        out_path = utils_path._get_temp_filepath(
            name="resampled_raster.tif",
            prefix=prefix,
            suffix=suffix,
            add_uuid=add_uuid,
            add_timestamp=add_timestamp
        )
    else:
        if not utils_path._check_is_valid_output_filepath(out_path):
            raise ValueError(f"Invalid output path: {out_path}")

    ref = _open_raster(raster)
    metadata = get_metadata_raster(ref)

    if isinstance(target_size, (gdal.Dataset, str)):
        x_res, y_res = utils_gdal._get_raster_size(target_size)
        x_pixels = None
        y_pixels = None
    elif target_in_pixels:
        x_res = None
        y_res = None

        if isinstance(target_size, (list, tuple)):
            assert len(target_size) == 2, f"Invalid target_size. {target_size}"
            x_pixels = target_size[0]
            y_pixels = target_size[1]
        elif isinstance(target_size, (int, float)):
            x_pixels = target_size
            y_pixels = target_size
        else:
            raise RuntimeError(f"Invalid target_size. {target_size}")
    else:
        x_pixels = None
        y_pixels = None

        if isinstance(target_size, (list, tuple)):
            assert len(target_size) == 2, f"Invalid target_size. {target_size}"
            x_res = target_size[0]
            y_res = target_size[1]
        elif isinstance(target_size, (int, float)):
            x_res = target_size
            y_res = target_size
        else:
            raise RuntimeError(f"Invalid target_size. {target_size}")

    out_format = utils_gdal._get_raster_driver_name_from_path(out_path)

    src_nodata = metadata["nodata_value"]
    out_nodata = None
    if dst_nodata == "infer":
        dst_nodata = src_nodata
    else:
        assert isinstance(dst_nodata, (int, float, type(None))), "dst_nodata must be an int, float, 'infer', or 'None'"
        out_nodata = dst_nodata

    if dtype is None:
        dtype = metadata["dtype"]

    if out_nodata is not None and not utils_translate._check_is_value_within_dtype_range(out_nodata, dtype): # type: ignore
        raise ValueError(f"Invalid nodata value for datatype. value: {out_nodata}, dtype: {dtype}")

    utils_io._delete_if_required(out_path, overwrite)

    if verbose == 0:
        gdal.PushErrorHandler("CPLQuietErrorHandler")

    if x_pixels is None or y_pixels is None:
        options = gdal.WarpOptions(
            format=out_format,
            xRes=x_res,
            yRes=y_res,
            outputType=utils_translate._translate_dtype_numpy_to_gdal(dtype), # type: ignore
            resampleAlg=utils_translate._translate_resample_method(resample_alg),
            creationOptions=utils_gdal._get_default_creation_options(creation_options),
            srcNodata=metadata["nodata_value"],
            dstNodata=out_nodata,
            multithread=True,
            warpMemoryLimit=utils_gdal._get_dynamic_memory_limit(ram, min_mb=ram_min if ram_min is not None else 100, max_mb=ram_max),
        )
    else:
        options = gdal.WarpOptions(
            format=out_format,
            width=int(x_pixels),
            height=int(y_pixels),
            outputType=utils_translate._translate_dtype_numpy_to_gdal(dtype), # type: ignore
            resampleAlg=utils_translate._translate_resample_method(resample_alg),
            creationOptions=utils_gdal._get_default_creation_options(creation_options),
            srcNodata=metadata["nodata_value"],
            dstNodata=out_nodata,
            multithread=True,
            warpMemoryLimit=utils_gdal._get_dynamic_memory_limit(ram, min_mb=ram_min if ram_min is not None else 100, max_mb=ram_max),
        )

    resampled = gdal.Warp(out_path, ref, options=options)

    if verbose == 0:
        gdal.PopErrorHandler()

    if resampled is None:
        raise RuntimeError(f"Error while resampling raster: {out_path}")

    return out_path


def raster_resample(
    raster: Union[str, gdal.Dataset, List[Union[str, gdal.Dataset]]],
    target_size: Union[List[Union[int, float]], int, float, gdal.Dataset, str],
    out_path: Optional[Union[str, List[str]]] = None,
    *,
    target_in_pixels: bool = False,
    resample_alg: str = "nearest",
    creation_options: Optional[List[str]] = None,
    dtype: Optional[str] = None,
    dst_nodata: Union[float, int, str] = "infer",
    prefix: str = "",
    suffix: str = "",
    overwrite: bool = True,
    add_uuid: bool = False,
    add_timestamp: bool = False,
    verbose: int = 0,
    ram: float = 0.8,
    ram_max: Optional[int] = None,
    ram_min: Optional[int] = 100,
):
    """Resampled raster(s) given a target size.
    Beware, if your input is in latitude and longitude, you'll need to specify the target_size in degrees as well.

    Parameters
    ----------
    raster : str/gdal.Dataset/list
        The input raster(s) to resample.

    target_size : str/int/ogr.DataSource/gdal.Dataset
        The desired resolution for the resampled raster(s), in the same unit as the raster projection.
        x_res, y_res - or x_pixels, y_pixels if target_in_pixels is True.

    target_in_pixels : bool, optional
        If True, interprets target_size as the number of pixels, default: False

    out_path : str, optional
        The output path for the resampled raster(s). If not provided, the output path is inferred from the input raster(s), default: None

    resample_alg : str, optional
        The resampling algorithm to use, default: "nearest"

    creation_options : list, optional
        A list of creation options for the output raster(s), default: None

    dtype : str, optional
        The output data type, default: None

    dst_nodata : str/int/float, optional
        The nodata value for the output raster(s), default: "infer"

    prefix : str, optional
        A prefix to add to the output path, default: ""

    suffix : str, optional
        A suffix to add to the output path, default: ""

    add_uuid : bool, optional
        If True, adds a uuid to the output path, default: False

    add_timestamp : bool, optional
        If True, adds a timestamp to the output path, default: False

    verbose : int, optional
        The verbosity level, default: 0

    ram : float, optional
        The amount of RAM to use for the resampling, default: 0.8 (80% of the total RAM)

    ram_max : int, optional
        The maximum amount of RAM to use for the resampling, default: None

    ram_min : int, optional
        The minimum amount of RAM to use for the resampling, default: 100

    overwrite : bool, optional
        If True, overwrites the output raster(s) if it/they already exist, default: True

    Returns
    -------
    str/List[str]
        The output path(s) of the resampled raster(s).
    """
    utils_base._type_check(raster, [str, gdal.Dataset, [str, gdal.Dataset]], "raster")
    utils_base._type_check(target_size, [tuple, [int, float], int, float, str, gdal.Dataset], "target_size")
    utils_base._type_check(target_in_pixels, [bool], "target_in_pixels")
    utils_base._type_check(out_path, [str, [str], None], "out_path")
    utils_base._type_check(resample_alg, [str], "resample_alg")
    utils_base._type_check(overwrite, [bool], "overwrite")
    utils_base._type_check(creation_options, [[str], None], "creation_options")
    utils_base._type_check(dst_nodata, [str, int, float, None], "dst_nodata")
    utils_base._type_check(dtype, [str, None, np.dtype, type(np.int8)], "dtype")
    utils_base._type_check(prefix, [str], "prefix")
    utils_base._type_check(suffix, [str], "suffix")
    utils_base._type_check(add_uuid, [bool], "add_uuid")
    utils_base._type_check(add_timestamp, [bool], "add_timestamp")
    utils_base._type_check(verbose, [int], "verbose")
    utils_base._type_check(ram, [int, float], "ram")
    utils_base._type_check(ram_max, [int, float, None], "ram_max")
    utils_base._type_check(ram_min, [int, float, None], "ram_min")

    input_is_list = isinstance(raster, list)

    in_paths = utils_io._get_input_paths(raster, "raster") # type: ignore
    out_paths = utils_io._get_output_paths(
        in_paths, # type: ignore
        out_path,
        add_uuid=add_uuid,
        add_timestamp=add_timestamp,
        prefix=prefix,
        suffix=suffix,
    )

    utils_io._check_overwrite_policy(out_paths, overwrite)
    utils_io._delete_if_required_list(out_paths, overwrite)

    resampled_rasters = []
    for idx, in_raster in enumerate(in_paths):
        resampled_rasters.append(
            _raster_resample(
                in_raster,
                target_size,
                target_in_pixels=target_in_pixels,
                out_path=out_paths[idx],
                resample_alg=resample_alg,
                overwrite=overwrite,
                creation_options=creation_options,
                dtype=dtype,
                dst_nodata=dst_nodata,
                ram=ram,
                ram_max=ram_max,
                ram_min=ram_min,
                add_uuid=add_uuid,
                add_timestamp=add_timestamp,
                prefix=prefix,
                suffix=suffix,
                verbose=verbose,
            )
        )

    if input_is_list:
        return resampled_rasters

    return resampled_rasters[0]


def resample_array(
    arr,
    target_shape_pixels,
    resample_alg="bilinear",
):
    """Resample a numpy array using the GDAL algorithms."""
    utils_base._type_check(arr, [np.ndarray, np.ma.MaskedArray], "arr")
    utils_base._type_check(target_shape_pixels, [tuple, [int, float]], "target_shape_pixels")
    utils_base._type_check(resample_alg, [str], "resample_alg")

    assert len(arr.shape) in [2, 3], f"Invalid array shape: {arr.shape}"
    assert len(target_shape_pixels) in [2, 3], f"Invalid target_shape_pixels: {target_shape_pixels}"

    if len(target_shape_pixels) == 3:
        target_shape_pixels = (target_shape_pixels[1], target_shape_pixels[2])
    elif len(target_shape_pixels) == 2:
        target_shape_pixels = (target_shape_pixels[0], target_shape_pixels[1])

    arr_as_raster = raster_create_from_array(arr)
    resampled = _raster_resample(
        arr_as_raster,
        list(target_shape_pixels),
        target_in_pixels=True,
        resample_alg=resample_alg,
    )
    out_arr = raster_to_array(resampled)

    utils_gdal.delete_dataset_if_in_memory(arr_as_raster)
    utils_gdal.delete_dataset_if_in_memory(resampled)

    return out_arr
