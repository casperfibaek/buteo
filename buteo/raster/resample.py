"""
### Resample rasters. ###

Module to resample rasters to a target resolution.
Can uses references from vector or other raster datasets.
"""

# Standard library
import sys; sys.path.append("../../")
from typing import Union, Optional, List

# External
from osgeo import gdal
import numpy as np

# Internal
from buteo.utils import (
    utils_base,
    utils_gdal,
    utils_path,
    utils_translate,
)
from buteo.raster import core_raster



def _raster_resample(
    raster: Union[str, gdal.Dataset, List],
    target_size: Union[int, float],
    out_path: Optional[str] = None,
    *,
    target_in_pixels: bool = False,
    resample_alg: str = "nearest",
    overwrite: bool = True,
    creation_options: Optional[List[str]] = None,
    dtype: Optional[str] = None,
    dst_nodata: Union[float, int, str] = "infer",
    prefix: str = "",
    suffix: str = "_resampled",
    add_uuid: bool = False,
) -> Union[str, gdal.Dataset]:
    """ Internal. """
    assert isinstance(raster, (gdal.Dataset, str)), f"The input raster must be in the form of a str or a gdal.Dataset: {raster}"

    out_path = utils_gdal._parse_output_data(
        raster,
        output_data=out_path,
        overwrite=overwrite,
        prefix=prefix,
        suffix=suffix,
        add_uuid=add_uuid,
    )

    ref = core_raster._raster_open(raster)
    metadata = core_raster._raster_to_metadata(ref)

    x_res, y_res, x_pixels, y_pixels = utils_gdal._get_raster_size(
        target_size, target_in_pixels=target_in_pixels
    )

    out_format = utils_gdal._get_raster_driver_name_from_path(out_path)

    src_nodata = metadata["nodata_value"]
    out_nodata = None
    if dst_nodata == "infer":
        dst_nodata = src_nodata
    else:
        assert isinstance(dst_nodata, (int, float, type(None))), "dst_nodata must be an int, float, 'infer', or 'None'"
        out_nodata = dst_nodata

    if dtype is None:
        dtype = metadata["datatype"]

    utils_path._delete_if_required(out_path, overwrite)

    resampled = gdal.Warp(
        out_path,
        ref,
        width=x_pixels,
        height=y_pixels,
        xRes=x_res,
        yRes=y_res,
        format=out_format,
        outputType=utils_translate._translate_str_to_gdal_dtype(dtype),
        resampleAlg=utils_translate._translate_resample_method(resample_alg),
        creationOptions=utils_gdal._get_default_creation_options(creation_options),
        srcNodata=metadata["nodata_value"],
        dstNodata=out_nodata,
        multithread=True,
    )

    if resampled is None:
        raise RuntimeError(f"Error while resampling raster: {out_path}")

    return out_path


def raster_resample(
    raster,
    target_size,
    out_path=None,
    *,
    target_in_pixels=False,
    resample_alg="nearest",
    creation_options=None,
    dtype=None,
    dst_nodata="infer",
    prefix="",
    suffix="",
    add_uuid=False,
    overwrite=True,
):
    """
    Resampled raster(s) given a target size.
    Beware, if your input is in latitude and longitude, you'll need to specify the target_size in degrees as well.

    Parameters
    ----------
    raster : str/gdal.Dataset/list
        The input raster(s) to resample.
    
    target_size : str/int/ogr.DataSource/gdal.Dataset
        The desired resolution for the resampled raster(s), in the same unit as the raster projection.

    target_in_pixels : bool, optional
        If True, interprets target_size as the number of pixels, by default False

    out_path : str, optional
        The output path for the resampled raster(s). If not provided, the output path is inferred from the input raster(s), by default None

    resample_alg : str, optional
        The resampling algorithm to use, by default "nearest"

    creation_options : list, optional
        A list of creation options for the output raster(s), by default None

    dtype : str, optional
        The output data type, by default None

    dst_nodata : str/int/float, optional
        The nodata value for the output raster(s), by default "infer"

    prefix : str, optional
        A prefix to add to the output path, by default ""

    suffix : str, optional
        A suffix to add to the output path, by default ""

    add_uuid : bool, optional
        If True, adds a uuid to the output path, by default False

    overwrite : bool, optional
        If True, overwrites the output raster(s) if it/they already exist, by default True

    Returns
    -------
    str/list[str]
        The output path(s) of the resampled raster(s).
    """
    utils_base.type_check(raster, [str, gdal.Dataset, [str, gdal.Dataset]], "raster")
    utils_base.type_check(target_size, [tuple, [int, float], int, float, str, gdal.Dataset], "target_size")
    utils_base.type_check(target_in_pixels, [bool], "target_in_pixels")
    utils_base.type_check(out_path, [str, [str], None], "out_path")
    utils_base.type_check(resample_alg, [str], "resample_alg")
    utils_base.type_check(overwrite, [bool], "overwrite")
    utils_base.type_check(creation_options, [[str], None], "creation_options")
    utils_base.type_check(dst_nodata, [str, int, float, None], "dst_nodata")
    utils_base.type_check(dtype, [str, None], "dtype")
    utils_base.type_check(prefix, [str], "prefix")
    utils_base.type_check(suffix, [str], "postfix")

    if utils_path._check_is_path_glob(raster):
        raster = utils_path._get_paths_from_glob(raster)

    raster_list = utils_base._get_variable_as_list(raster)
    assert utils_gdal._check_is_raster_list(raster_list), f"Invalid raster in raster list: {raster_list}"

    path_list = utils_gdal._parse_output_data(
        raster_list,
        output_data=out_path,
        overwrite=overwrite,
        prefix=prefix,
        suffix=suffix,
        add_uuid=add_uuid,
    )

    resampled_rasters = []
    for index, in_raster in enumerate(raster_list):
        resampled_rasters.append(
            _raster_resample(
                in_raster,
                target_size,
                target_in_pixels=target_in_pixels,
                out_path=path_list[index],
                resample_alg=resample_alg,
                overwrite=overwrite,
                creation_options=creation_options,
                dtype=dtype,
                dst_nodata=dst_nodata,
                prefix=prefix,
                suffix=suffix,
            )
        )

    if isinstance(raster, list):
        return resampled_rasters

    return resampled_rasters[0]


def resample_array(arr, target_shape_pixels, resample_alg="nearest"):
    """ Resample a numpy array using the GDAL algorithms. """
    utils_base.type_check(arr, [np.ndarray, np.ma.MaskedArray], "arr")
    utils_base.type_check(target_shape_pixels, [tuple, [int, float]], "target_shape_pixels")
    utils_base.type_check(resample_alg, [str], "resample_alg")

    if len(target_shape_pixels) > 2:
        target_shape_pixels = target_shape_pixels[:2]

    arr_as_raster = core_raster.raster_create_from_array(arr)
    resampled = _raster_resample(arr_as_raster, target_shape_pixels, target_in_pixels=True, resample_alg=resample_alg)
    out_arr = core_raster.raster_to_array(resampled)

    utils_gdal.delete_dataset_if_in_memory(arr_as_raster)
    utils_gdal.delete_dataset_if_in_memory(resampled)

    return out_arr
