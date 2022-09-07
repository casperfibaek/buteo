"""
### Resample rasters. ###

Module to resample rasters to a target resolution.
Can uses references from vector or other raster datasets.
"""

# Standard library
import sys; sys.path.append("../../")

# External
from osgeo import gdal
import numpy as np

# Internal
from buteo.utils import core_utils, gdal_utils, gdal_enums
from buteo.raster import core_raster



def _resample_raster(
    raster,
    target_size,
    out_path=None,
    *,
    target_in_pixels=False,
    resample_alg="nearest",
    overwrite=True,
    creation_options=None,
    dtype=None,
    dst_nodata="infer",
    prefix="",
    suffix="_resampled",
    add_uuid=False,
):
    """ Internal. """
    assert isinstance(raster, (gdal.Dataset, str)), f"The input raster must be in the form of a str or a gdal.Dataset: {raster}"

    out_path = gdal_utils.create_output_path(
        raster,
        out_path=out_path,
        overwrite=overwrite,
        prefix=prefix,
        suffix=suffix,
        add_uuid=add_uuid,
    )

    ref = core_raster._open_raster(raster)
    metadata = core_raster._raster_to_metadata(ref)

    x_res, y_res, x_pixels, y_pixels = gdal_utils.parse_raster_size(
        target_size, target_in_pixels=target_in_pixels
    )

    out_format = gdal_utils.path_to_driver_raster(out_path)

    src_nodata = metadata["nodata_value"]
    out_nodata = None
    if dst_nodata == "infer":
        dst_nodata = src_nodata
    else:
        assert isinstance(dst_nodata, (int, float, type(None))), "dst_nodata must be an int, float, 'infer', or 'None'"
        out_nodata = dst_nodata

    if dtype is None:
        dtype = metadata["datatype"]

    core_utils.remove_if_required(out_path, overwrite)

    resampled = gdal.Warp(
        out_path,
        ref,
        width=x_pixels,
        height=y_pixels,
        xRes=x_res,
        yRes=y_res,
        format=out_format,
        outputType=gdal_enums.translate_str_to_gdal_dtype(dtype),
        resampleAlg=gdal_enums.translate_resample_method(resample_alg),
        creationOptions=gdal_utils.default_creation_options(creation_options),
        srcNodata=metadata["nodata_value"],
        dstNodata=out_nodata,
        multithread=True,
    )

    if resampled is None:
        raise Exception(f"Error while resampling raster: {out_path}")

    return out_path


def resample_raster(
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
    Reprojects raster(s) given a target projection. </br>
    **Beware** if your input is in latitude and longitude, you'll need to specify the target_size in degrees as well.

    ## Args:
    `raster` (_str_/_list_/_gdal.Dataset): The raster(s) to reproject. </br>
    `target_size` (_str_/_int_/_ogr.DataSource_/_gdal.Dataset_): The target resolution of the
    raster. In the same unit as the projection of the raster.
    It's better to reproject to a projected coordinate system for resampling.
    If a raster is the target_size the function will read the pixel size from
    that raster. </br>

    ## Kwargs:
    `target_in_pixels` (_bool_): If True, the target_size will be interpreted as the number of pixels. (Default: **False**) </br>
    `out_path` (_str_/_None_): The output path. If not provided, the output path is inferred from the input. (Default: **None**) </br>
    `resample_alg` (_str_): The resampling algorithm. (Default: **nearest**) </br>
    `copy_if_same` (_bool_): If the input and output projections are the same, copy the input raster to the output path. (Default: **True**) </br>
    `overwrite` (_bool_): If the output path already exists, overwrite it. (Default: **True**) </br>
    `creation_options` (_list_/_None_): A list of creation options for the output raster. (Default: **None**) </br>
    `dst_nodata` (_str_/_int_/_float_): The nodata value for the output raster. (Default: **infer**) </br>
    `prefix` (_str_): The prefix to add to the output path. (Default: **""**) </br>
    `suffix` (_str_): The suffix to add to the output path. (Default: **"_reprojected"**) </br>
    `add_uuid` (_bool_): If True, add a UUID to the output path. (Default: **False**) </br>

    ## Returns:
    (_str_/_list_): The output path(s) of the resampled raster(s).
    """
    core_utils.type_check(raster, [str, gdal.Dataset, [str, gdal.Dataset]], "raster")
    core_utils.type_check(target_size, [tuple, [int, float], int, float, str, gdal.Dataset], "target_size")
    core_utils.type_check(target_in_pixels, [bool], "target_in_pixels")
    core_utils.type_check(out_path, [str, [str], None], "out_path")
    core_utils.type_check(resample_alg, [str], "resample_alg")
    core_utils.type_check(overwrite, [bool], "overwrite")
    core_utils.type_check(creation_options, [[str], None], "creation_options")
    core_utils.type_check(dst_nodata, [str, int, float, None], "dst_nodata")
    core_utils.type_check(dtype, [str, None], "dtype")
    core_utils.type_check(prefix, [str], "prefix")
    core_utils.type_check(suffix, [str], "postfix")

    if core_utils.is_str_a_glob(raster):
        raster = core_utils.parse_glob_path(raster)

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

    resampled_rasters = []
    for index, in_raster in enumerate(raster_list):
        resampled_rasters.append(
            _resample_raster(
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
    core_utils.type_check(arr, [np.ndarray, np.ma.MaskedArray], "arr")
    core_utils.type_check(target_shape_pixels, [tuple, [int, float]], "target_shape_pixels")
    core_utils.type_check(resample_alg, [str], "resample_alg")

    if len(target_shape_pixels) > 2:
        target_shape_pixels = target_shape_pixels[:2]

    arr_as_raster = core_raster.create_raster_from_array(arr)
    resampled = _resample_raster(arr_as_raster, target_shape_pixels, target_in_pixels=True, resample_alg=resample_alg)
    out_arr = core_raster.raster_to_array(resampled)

    gdal_utils.delete_if_in_memory(arr_as_raster)
    gdal_utils.delete_if_in_memory(resampled)

    return out_arr
