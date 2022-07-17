"""
Module to resample rasters to a target resolution.
Can uses references other raster datasets.

TODO:
    - Remove typings
    - Improve documentations
    - Remove internal function.
"""

import sys; sys.path.append("../../") # Path: buteo/raster/resample.py

from osgeo import gdal

from buteo.utils.core_utils import (
    remove_if_overwrite,
    type_check,
)
from buteo.utils.gdal_utils import (
    path_to_driver_raster,
    default_options,
    translate_resample_method,
    gdal_nodata_value_from_type,
    raster_size_from_list,
    translate_datatypes,
)
from buteo.raster.core_raster import (
    open_raster,
    ready_io_raster,
    raster_to_metadata,
)


def _resample_raster(
    raster,
    target_size,
    target_in_pixels,
    out_path=None,
    *,
    resample_alg="nearest",
    overwrite=True,
    creation_options=[],
    dtype=None,
    dst_nodata="infer",
    prefix="",
    postfix="_resampled",
    add_uuid=False,
):
    """OBS: Internal. Single output.

    Reprojects a raster given a target projection. Beware if your input is in
    latitude and longitude, you'll need to specify the target_size in degrees as well.
    """
    raster_list, path_list = ready_io_raster(
        raster,
        out_path,
        overwrite=overwrite,
        prefix=prefix,
        postfix=postfix,
        add_uuid=add_uuid,
    )

    ref = open_raster(raster_list[0])
    metadata = raster_to_metadata(ref)
    out_name = path_list[0]

    x_res, y_res, x_pixels, y_pixels = raster_size_from_list(
        target_size, target_in_pixels
    )

    out_creation_options = default_options(creation_options)
    out_format = path_to_driver_raster(out_name)

    src_nodata = metadata["nodata_value"]
    out_nodata = None
    if src_nodata is not None:
        out_nodata = src_nodata
    else:
        if dst_nodata == "infer":
            out_nodata = gdal_nodata_value_from_type(metadata["datatype_gdal_raw"])
        elif isinstance(dst_nodata, str):
            raise TypeError(f"dst_nodata is in a wrong format: {dst_nodata}")
        else:
            out_nodata = dst_nodata

    remove_if_overwrite(out_path, overwrite)

    resampled = gdal.Warp(
        out_name,
        ref,
        width=x_pixels,
        height=y_pixels,
        xRes=x_res,
        yRes=y_res,
        format=out_format,
        outputType=translate_datatypes(dtype),
        resampleAlg=translate_resample_method(resample_alg),
        creationOptions=out_creation_options,
        srcNodata=metadata["nodata_value"],
        dstNodata=out_nodata,
        multithread=True,
    )

    if resampled is None:
        raise Exception(f"Error while resampling raster: {out_name}")

    return out_name


def resample_raster(
    raster,
    target_size,
    target_in_pixels=False,
    out_path=None,
    *,
    resample_alg="nearest",
    overwrite=True,
    creation_options=[],
    dtype=None,
    dst_nodata="infer",
    prefix="",
    postfix="_resampled",
):
    """Reprojects a raster given a target projection. Beware if your input is in
        latitude and longitude, you'll need to specify the target_size in degrees as well.

    Args:
        raster (list, path | raster): The raster to reproject.

        target_size (str | int | vector | raster): The target resolution of the
        raster. In the same unit as the projection of the raster.
        It's better to reproject to a projected coordinate system for resampling.
        If a raster is the target_size the function will read the pixel size from
        that raster.

    **kwargs:
        out_path (path | None): The destination to save to. If None then
        the output is an in-memory raster.

        resample_alg (str): The algorithm to resample the raster. The following
        are available:
            'nearest', 'bilinear', 'cubic', 'cubicSpline', 'lanczos', 'average',
            'mode', 'max', 'min', 'median', 'q1', 'q3', 'sum', 'rms'.

        overwite (bool): Is it possible to overwrite the out_path if it exists.

        creation_options (list): A list of options for the GDAL creation. Only
        used if an outpath is specified. Defaults are:
            "TILED=YES"
            "NUM_THREADS=ALL_CPUS"
            "BIGG_TIF=YES"
            "COMPRESS=LZW"

        dst_nodata (str | int | float): If dst_nodata is 'infer' the destination nodata
        is the src_nodata if one exists, otherwise it's automatically chosen based
        on the datatype. If an int or a float is given, it is used as the output nodata.

    Returns:
        An in-memory raster. If an out_path is given the output is a string containing
        the path to the newly created raster.
    """
    type_check(raster, [list, str, gdal.Dataset], "raster")
    type_check(target_size, [tuple, int, float, str, gdal.Dataset], "target_size")
    type_check(target_in_pixels, [bool], "target_in_pixels")
    type_check(out_path, [list, str], "out_path", allow_none=True)
    type_check(resample_alg, [str], "resample_alg")
    type_check(overwrite, [bool], "overwrite")
    type_check(creation_options, [list], "creation_options")
    type_check(dst_nodata, [str, int, float], "dst_nodata", allow_none=True)
    type_check(prefix, [str], "prefix")
    type_check(postfix, [str], "postfix")

    raster_list, path_list = ready_io_raster(
        raster, out_path, overwrite=overwrite, prefix=prefix, postfix=postfix
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
                postfix=postfix,
            )
        )

    if isinstance(raster, list):
        return resampled_rasters

    return resampled_rasters[0]
