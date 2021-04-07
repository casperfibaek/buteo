import sys; sys.path.append('../../')
from uuid import uuid4
from typing import Union
from osgeo import gdal
from buteo.utils import (
    remove_if_overwrite,
    overwrite_required,
    type_check,
    folder_exists,
)
from buteo.gdal_utils import (
    path_to_driver,
    raster_to_reference,
    default_options,
    ready_io_raster,
    translate_resample_method,
    gdal_nodata_value_from_type,
    raster_size_from_list,
    translate_datatypes,
    to_raster_list,
)
from buteo.raster.io import (
    default_options,
    raster_to_metadata,
)


def resample_raster(
    raster: Union[list, str, gdal.Dataset],
    target_size: Union[tuple, int, float, str, gdal.Dataset],
    target_in_pixels: bool=False,
    out_path: Union[list, str, None]=None,
    resample_alg: str="nearest",
    overwrite: bool=True,
    creation_options: list=[],
    dtype=None,
    dst_nodata: Union[str, int, float]="infer",
    opened: bool=False,
    prefix: str="",
    postfix: str="_resampled",
) -> Union[list, gdal.Dataset, str]:
    """ Reprojects a raster given a target projection. Beware if your input is in
        latitude and longitude, you'll need to specify the target_sizedegrees as well.

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
    type_check(dst_nodata, [str, int, float], "dst_nodata")
    type_check(prefix, [str], "prefix")
    type_check(postfix, [list], "postfix")
    type_check(opened, [bool], "opened")

    raster_list, out_names = ready_io_raster(raster, out_path, overwrite, prefix, postfix)

    resampled_rasters = []

    for index, in_raster in enumerate(raster_list):
        ref = raster_to_reference(in_raster)
        metadata = raster_to_metadata(ref)

        x_res, y_res, x_pixels, y_pixels = raster_size_from_list(target_size, target_in_pixels)

        out_name = out_names[index]
        out_creation_options = default_options(creation_options)
        out_format = path_to_driver(out_path)
        
        src_nodata = metadata["nodata_value"]
        out_nodata = None
        if src_nodata is not None:
            out_nodata = src_nodata
        else:
            if dst_nodata == "infer":
                out_nodata = gdal_nodata_value_from_type(metadata["dtype_gdal_raw"])
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

        if opened:
            resampled_rasters.append(resampled)
        else:
            resampled_rasters.append(out_name)

    if isinstance(raster, list):
        return resampled_rasters

    return resampled_rasters[0]