import sys; sys.path.append('../../')
from typing import Union
from osgeo import gdal, ogr, osr
from buteo.utils import remove_if_overwrite, type_check
from buteo.gdal_utils import (
    parse_projection,
    path_to_driver,
    raster_to_reference,
    default_options,
    translate_resample_method,
    gdal_nodata_value_from_type,
    ready_io_raster,
)
from buteo.raster.io import (
    default_options,
    raster_to_disk,
    raster_to_memory,
    raster_to_metadata,
)


def reproject_raster(
    raster: Union[list, str, gdal.Dataset],
    projection: Union[int, str, gdal.Dataset, ogr.DataSource, osr.SpatialReference],
    out_path: Union[list, str, None]=None,
    resample_alg: str="nearest",
    overwrite: bool=True,
    creation_options: list=[],
    dst_nodata: Union[str, int, float]="infer",
    prefix: str="",
    postfix: str="_reprojected",
    opened: bool=False,
) -> Union[gdal.Dataset, str]:
    """ Reproject a raster(s) to a target coordinate reference system.

    Args:
        raster(s) (list, path | raster): The raster(s) to reproject.

        projection (str | int | vector | raster): The projection is infered from
        the input. The input can be: WKT proj, EPSG proj, Proj, osr proj, or read
        from a vector or raster datasource either from path or in-memory.

    **kwargs:
        out_path (list, path | None): The destination to save to. If None then
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
    type_check(projection, [int, str, gdal.Dataset, ogr.DataSource, osr.SpatialReference], "projection")
    type_check(out_path, [list, str], "out_path", allow_none=True)
    type_check(resample_alg, [str], "resample_alg")
    type_check(overwrite, [bool], "overwrite")
    type_check(creation_options, [list], "creation_options")
    type_check(dst_nodata, [str, int, float], "dst_nodata")
    type_check(prefix, [str], "prefix")
    type_check(postfix, [str], "postfix")
    type_check(opened, [bool], "opened")

    raster_list, out_names = ready_io_raster(raster, out_path, overwrite, prefix, postfix)

    reprojected_rasters = []

    for index, in_raster in enumerate(raster_list):
        ref = raster_to_reference(in_raster)
        metadata = raster_to_metadata(ref)

        out_name = out_names[index]
        out_creation_options = default_options(creation_options)
        out_format = path_to_driver(out_name)

        original_projection = parse_projection(ref)
        target_projection = parse_projection(projection)

        if original_projection.IsSame(target_projection):
            if out_path is None:
                reprojected_rasters.append(raster_to_memory(ref, opened=opened))
            else:
                reprojected_rasters.append(raster_to_disk(raster, out_name, opened=opened))

            continue
        
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

        reprojected = gdal.Warp(
            out_name,
            ref,
            format=out_format,
            srcSRS=original_projection,
            dstSRS=target_projection,
            resampleAlg=translate_resample_method(resample_alg),
            creationOptions=out_creation_options,
            srcNodata=metadata["nodata_value"],
            dstNodata=out_nodata,
            multithread=True,
        )

        if opened:
            reprojected_rasters.append(reprojected)
        else:
            reprojected_rasters.append(out_name)

    if isinstance(raster, list):
        return reprojected_rasters
    
    return reprojected_rasters[0]
