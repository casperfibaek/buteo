import sys; sys.path.append('../../')
from typing import Union
from osgeo import gdal, ogr, osr
from buteo.utils import remove_if_overwrite, overwrite_required
from buteo.gdal_utils import (
    parse_projection,
    path_to_driver,
    raster_to_reference,
    default_options,
    translate_resample_method,
    gdal_nodata_value_from_type,
)
from buteo.raster.io import (
    default_options,
    raster_to_disk,
    raster_to_memory,
    raster_to_metadata,
)


def reproject_raster(
    raster: Union[str, gdal.Dataset],
    projection: Union[int, str, gdal.Dataset, ogr.DataSource, osr.SpatialReference],
    out_path: Union[str, None]=None,
    resample_alg: str="nearest",
    overwrite: bool=True,
    creation_options: list=[],
    dst_nodata: Union[str, int, float]="infer",
) -> Union[gdal.Dataset, str]:
    """ Reproject a raster to a target coordinate reference system.

    Args:
        raster (path | raster): The raster to reproject.

        projection (str | int | vector | raster): The projection is infered from
        the input. The input can be: WKT proj, EPSG proj, Proj, osr proj, or read
        from a vector or raster datasource either from path or in-memory.

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
    # Throws an error if file exists and overwrite is False.
    overwrite_required(out_path, overwrite)

    ref = raster_to_reference(raster)
    metadata = raster_to_metadata(ref)

    original_projection = parse_projection(ref)
    target_projection = parse_projection(projection)

    if original_projection.IsSame(target_projection):
        if out_path is None:
            return raster_to_memory(ref)
        
        return raster_to_disk(raster, out_path)

    out_name = None
    out_format = None
    out_creation_options = []
    if out_path is None:
        out_name = metadata["name"]
        out_format = "MEM"
    else:
        out_creation_options = default_options(creation_options)
        out_name = out_path
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

    if out_path is not None:
        return out_path
    else:
        return reprojected
