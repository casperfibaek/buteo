import sys; sys.path.append('../../')
import os
from typing import Union
from osgeo import gdal, osr, ogr
from buteo.utils import remove_if_overwrite
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
    projection: Union[str, ogr.DataSource, gdal.Dataset, osr.SpatialReference],
    out_path: Union[str, None]=None,
    resample_alg: str="nearest",
    align_pixels: bool=False,
    overwrite: bool=True,
    creation_options: list=[],
    dst_nodata: Union[str, int, float]="infer",
) -> Union[gdal.Dataset, str]:
    """
        Reprojects raster to reference raster projection or target projection.
    """
    # Verify Inputs
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
        targetAlignedPixels=align_pixels,
        creationOptions=out_creation_options,
        srcNodata=metadata["nodata_value"],
        dstNodata=out_nodata,
        multithread=True,
    )

    if out_path is not None:
        return out_path
    else:
        return reprojected
