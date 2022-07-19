"""
### Reproject rasters. ###

Module to reproject rasters to a target coordinate reference system.
Can uses references from vector or other raster datasets.
"""

# Standard library
import sys; sys.path.append("../../")

# External
from osgeo import gdal, ogr, osr

# Internal
from buteo.utils import core_utils, gdal_utils, gdal_enums
from buteo.raster import core_raster



def _reproject_raster(
    raster,
    projection,
    out_path=None,
    *,
    resample_alg="nearest",
    copy_if_same=True,
    overwrite=True,
    creation_options=None,
    dst_nodata="infer",
    prefix="",
    suffix="",
    add_uuid=False,
):
    """ Internal. """
    raster_list = core_utils.ensure_list(raster)
    path_list = core_utils.create_output_paths(
        raster_list,
        out_path=out_path,
        overwrite=overwrite,
        prefix=prefix,
        suffix=suffix,
        add_uuid=add_uuid,
    )

    out_name = path_list[0]
    ref = core_raster._open_raster(raster_list[0])
    metadata = core_raster._raster_to_metadata(ref)

    out_creation_options = gdal_utils.default_creation_options(creation_options)
    out_format = gdal_utils.path_to_driver_raster(out_name)

    original_projection = gdal_utils.parse_projection(ref)
    target_projection = gdal_utils.parse_projection(projection)

    if not isinstance(original_projection, osr.SpatialReference):
        raise Exception("Error while parsing input projection.")

    if not isinstance(target_projection, osr.SpatialReference):
        raise Exception("Error while parsing target projection.")

    if original_projection.IsSame(target_projection):
        if not copy_if_same:
            return gdal_utils.get_path_from_dataset(ref)

    src_nodata = metadata["nodata_value"]
    out_nodata = None
    if src_nodata is not None:
        out_nodata = src_nodata
    else:
        if dst_nodata == "infer":
            out_nodata = gdal_enums.translate_gdal_dtype_to_str(metadata["datatype_gdal_raw"])
        elif isinstance(dst_nodata, str):
            raise TypeError(f"dst_nodata is in a wrong format: {dst_nodata}")
        else:
            out_nodata = dst_nodata

    core_utils.remove_if_required(out_path, overwrite)

    reprojected = gdal.Warp(
        out_name,
        ref,
        format=out_format,
        srcSRS=original_projection,
        dstSRS=target_projection,
        resampleAlg=gdal_enums.translate_resample_method(resample_alg),
        creationOptions=out_creation_options,
        srcNodata=metadata["nodata_value"],
        dstNodata=out_nodata,
        multithread=True,
    )

    if reprojected is None:
        raise Exception(f"Error while reprojecting raster: {raster}")

    return out_name


def reproject_raster(
    raster,
    projection,
    out_path=None,
    *,
    resample_alg="nearest",
    copy_if_same=True,
    overwrite=True,
    creation_options=None,
    dst_nodata="infer",
    prefix="",
    suffix="_reprojected",
    add_uuid=False,
):
    """
    Reproject a raster(s) to a target coordinate reference system.

    ## Args:
    `raster` (_str_/_list_/_gdal.Dataset): The raster(s) to reproject. </br>
    `projection` (_str_/_int_/_ogr.DataSource_/_gdal.Dataset_): The projection is infered from
    the input. The input can be: WKT proj, EPSG proj, Proj, osr proj, or read
    from a vector or raster datasource either from path or in-memory. </br>

    ## Kwargs:
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
    (_str_/_list_): The output path(s) of the reprojected raster(s).
    """
    core_utils.type_check(raster, [str, gdal.Dataset, [str, gdal.Dataset]], "raster")
    core_utils.type_check(projection, [int, str, gdal.Dataset, ogr.DataSource, osr.SpatialReference], "projection")
    core_utils.type_check(out_path, [list, str, None], "out_path")
    core_utils.type_check(resample_alg, [str], "resample_alg")
    core_utils.type_check(overwrite, [bool], "overwrite")
    core_utils.type_check(creation_options, [[str], None], "creation_options")
    core_utils.type_check(dst_nodata, [str, int, float], "dst_nodata")
    core_utils.type_check(prefix, [str], "prefix")
    core_utils.type_check(suffix, [str], "postfix")
    core_utils.type_check(add_uuid, [bool], "add_uuid")

    raster_list = core_utils.ensure_list(raster)
    path_list = core_utils.create_output_paths(
        raster_list,
        out_path=out_path,
        overwrite=overwrite,
        prefix=prefix,
        suffix=suffix,
        add_uuid=add_uuid,
    )

    output = []
    for index, in_raster in enumerate(raster_list):
        output.append(
            _reproject_raster(
                in_raster,
                projection,
                out_path=path_list[index],
                resample_alg=resample_alg,
                copy_if_same=copy_if_same,
                overwrite=overwrite,
                creation_options=creation_options,
                dst_nodata=dst_nodata,
                prefix=prefix,
                suffix=suffix,
            )
        )

    if isinstance(raster, list):
        return output

    return output[0]
