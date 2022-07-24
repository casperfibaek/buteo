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
    dtype=None,
    prefix="",
    suffix="",
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

    out_format = gdal_utils.path_to_driver_raster(out_path)

    original_projection = gdal_utils.parse_projection(ref)
    target_projection = gdal_utils.parse_projection(projection)

    if not isinstance(original_projection, osr.SpatialReference):
        raise Exception("Error while parsing input projection.")

    if not isinstance(target_projection, osr.SpatialReference):
        raise Exception("Error while parsing target projection.")

    # The input is already in the correct projection.
    if not copy_if_same and gdal_utils.projections_match(original_projection, target_projection):
        return gdal_utils.get_path_from_dataset(ref)

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

    reprojected = gdal.Warp(
        out_path,
        ref,
        format=out_format,
        srcSRS=original_projection,
        dstSRS=target_projection,
        resampleAlg=gdal_enums.translate_resample_method(resample_alg),
        outputType=gdal_enums.translate_str_to_gdal_dtype(dtype),
        creationOptions=gdal_utils.default_creation_options(creation_options),
        srcNodata=src_nodata,
        dstNodata=out_nodata,
        multithread=True,
    )

    if reprojected is None:
        raise Exception(f"Error while reprojecting raster: {raster}")

    return out_path


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
    dtype=None,
    prefix="",
    suffix="",
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
    core_utils.type_check(dtype, [str, None], "dtype")
    core_utils.type_check(prefix, [str], "prefix")
    core_utils.type_check(suffix, [str], "postfix")
    core_utils.type_check(add_uuid, [bool], "add_uuid")

    if core_utils.is_str_a_glob(raster):
        raster = core_utils.parse_glob_path(raster)

    raster_list = core_utils.ensure_list(raster)

    assert gdal_utils.is_raster_list(raster_list), f"The input raster(s) contains invalid elements: {raster_list}"

    path_list = gdal_utils.create_output_path_list(
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
                dtype=dtype,
                prefix=prefix,
                suffix=suffix,
            )
        )

    if isinstance(raster, list):
        return output

    return output[0]
