"""
### Reproject rasters. ###

Module to reproject rasters to a target coordinate reference system.
Can uses references from vector or other raster datasets.
"""

# Standard library
import sys; sys.path.append("../../")
from typing import Union, Optional, List

# External
from osgeo import gdal, ogr, osr

# Internal
from buteo.utils import (
    utils_io,
    utils_base,
    utils_gdal,
    utils_path,
    utils_projection,
    utils_translate,
)
from buteo.raster import core_raster


def _find_common_projection(
    rasters: Union[str, gdal.Dataset, List[Union[str, gdal.Dataset]]],
):
    """
    Finds the common projection of a list of rasters.

    If no rasters have the majority of the same projection, the function will return the
    projection of the first raster. If only one raster is provided, the projection of that
    raster will be returned.

    Parameters
    ----------
    rasters : Union[str, gdal.Dataset, List[Union[str, gdal.Dataset]]]
        A list of rasters.

    Returns
    -------
    osr.SpatialReference
        The common projection.
    """
    utils_base._type_check(rasters, [str, gdal.Dataset, [str, gdal.Dataset]], "rasters")

    rasters = utils_base._get_variable_as_list(rasters)

    assert utils_gdal._check_is_raster_list(rasters), "rasters must be a list of rasters."

    # Get the projection of each raster
    projections = [utils_projection.parse_projection(raster) for raster in rasters]

    # Get the most common projection
    common_projection = max(set(projections), key=projections.count)

    return common_projection


def _raster_reproject(
    raster: Union[str, gdal.Dataset],
    projection: Union[int, str, gdal.Dataset, ogr.DataSource, osr.SpatialReference],
    out_path: Optional[Union[str, List[str]]] = None,
    *,
    resample_alg: str = "nearest",
    copy_if_same: bool = True,
    overwrite: bool = True,
    creation_options: Optional[List[str]] = None,
    dst_nodata: Union[str, int, float] = "infer",
    dtype: Optional[str] = None,
    prefix: str = "",
    suffix: str = "reprojected",
    add_uuid: bool = False,
    add_timestamp: bool = True,
    memory: float = 0.8,
) -> Union[str, List[str]]:
    """ Internal. """
    assert isinstance(raster, (gdal.Dataset, str)), f"The input raster must be in the form of a str or a gdal.Dataset: {raster}"

    if out_path is None:
        out_path = utils_path._get_temp_filepath(
            name="reprojected_raster.tif",
            prefix=prefix,
            suffix=suffix,
            add_uuid=add_uuid,
            add_timestamp=add_timestamp,
        )
    else:
        if not utils_path._check_is_valid_output_filepath(out_path):
            raise ValueError(f"Invalid output path: {out_path}")

    ref = core_raster._raster_open(raster)
    metadata = core_raster._get_basic_metadata_raster(ref)

    out_format = utils_gdal._get_raster_driver_name_from_path(out_path)

    original_projection = utils_projection.parse_projection(ref)
    target_projection = utils_projection.parse_projection(projection)

    if not isinstance(original_projection, osr.SpatialReference):
        raise RuntimeError("Error while parsing input projection.")

    if not isinstance(target_projection, osr.SpatialReference):
        raise RuntimeError("Error while parsing target projection.")

    # The input is already in the correct projection.
    if not copy_if_same and utils_projection._check_projections_match(original_projection, target_projection):
        return utils_gdal._get_path_from_dataset(ref)

    src_nodata = metadata["nodata_value"]
    out_nodata = None
    if dst_nodata == "infer":
        dst_nodata = src_nodata
    else:
        assert isinstance(dst_nodata, (int, float, type(None))), "dst_nodata must be an int, float, 'infer', or 'None'"
        out_nodata = dst_nodata

    if dtype is None:
        dtype = metadata["dtype"]

    utils_path._delete_if_required(out_path, overwrite)

    # Translate for GdalWarp
    resample_alg = utils_translate._translate_resample_method(resample_alg)
    creation_options = utils_gdal._get_default_creation_options(creation_options)
    memory_limit = utils_gdal._get_dynamic_memory_limit(memory)
    output_dtype = utils_translate._translate_dtype_numpy_to_gdal(
        utils_translate._parse_dtype(dtype),
    )
    src_projection = original_projection.ExportToWkt()
    dst_projection = target_projection.ExportToWkt()

    warp_options = gdal.WarpOptions(
        format=out_format,
        srcSRS=src_projection,
        dstSRS=dst_projection,
        resampleAlg=resample_alg,
        outputType=output_dtype,
        creationOptions=creation_options,
        srcNodata=src_nodata,
        dstNodata=out_nodata,
        warpMemoryLimit=memory_limit,
        multithread=True,
    )

    reprojected = gdal.Warp(out_path, ref, options=warp_options)

    if reprojected is None:
        raise RuntimeError(f"Error while reprojecting raster: {raster}")

    return out_path


def raster_reproject(
    raster: Union[str, gdal.Dataset, List[Union[str, gdal.Dataset]]],
    projection: Union[int, str, gdal.Dataset, ogr.DataSource, osr.SpatialReference],
    out_path: Optional[Union[str, List[str]]] = None,
    *,
    resample_alg: str = "nearest",
    copy_if_same: bool = True,
    overwrite: bool = True,
    creation_options: Optional[List[str]] = None,
    dst_nodata: Union[str, int, float] = "infer",
    dtype: Optional[str] = None,
    prefix: str = "",
    suffix: str = "",
    add_uuid: bool = False,
) -> Union[str, List[str]]:
    """
    Reproject raster(s) to a target coordinate reference system.

    Parameters
    ----------
    raster : Union[str, gdal.Dataset, List[Union[str, gdal.Dataset]]]
        The raster(s) to reproject.

    projection : Union[int, str, gdal.Dataset, ogr.DataSource, osr.SpatialReference]
        The projection to reproject the raster to. The input can be a WKT proj,
        EPSG proj, Proj, osr proj, or read from a vector or raster datasource
        either from path or in-memory.

    out_path : Optional[Union[str, List[str]]], optional
        The output path, default: None. If not provided, the output path is inferred
        from the input.

    resample_alg : str, optional
        The resampling algorithm, default: "nearest".

    copy_if_same : bool, optional
        If the input and output projections are the same, copy the input raster to the
        output path, default: True.

    overwrite : bool, optional
        If the output path already exists, overwrite it, default: True.

    creation_options : Optional[List[str]], optional
        A list of creation options for the output raster, default: None.

    dst_nodata : Union[str, int, float], optional
        The nodata value for the output raster, default: "infer".

    dtype : Optional[str], optional
        The data type for the output raster, default: None.

    prefix : str, optional
        The prefix to add to the output path, default: "".

    suffix : str, optional
        The suffix to add to the output path, default: "".

    add_uuid : bool, optional
        If True, add a UUID to the output path, default: False.

    Returns
    -------
    Union[str, List[str]]
        The output path(s).
    """
    utils_base._type_check(raster, [str, gdal.Dataset, [str, gdal.Dataset]], "raster")
    utils_base._type_check(projection, [int, str, gdal.Dataset, ogr.DataSource, osr.SpatialReference], "projection")
    utils_base._type_check(out_path, [list, str, None], "out_path")
    utils_base._type_check(resample_alg, [str], "resample_alg")
    utils_base._type_check(overwrite, [bool], "overwrite")
    utils_base._type_check(creation_options, [[str], None], "creation_options")
    utils_base._type_check(dst_nodata, [str, int, float], "dst_nodata")
    utils_base._type_check(dtype, [str, None], "dtype")
    utils_base._type_check(prefix, [str], "prefix")
    utils_base._type_check(suffix, [str], "postfix")
    utils_base._type_check(add_uuid, [bool], "add_uuid")

    input_is_list = isinstance(raster, list)

    raster_list = utils_io._get_input_paths(raster, "raster")
    out_path_list = utils_io._get_output_paths(
        raster_list,
        out_path,
        overwrite=overwrite,
        prefix=prefix,
        suffix=suffix,
        add_uuid=add_uuid,
        change_ext="tif",
    )

    utils_path._delete_if_required_list(out_path_list, overwrite)

    output = []
    for index, in_raster in enumerate(raster_list):
        output.append(
            _raster_reproject(
                in_raster,
                projection,
                out_path=out_path_list[index],
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

    if input_is_list:
        return output

    return output[0]
