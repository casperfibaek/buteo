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
from buteo.utils import core_utils, gdal_utils, gdal_enums
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
    core_utils.type_check(rasters, [str, gdal.Dataset, [str, gdal.Dataset]], "rasters")

    rasters = core_utils.ensure_list(rasters)

    assert gdal_utils.is_raster_list(rasters), "rasters must be a list of rasters."

    # Get the projection of each raster
    projections = [gdal_utils.parse_projection(raster) for raster in rasters]

    # Get the most common projection
    common_projection = max(set(projections), key=projections.count)

    return common_projection


def _reproject_raster(
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
    suffix: str = "",
    add_uuid: bool = False,
) -> Union[str, List[str]]:
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
        raise RuntimeError("Error while parsing input projection.")

    if not isinstance(target_projection, osr.SpatialReference):
        raise RuntimeError("Error while parsing target projection.")

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
        raise RuntimeError(f"Error while reprojecting raster: {raster}")

    return out_path


def reproject_raster(
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


def match_raster_projections(
    rasters: Union[str, gdal.Dataset, List[Union[str, gdal.Dataset]]],
    reference: Union[str, gdal.Dataset, ogr.DataSource],
    *,
    out_path: Optional[Union[str, List[str]]] = None,
    overwrite: bool = True,
    dst_nodata: Union[str, int, float] = "infer",
    copy_if_already_correct: bool = True,
    creation_options: Optional[List[str]] = None,
) -> List[str]:
    """
    Match a raster or list of rasters to a master layer. The master can be
    either an OGR layer or a GDAL layer.

    Parameters
    ----------
    rasters : Union[str, gdal.Dataset, List[Union[str, gdal.Dataset]]]
        A list of rasters to match.

    reference : Union[str, gdal.Dataset, ogr.DataSource]
        Path to the reference raster or vector.

    out_path : Optional[Union[str, List[str]]], optional
        Paths to the output. If not provided, the output will be in-memory rasters., default: None

    overwrite : bool, optional
        If True, existing rasters will be overwritten., default: True

    dst_nodata : Union[str, int, float], optional
        Value to use for no-data pixels. If not provided, the value will be transfered from the original., default: "infer"

    copy_if_already_correct : bool, optional
        If True, the raster will be copied if it is already in the correct projection., default: True

    creation_options : Optional[List[str]], optional
        List of creation options to pass to the output raster., default: None

    Returns
    -------
    List[str]
        A list of reprojected input rasters with the correct projection.
    """
    core_utils.type_check(rasters, [str, gdal.Dataset, [str, gdal.Dataset]], "rasters")
    core_utils.type_check(reference, [str, gdal.Dataset, ogr.DataSource], "reference")
    core_utils.type_check(out_path, [str, list, None], "out_path")
    core_utils.type_check(overwrite, [bool], "overwrite")
    core_utils.type_check(dst_nodata, [str, int, float], "dst_nodata")
    core_utils.type_check(copy_if_already_correct, [bool], "copy_if_already_correct")
    core_utils.type_check(creation_options, [[str], None], "creation_options")

    rasters = core_utils.ensure_list(rasters)

    assert gdal_utils.is_raster_list(rasters), "rasters must be a list of rasters."

    try:
        target_projection = gdal_utils.parse_projection(reference)
    except Exception:
        raise ValueError(f"Unable to parse projection from master. Received: {reference}") from None

    add_uuid = out_path is None

    path_list = gdal_utils.create_output_path_list(rasters, out_path, overwrite=overwrite, add_uuid=add_uuid, ext=".tif")

    output = []

    for index, in_raster in enumerate(rasters):
        path = _reproject_raster(
            in_raster,
            target_projection,
            out_path=path_list[index],
            overwrite=overwrite,
            copy_if_same=copy_if_already_correct,
            dst_nodata=dst_nodata,
            creation_options=gdal_utils.default_creation_options(creation_options),
        )

        output.append(path)

    return output
