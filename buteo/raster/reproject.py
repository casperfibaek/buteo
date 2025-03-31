"""### Reproject rasters. ###

Module to reproject rasters to a target coordinate reference system.
Can uses references from vector or other raster datasets.
"""

# Standard library
from typing import Union, Optional, List, Sequence

# External
import numpy as np
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
from buteo.core_raster.core_raster_info import get_metadata_raster
from buteo.core_raster.core_raster_read import _open_raster


def _find_common_projection(
    rasters: Union[str, gdal.Dataset, List[Union[str, gdal.Dataset]]],
):
    """Finds the common projection of a list of rasters.

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
        The most frequently occuring projection.
    """
    utils_base._type_check(rasters, [str, gdal.Dataset, [str, gdal.Dataset]], "rasters")

    rasters_list = utils_base._get_variable_as_list(rasters)

    assert utils_gdal._check_is_raster_list(rasters_list), "rasters must be a list of rasters."

    # Get the projection of each raster
    projections = [utils_projection.parse_projection(raster) for raster in rasters_list]

    # Get the most common projection
    common_projection = max(set(projections), key=projections.count)

    return common_projection


def _raster_reproject(
    raster: Union[str, gdal.Dataset],
    projection: Union[int, str, gdal.Dataset, ogr.DataSource, osr.SpatialReference],
    out_path: Optional[str] = None,
    *,
    resample_alg: str = "nearest",
    copy_if_same: bool = True,
    overwrite: bool = True,
    creation_options: Optional[List[str]] = None,
    dst_nodata: Union[str, int, float] = "infer",
    dtype: Optional[Union[str, np.dtype, type]] = None,
    prefix: str = "",
    suffix: str = "reprojected",
    add_uuid: bool = False,
    add_timestamp: bool = True,
    memory: float = 0.8,
) -> str:
    """Internal reproject implementation.
    
    Parameters
    ----------
    raster : Union[str, gdal.Dataset]
        The raster to reproject.
        
    projection : Union[int, str, gdal.Dataset, ogr.DataSource, osr.SpatialReference]
        The projection to reproject to.
        
    out_path : Optional[str], optional
        Output path, by default None
        
    resample_alg : str, optional
        Resampling algorithm, by default "nearest"
        
    copy_if_same : bool, optional
        Copy if input and output projections are the same, by default True
        
    overwrite : bool, optional
        Overwrite existing files, by default True
        
    creation_options : Optional[List[str]], optional
        GDAL creation options, by default None
        
    dst_nodata : Union[str, int, float], optional
        Output nodata value, by default "infer"
        
    dtype : Optional[Union[str, np.dtype, type]], optional
        Output data type, by default None
        
    prefix : str, optional
        Prefix for output filename, by default ""
        
    suffix : str, optional
        Suffix for output filename, by default "reprojected"
        
    add_uuid : bool, optional
        Add UUID to output filename, by default False
        
    add_timestamp : bool, optional
        Add timestamp to output filename, by default True
        
    memory : float, optional
        Memory limit as a fraction of available memory, by default 0.8
        
    Returns
    -------
    str
        Output path
    """
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

    ref = _open_raster(raster)
    metadata = get_metadata_raster(ref)

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

    utils_io._delete_if_required(out_path, overwrite)

    # Translate for GdalWarp
    resample_alg_gdal = utils_translate._translate_resample_method(resample_alg)
    creation_options = utils_gdal._get_default_creation_options(creation_options)
    memory_limit = utils_gdal._get_dynamic_memory_limit(memory)

    # Handle dtype conversion
    if dtype is None:
        # Use metadata dtype directly if none provided
        output_dtype = utils_translate._translate_dtype_numpy_to_gdal(
            utils_translate._parse_dtype(metadata["dtype"]),
        )
    else:
        output_dtype = utils_translate._translate_dtype_numpy_to_gdal(
            utils_translate._parse_dtype(dtype),
        )
    src_projection = original_projection.ExportToWkt()
    dst_projection = target_projection.ExportToWkt()

    warp_options = gdal.WarpOptions(
        format=out_format,
        srcSRS=src_projection,
        dstSRS=dst_projection,
        resampleAlg=resample_alg_gdal,
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
    raster: Union[str, gdal.Dataset, ogr.DataSource, List[Union[str, gdal.Dataset, ogr.DataSource]]],
    projection: Union[int, str, gdal.Dataset, ogr.DataSource, osr.SpatialReference],
    out_path: Optional[Union[str, List[str]]] = None,
    *,
    resample_alg: str = "nearest",
    copy_if_same: bool = True,
    overwrite: bool = True,
    creation_options: Optional[List[str]] = None,
    dst_nodata: Union[str, int, float] = "infer",
    dtype: Optional[Union[str, np.dtype, type]] = None,
    prefix: str = "",
    suffix: str = "",
    add_uuid: bool = False,
    add_timestamp: bool = True,
) -> Union[str, List[str]]:
    """Reproject raster(s) to a target coordinate reference system.

    Parameters
    ----------
    raster : Union[str, gdal.Dataset, ogr.DataSource, List[Union[str, gdal.Dataset, ogr.DataSource]]]
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

    dtype : Optional[Union[str, np.dtype, type]], optional
        The data type for the output raster, default: None.

    prefix : str, optional
        The prefix to add to the output path, default: "".

    suffix : str, optional
        The suffix to add to the output path, default: "".

    add_uuid : bool, optional
        If True, add a UUID to the output path, default: False.
        
    add_timestamp : bool, optional
        If True, add a timestamp to the output path, default: True.

    Returns
    -------
    Union[str, List[str]]
        The output path(s).
    """
    utils_base._type_check(raster, [str, gdal.Dataset, ogr.DataSource, [str, gdal.Dataset, ogr.DataSource]], "raster")
    utils_base._type_check(projection, [int, str, gdal.Dataset, ogr.DataSource, osr.SpatialReference], "projection")
    utils_base._type_check(out_path, [str, [str], None], "out_path")
    utils_base._type_check(resample_alg, [str], "resample_alg")
    utils_base._type_check(copy_if_same, [bool], "copy_if_same")
    utils_base._type_check(overwrite, [bool], "overwrite")
    utils_base._type_check(creation_options, [[str], None], "creation_options")
    utils_base._type_check(dst_nodata, [str, int, float], "dst_nodata")
    utils_base._type_check(dtype, [str, None, np.dtype, type], "dtype")
    utils_base._type_check(prefix, [str], "prefix")
    utils_base._type_check(suffix, [str], "suffix")
    utils_base._type_check(add_uuid, [bool], "add_uuid")
    utils_base._type_check(add_timestamp, [bool], "add_timestamp")

    input_is_list = isinstance(raster, list)

    in_paths = utils_io._get_input_paths(raster, "raster")
    
    # Handle output paths
    if out_path is None:
        # Create temp paths for each input
        out_paths = []
        for path in in_paths:
            temp_path = utils_path._get_temp_filepath(
                path,
                prefix=prefix,
                suffix=suffix,
                add_uuid=add_uuid,
                add_timestamp=add_timestamp,
                ext="tif",
            )
            out_paths.append(temp_path)
    elif isinstance(out_path, list):
        # Use provided output paths directly
        if len(out_path) != len(in_paths):
            raise ValueError("Number of output paths must match number of input paths")
        out_paths = out_path
    else:
        # Single output path for a single input
        if len(in_paths) > 1:
            raise ValueError("Single output path provided for multiple inputs")
        out_paths = [out_path]

    utils_io._check_overwrite_policy(out_paths, overwrite)
    utils_io._delete_if_required_list(out_paths, overwrite)

    output = []
    for index, in_raster in enumerate(in_paths):
        output.append(
            _raster_reproject(
                in_raster,
                projection,
                out_path=out_paths[index],
                resample_alg=resample_alg,
                copy_if_same=copy_if_same,
                overwrite=overwrite,
                creation_options=creation_options,
                dst_nodata=dst_nodata,
                dtype=dtype,
                prefix=prefix,
                suffix=suffix,
                add_timestamp=add_timestamp,
            )
        )

    if input_is_list:
        return output

    return output[0]
