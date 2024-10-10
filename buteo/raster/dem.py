"""Slope, aspect, hillshade, and other DEM functions."""

# Standard library
import os
import sys; sys.path.append("../../")
from typing import Union, Optional, List
from uuid import uuid4

# External
from osgeo import gdal
import numpy as np

# Internal
from buteo.raster.core_raster_io import raster_to_array, array_to_raster
from buteo.utils import utils_base, utils_gdal, utils_io, utils_path



def raster_dem_to_slope(
    raster: Union[str, gdal.Dataset, List[Union[str, gdal.Dataset]]],
    out_path: Optional[Union[str, List[str]]] = None,
    slope_format: str = "percent",
    z_factor: float = 1.0,
    creation_options: Optional[List[str]] = None,
    overwrite: bool = True,
    prefix: str = "",
    suffix: str = "",
    add_uuid: bool = False,
    add_timestamp: bool = False,
    out_format: str = "tif",
) -> str:
    """Calculate the slope of a raster DEM.

    Parameters
    ----------
    raster : Union[str, List, gdal.Dataset]
        The raster(s) to be converted to slope.

    out_path : Optional[str], optional
        The path to the output raster. If None, the raster is created in memory., default: None

    slope_format : str, optional
        "percent" or "degree", by default "percent".

    z_factor : float, optional
        Z factor for slope calculation, by default 1.0.

    creation_options : list, optional
        A list of GDAL creation options for the output raster(s).

    overwrite : bool, optional
        If True, the output raster will be overwritten if it already exists., default: True

    prefix : str, optional
        The prefix to be added to the output raster name., default: ""

    suffix : str, optional
        The suffix to be added to the output raster name., default: ""

    add_uuid : bool, optional
        If True, a unique identifier will be added to the output raster name., default: False

    add_timestamp : bool, optional
        If True, a timestamp will be added to the output raster name., default: False

    out_format : str, optional
        The output format of the raster. If None, the format is inferred from the output path., default: ".tif"

    Returns
    -------
    Union[str, List[str]]
        The path(s) to the raster(s) with slope.
    """
    utils_base._type_check(raster, [str, gdal.Dataset, [str, gdal.Dataset]], "raster")
    utils_base._type_check(out_path, [str, [str], None], "out_path")
    utils_base._type_check(z_factor, [float, int], "z_factor")
    utils_base._type_check(slope_format, [str], "slope_format")
    utils_base._type_check(creation_options, [[str], None], "creation_options")
    utils_base._type_check(overwrite, [bool], "overwrite")
    utils_base._type_check(prefix, [str], "prefix")
    utils_base._type_check(suffix, [str], "suffix")
    utils_base._type_check(add_uuid, [bool], "add_uuid")
    utils_base._type_check(add_timestamp, [bool], "add_timestamp")

    input_is_list = isinstance(raster, list)

    input_list = utils_io._get_input_paths(raster, "raster")
    output_list = utils_io._get_output_paths(
        input_list,
        out_path,
        overwrite=overwrite,
        prefix=prefix,
        suffix=suffix,
        add_uuid=add_uuid,
        add_timestamp=add_timestamp,
        change_ext=out_format,
    )

    utils_path._delete_if_required_list(output_list, overwrite)

    creation_options = utils_gdal._get_default_creation_options(creation_options)

    slope_options = gdal.DEMProcessingOptions(
        computeEdges=True,
        slopeFormat=slope_format,
        zFactor=z_factor,
        creationOptions=creation_options,
    )

    slope_rasters = []
    for idx, in_raster in enumerate(input_list):
        ds = gdal.DEMProcessing(
            output_list[idx],
            in_raster,
            'slope',
            options=slope_options,
        )

        if ds is None:
            raise RuntimeError(f"Error: Could not calculate slope: {in_raster}")

        ds.FlushCache()  # Write to disk.
        ds = None  # Close the dataset and release resources

        slope_rasters.append(output_list[idx])

    if input_is_list:
        return slope_rasters

    return slope_rasters[0]



def raster_dem_to_aspect(
    raster: Union[str, gdal.Dataset, List[Union[str, gdal.Dataset]]],
    out_path: Optional[Union[str, List[str]]] = None,
    zero_for_flat: bool = True,
    creation_options: Optional[List[str]] = None,
    overwrite: bool = True,
    prefix: str = "",
    suffix: str = "",
    add_uuid: bool = False,
    add_timestamp: bool = False,
    out_format: str = "tif",
) -> str:
    """Calculate the aspect in degrees of a raster DEM.

    Parameters
    ----------
    raster : Union[str, List, gdal.Dataset]
        The raster(s) to be converted to aspect.

    out_path : Optional[str], optional
        The path to the output raster. If None, the raster is created in memory., default: None

    zero_for_flat : bool, optional
        If True, set aspect to 0 for flat areas, by default False.

    creation_options : list, optional
        A list of GDAL creation options for the output raster(s).

    overwrite : bool, optional
        If True, the output raster will be overwritten if it already exists., default: True

    prefix : str, optional
        The prefix to be added to the output raster name., default: ""

    suffix : str, optional
        The suffix to be added to the output raster name., default: ""

    add_uuid : bool, optional
        If True, a unique identifier will be added to the output raster name., default: False

    add_timestamp : bool, optional
        If True, a timestamp will be added to the output raster name., default: False

    out_format : str, optional
        The output format of the raster. If None, the format is inferred from the output path., default: ".tif"

    Returns
    -------
    Union[str, List[str]]
        The path(s) to the raster(s) with slope.
    """
    utils_base._type_check(raster, [str, gdal.Dataset, [str, gdal.Dataset]], "raster")
    utils_base._type_check(out_path, [str, [str], None], "out_path")
    utils_base._type_check(zero_for_flat, [bool], "zero_for_flat")
    utils_base._type_check(creation_options, [[str], None], "creation_options")
    utils_base._type_check(overwrite, [bool], "overwrite")
    utils_base._type_check(prefix, [str], "prefix")
    utils_base._type_check(suffix, [str], "suffix")
    utils_base._type_check(add_uuid, [bool], "add_uuid")
    utils_base._type_check(add_timestamp, [bool], "add_timestamp")

    input_is_list = isinstance(raster, list)

    input_list = utils_io._get_input_paths(raster, "raster")
    output_list = utils_io._get_output_paths(
        input_list,
        out_path,
        overwrite=overwrite,
        prefix=prefix,
        suffix=suffix,
        add_uuid=add_uuid,
        add_timestamp=add_timestamp,
        change_ext=out_format,
    )

    utils_path._delete_if_required_list(output_list, overwrite)

    creation_options = utils_gdal._get_default_creation_options(creation_options)

    aspect_options = gdal.DEMProcessingOptions(
        computeEdges=True,
        creationOptions=creation_options,
        zeroForFlat=zero_for_flat,
    )

    aspect_rasters = []
    for idx, in_raster in enumerate(input_list):
        ds = gdal.DEMProcessing(
            output_list[idx],
            in_raster,
            'aspect',
            options=aspect_options,
        )

        if ds is None:
            raise RuntimeError(f"Error: Could not calculate slope: {in_raster}")

        ds.FlushCache()  # Write to disk.
        ds = None  # Close the dataset and release resources

        aspect_rasters.append(output_list[idx])

    if input_is_list:
        return aspect_rasters

    return aspect_rasters[0]



def raster_dem_to_hillshade(
    raster: Union[str, gdal.Dataset, List[Union[str, gdal.Dataset]]],
    out_path: Optional[Union[str, List[str]]] = None,
    z_factor: float = 1.0,
    creation_options: Optional[List[str]] = None,
    overwrite: bool = True,
    prefix: str = "",
    suffix: str = "",
    add_uuid: bool = False,
    add_timestamp: bool = False,
    out_format: str = "tif",
) -> str:
    """Calculate the hillshade of a raster DEM.

    Parameters
    ----------
    raster : Union[str, List, gdal.Dataset]
        The raster(s) to be converted to hillshade.

    out_path : Optional[str], optional
        The path to the output raster. If None, the raster is created in memory., default: None

    z_factor : float, optional
        Z factor for hillshade calculation, by default 1.0.

    creation_options : list, optional
        A list of GDAL creation options for the output raster(s).

    overwrite : bool, optional
        If True, the output raster will be overwritten if it already exists., default: True

    prefix : str, optional
        The prefix to be added to the output raster name., default: ""

    suffix : str, optional
        The suffix to be added to the output raster name., default: ""

    add_uuid : bool, optional
        If True, a unique identifier will be added to the output raster name., default: False

    add_timestamp : bool, optional
        If True, a timestamp will be added to the output raster name., default: False

    out_format : str, optional
        The output format of the raster. If None, the format is inferred from the output path., default: ".tif"

    Returns
    -------
    Union[str, List[str]]
        The path(s) to the raster(s) with slope.
    """
    utils_base._type_check(raster, [str, gdal.Dataset, [str, gdal.Dataset]], "raster")
    utils_base._type_check(out_path, [str, [str], None], "out_path")
    utils_base._type_check(z_factor, [float, int], "z_factor")
    utils_base._type_check(creation_options, [[str], None], "creation_options")
    utils_base._type_check(overwrite, [bool], "overwrite")
    utils_base._type_check(prefix, [str], "prefix")
    utils_base._type_check(suffix, [str], "suffix")
    utils_base._type_check(add_uuid, [bool], "add_uuid")
    utils_base._type_check(add_timestamp, [bool], "add_timestamp")

    input_is_list = isinstance(raster, list)

    input_list = utils_io._get_input_paths(raster, "raster")
    output_list = utils_io._get_output_paths(
        input_list,
        out_path,
        overwrite=overwrite,
        prefix=prefix,
        suffix=suffix,
        add_uuid=add_uuid,
        add_timestamp=add_timestamp,
        change_ext=out_format,
    )

    utils_path._delete_if_required_list(output_list, overwrite)

    creation_options = utils_gdal._get_default_creation_options(creation_options)

    hillshade_options = gdal.DEMProcessingOptions(
        computeEdges=True,
        creationOptions=creation_options,
        zFactor=z_factor,
    )

    hillshade_rasters = []
    for idx, in_raster in enumerate(input_list):
        ds = gdal.DEMProcessing(
            output_list[idx],
            in_raster,
            'hillshade',
            options=hillshade_options,
        )

        if ds is None:
            raise RuntimeError(f"Error: Could not calculate slope: {in_raster}")

        ds.FlushCache()  # Write to disk.
        ds = None  # Close the dataset and release resources

        hillshade_rasters.append(output_list[idx])

    if input_is_list:
        return hillshade_rasters

    return hillshade_rasters[0]


def raster_dem_to_orientation(
    raster: Union[str, gdal.Dataset, List[Union[str, gdal.Dataset]]],
    out_path: Optional[Union[str, List[str]]] = None,
    include_height: bool = True,
    height_normalisation: bool = True,
    height_normalisation_value: float = 8849.0, # mt. everest
    creation_options: Optional[List[str]] = None,
    overwrite: bool = True,
    prefix: str = "",
    suffix: str = "",
    add_uuid: bool = False,
    add_timestamp: bool = False,
    out_format: str = "tif",
) -> str:
    """Calculate the orientation of a raster DEM.

    Parameters
    ----------
    raster : Union[str, List, gdal.Dataset]
        The raster(s) to be converted to orientation.

    out_path : Optional[str], optional
        The path to the output raster. If None, the raster is created in memory., default: None

    include_height : bool, optional
        If True, include height in orientation calculation, by default False.

    height_normalisation : bool, optional
        If True, normalise height to 0-1, by default False.

    height_normalisation_value : float, optional
        Value to normalise height to, by default 8849m. The height of mt. everest.

    creation_options : list, optional
        A list of GDAL creation options for the output raster(s).

    overwrite : bool, optional
        If True, the output raster will be overwritten if it already exists., default: True

    prefix : str, optional
        The prefix to be added to the output raster name., default: ""

    suffix : str, optional
        The suffix to be added to the output raster name., default: ""

    add_uuid : bool, optional
        If True, a unique identifier will be added to the output raster name., default: False

    add_timestamp : bool, optional
        If True, a timestamp will be added to the output raster name., default: False

    out_format : str, optional
        The output format of the raster. If None, the format is inferred from the output path., default: ".tif"

    Returns
    -------
    Union[str, List[str]]
        The path(s) to the raster(s) with slope.
    """
    utils_base._type_check(raster, [str, gdal.Dataset, [str, gdal.Dataset]], "raster")
    utils_base._type_check(out_path, [str, [str], None], "out_path")
    utils_base._type_check(include_height, [bool], "include_height")
    utils_base._type_check(height_normalisation, [bool], "height_normalisation")
    utils_base._type_check(height_normalisation_value, [float, int], "height_normalisation_value")
    utils_base._type_check(creation_options, [[str], None], "creation_options")
    utils_base._type_check(overwrite, [bool], "overwrite")
    utils_base._type_check(prefix, [str], "prefix")
    utils_base._type_check(suffix, [str], "suffix")
    utils_base._type_check(add_uuid, [bool], "add_uuid")
    utils_base._type_check(add_timestamp, [bool], "add_timestamp")

    input_is_list = isinstance(raster, list)

    input_list = utils_io._get_input_paths(raster, "raster")
    output_list = utils_io._get_output_paths(
        input_list,
        out_path,
        overwrite=overwrite,
        prefix=prefix,
        suffix=suffix,
        add_uuid=add_uuid,
        add_timestamp=add_timestamp,
        change_ext=out_format,
    )

    utils_path._delete_if_required_list(output_list, overwrite)

    creation_options = utils_gdal._get_default_creation_options(creation_options)

    orientation_rasters = []
    for idx, in_raster in enumerate(input_list):
        aspect_raster = raster_dem_to_aspect(in_raster, zero_for_flat=True)
        slope_raster = raster_dem_to_slope(in_raster, slope_format="percent")

        aspect = raster_to_array(aspect_raster)
        slope = raster_to_array(slope_raster)

        utils_gdal.delete_dataset_if_in_memory_list([
            aspect_raster,
            slope_raster,
        ])

        if include_height:
            dem = raster_to_array(in_raster)

            if height_normalisation:
                dem_norm = np.zeros(dem.shape, dtype=np.float32)
                dem_norm = np.divide(dem, height_normalisation_value, out=dem_norm, where=dem != 0)
            else:
                dem_norm = dem

            dem = None

        bands = 4 if include_height else 3

        destination = np.zeros((aspect.shape[0], aspect.shape[1], bands), dtype=np.float32)

        aspect_norm = np.zeros(aspect.shape, dtype=np.float32)
        aspect_norm = np.divide(aspect, 360.0, out=aspect_norm, where=aspect != 0)

        encoded_sin = ((np.sin(2 * np.pi * aspect_norm) + 1)) / 2.0
        encoded_cos = ((np.cos(2 * np.pi * aspect_norm) + 1)) / 2.0
        encoded_slope = np.clip(np.divide(slope, 90.0, where=slope != 0), 0.0, 1.0)

        destination[:, :, 0] = encoded_sin[:, :, 0]
        destination[:, :, 1] = encoded_cos[:, :, 0]
        destination[:, :, 2] = encoded_slope[:, :, 0]

        if include_height:
            destination[:, :, 3] = dem_norm[:, :, 0]

        array_to_raster(
            destination,
            reference=in_raster,
            out_path=output_list[idx],
            creation_options=creation_options,
        )

        orientation_rasters.append(output_list[idx])

    if input_is_list:
        return orientation_rasters

    return orientation_rasters[0]
