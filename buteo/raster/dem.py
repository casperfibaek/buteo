""" Slope, aspect, hillshade, and other DEM functions. """

# Standard library
import sys; sys.path.append("../../")
from typing import Union, Optional, List
from uuid import uuid4

# External
from osgeo import gdal
import numpy as np

# Internal
from buteo.raster.core_raster_io import raster_to_array, array_to_raster
from buteo.utils import utils_base, utils_gdal



def raster_dem_to_slope(
    input_raster: Union[str, gdal.Dataset],
    output_raster: str,
    slope_format: str = "percent",
    z_factor: float = 1.0,
    creation_options: Optional[List[str]] = None,
) -> str:
    """
    Slope in percent.
    
    Parameters
    ----------
    input_raster : str or gdal.Dataset
        Path to input raster or gdal.Dataset.

    output_raster : str
        Path to output raster.

    slope_format : str, optional
        "percent" or "degree", by default "percent".

    z_factor : float, optional
        Z factor for slope calculation, by default 1.0.

    creation_options : list, optional
        A list of GDAL creation options for the output raster(s).

    Returns
    -------
    str
        Path to output raster.
    """
    utils_base._type_check(input_raster, [str, gdal.Dataset], "input_raster")
    utils_base._type_check(output_raster, str, "output_raster")
    utils_base._type_check(z_factor, [float, int], "z_factor")
    utils_base._type_check(slope_format, [str], "slope_format")
    utils_base._type_check(creation_options, [[str], None], "creation_options")

    creation_options = utils_gdal._get_default_creation_options(creation_options)

    slope_options = gdal.DEMProcessingOptions(
        computeEdges=True,
        slopeFormat=slope_format,
        zFactor=z_factor,
        creationOptions=creation_options,
    )

    ds = gdal.DEMProcessing(
        output_raster,
        input_raster,
        'slope',
        options=slope_options,
        format=format,
    )

    if ds is None:
        raise RuntimeError("Error: Could not calculate slope")

    ds.FlushCache()  # Write to disk.
    ds = None  # Close the dataset and release resources

    return output_raster


def raster_dem_to_aspect(
    input_raster: Union[str, gdal.Dataset],
    output_raster: str,
    zero_for_flat: bool = True,
    creation_options: Optional[List[str]] = None,
) -> str:
    """
    Aspect in degrees.
    
    Parameters
    ----------
    input_raster : str or gdal.Dataset
        Path to input raster or gdal.Dataset.

    output_raster : str
        Path to output raster.

    zero_for_flat : bool, optional
        If True, set aspect to 0 for flat areas, by default False.

    Returns
    -------
    str
        Path to output raster.
    """
    utils_base._type_check(input_raster, [str, gdal.Dataset], "input_raster")
    utils_base._type_check(output_raster, str, "output_raster")
    utils_base._type_check(zero_for_flat, [bool], "zero_for_flat")
    utils_base._type_check(creation_options, [[str], None], "creation_options")

    creation_options = utils_gdal._get_default_creation_options(creation_options)

    aspect_options = gdal.DEMProcessingOptions(
        computeEdges=True,
        creationOptions=creation_options,
        zeroForFlat=zero_for_flat,
    )

    ds = gdal.DEMProcessing(
        output_raster,
        input_raster,
        'aspect',
        options=aspect_options,
        format=format,
    )

    if ds is None:
        raise RuntimeError("Error: Could not calculate aspect")

    ds.FlushCache()  # Write to disk.
    ds = None  # Close the dataset and release resources


def raster_dem_to_hillshade(
    input_raster: Union[str, gdal.Dataset],
    output_raster: str,
    z_factor: float = 1.0,
    creation_options: Optional[List[str]] = None,
) -> str:
    """
    Hillshade in degrees.
    
    Parameters
    ----------
    input_raster : str or gdal.Dataset
        Path to input raster or gdal.Dataset.

    output_raster : str
        Path to output raster.

    z_factor : float, optional
        Z factor for hillshade calculation, by default 1.0.

    Returns
    -------
    str
        Path to output raster.
    """
    utils_base._type_check(input_raster, [str, gdal.Dataset], "input_raster")
    utils_base._type_check(output_raster, str, "output_raster")
    utils_base._type_check(z_factor, [float, int], "z_factor")
    utils_base._type_check(creation_options, [[str], None], "creation_options")

    creation_options = utils_gdal._get_default_creation_options(creation_options)

    hillshade_options = gdal.DEMProcessingOptions(
        computeEdges=True,
        creationOptions=creation_options,
        zFactor=z_factor,
    )

    ds = gdal.DEMProcessing(
        output_raster,
        input_raster,
        'hillshade',
        options=hillshade_options,
        format=format,
    )

    if ds is None:
        raise RuntimeError("Error: Could not calculate hillshade")

    ds.FlushCache()  # Write to disk.
    ds = None  # Close the dataset and release resources


def raster_dem_to_orientation(
    input_raster: Union[str, gdal.Dataset],
    output_raster: str,
    include_height: bool = True,
    height_normalisation: bool = True,
    height_normalisation_value: float = 8849.0, # mt. everest
    creation_options: Optional[List[str]] = None,
) -> str:
    """
    Normalised orientation.
    
    Parameters
    ----------
    input_raster : str or gdal.Dataset
        Path to input raster or gdal.Dataset.

    output_raster : str
        Path to output raster.

    include_height : bool, optional
        If True, include height in orientation calculation, by default False.

    height_normalisation : bool, optional
        If True, normalise height to 0-1, by default False.

    height_normalisation_value : float, optional
        Value to normalise height to, by default 1.0.

    Returns
    -------
    str
        Path to output raster.
    """
    utils_base._type_check(input_raster, [str, gdal.Dataset], "input_raster")
    utils_base._type_check(output_raster, str, "output_raster")
    utils_base._type_check(include_height, [bool], "include_height")
    utils_base._type_check(height_normalisation, [bool], "height_normalisation")
    utils_base._type_check(height_normalisation_value, [float, int], "height_normalisation_value")
    utils_base._type_check(creation_options, [[str], None], "creation_options")

    creation_options = utils_gdal._get_default_creation_options(creation_options)

    tmp_path = f"/vsimem/{uuid4().int}_temp_aspect.tif"
    tmp_slope = f"/vsimem/{uuid4().int}_temp_slope.tif"

    raster_dem_to_aspect(input_raster, tmp_path, zero_for_flat=True, )
    raster_dem_to_slope(input_raster, tmp_slope, slope_format="degree")

    aspect = raster_to_array(tmp_path)
    slope = raster_to_array(tmp_slope)

    utils_gdal.delete_dataset_if_in_memory(tmp_path)
    utils_gdal.delete_dataset_if_in_memory(tmp_slope)

    if include_height:
        dem = raster_to_array(input_raster)

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
    encoded_slope = np.divide(slope, 90.0, where=slope != 0)

    destination[:, :, 0] = encoded_sin[:, :, 0]
    destination[:, :, 1] = encoded_cos[:, :, 0]
    destination[:, :, 2] = encoded_slope[:, :, 0]

    if include_height:
        destination[:, :, 3] = dem_norm[:, :, 0]

    array_to_raster(
        destination,
        reference=input_raster,
        out_path=output_raster,
        creation_options=creation_options,
    )

    return output_raster
