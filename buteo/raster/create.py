"""
### Functions for changing the datatype of a raster. ###
"""

# Standard library
import sys; sys.path.append("../../")
from typing import Union, List

# External
from osgeo import gdal, osr, ogr
import numpy as np

# Internal
from buteo.utils import (
    utils_gdal,
    utils_base,
    utils_path,
    utils_projection,
    utils_translate,
)


def raster_create_empty(
    out_path: Union[str, None] = None,
    width: int = 100,
    height: int = 100,
    pixel_size: Union[Union[float, int], List[Union[float, int]]] = 10.0,
    bands: int = 1,
    dtype: str = "uint8",
    x_min: Union[float, int] = 0.0,
    y_max: Union[float, int] = 0.0,
    nodata_value: Union[float, int, None] = None,
    projection: Union[int, str, gdal.Dataset, ogr.DataSource, osr.SpatialReference] = "EPSG:3857",
    creation_options: Union[List[str], None] = None,
    overwrite: bool = True,
) -> str:
    """
    Create an empty raster.

    Parameters
    ----------
    out_path : str, optional
        The output path. If None, a temporary file will be created.

    width : int, optional
        The width of the raster in pixels. Default: 100.

    height : int, optional
        The height of the raster in pixels. Default: 100.

    pixel_size : int or float or list or tuple, optional
        The pixel size in units of the projection. Default: 10.0.

    bands : int, optional
        The number of bands in the raster. Default: 1.

    dtype : str, optional
        The data type of the raster. Default: "uint8".

    x_min : int or float, optional
        The x coordinate of the top left corner of the raster. Default: 0.0.

    y_max : int or float, optional
        The y coordinate of the top left corner of the raster. Default: 0.0.

    nodata_value : int or float or None, optional
        The nodata value of the raster. Default: None.

    projection : int or str or gdal.Dataset or ogr.DataSource or osr.SpatialReference, optional
        The projection of the raster. Default: "EPSG:3857".

    creation_options : list or None, optional
        A list of creation options. Default: None.

    overwrite : bool, optional
        If True, overwrite the output file if it exists. Default: True.

    Returns
    -------
    str
        The path to the output raster.
    """
    utils_base.type_check(out_path, [str, type(None)], "out_path")
    utils_base.type_check(width, int, "width")
    utils_base.type_check(height, int, "height")
    utils_base.type_check(pixel_size, [int, float, list, tuple], "pixel_size")
    utils_base.type_check(bands, int, "bands")
    utils_base.type_check(dtype, str, "dtype")
    utils_base.type_check(x_min, [int, float], "x_min")
    utils_base.type_check(y_max, [int, float], "y_max")
    utils_base.type_check(projection, [int, str, gdal.Dataset, ogr.DataSource, osr.SpatialReference], "projection")
    utils_base.type_check(creation_options, [list, type(None)], "creation_options")
    utils_base.type_check(overwrite, bool, "overwrite")

     # Parse the driver
    driver_name = "GTiff" if out_path is None else utils_gdal._get_raster_driver_from_path(out_path)
    if driver_name is None:
        raise ValueError(f"Unable to parse filetype from path: {out_path}")

    driver = gdal.GetDriverByName(driver_name)
    if driver is None:
        raise ValueError(f"Error while creating driver from extension: {out_path}")

    output_name = None
    if out_path is None:
        output_name = utils_path._get_output_path("raster_from_array.tif", add_uuid=True)
    else:
        output_name = out_path

    utils_path._delete_if_required(output_name, overwrite)

    destination = driver.Create(
        output_name,
        width,
        height,
        bands,
        utils_translate._translate_str_to_gdal_dtype(dtype),
        utils_gdal._get_default_creation_options(creation_options),
    )

    parsed_projection = utils_projection.parse_projection(projection, return_wkt=True)

    destination.SetProjection(parsed_projection)

    pixel_width = pixel_size if isinstance(pixel_size, (int,  float)) else pixel_size[0]
    pixel_height = pixel_size if isinstance(pixel_size, (int,  float)) else pixel_size[1]

    transform = [x_min, pixel_width, 0, y_max, 0, -pixel_height] # negative for north-up

    destination.SetGeoTransform(transform)

    if nodata_value is not None:
        for band in range(1, bands + 1):
            destination.GetRasterBand(band).SetNoDataValue(nodata_value)

    destination.FlushCache()
    destination = None

    return output_name


def raster_create_from_array(
    arr: np.ndarray,
    out_path: str = None,
    pixel_size: Union[Union[float, int], List[Union[float, int]]] = 10.0,
    x_min: Union[float, int] = 0.0,
    y_max: Union[float, int] = 0.0,
    projection: Union[int, str, gdal.Dataset, ogr.DataSource, osr.SpatialReference] = "EPSG:3857",
    creation_options: Union[List[str], None] = None,
    overwrite: bool = True,
) -> str:
    """ Create a raster from a numpy array.

    Parameters
    ----------
    arr : np.ndarray
        The array to convert to a raster.

    out_path : str, optional
        The output path. If None, a temporary file will be created.

    pixel_size : int or float or list or tuple, optional
        The pixel size of the output raster. Default: 10.0.

    x_min : int or float, optional
        The x coordinate of the top left corner of the output raster. Default: 0.0.

    y_max : int or float, optional
        The y coordinate of the top left corner of the output raster. Default: 0.0.

    projection : int or str or gdal.Dataset or ogr.DataSource or osr.SpatialReference, optional
        The projection of the output raster. Default: "EPSG:3857".

    creation_options : list or None, optional
        The creation options for the output raster. Default: None.

    overwrite : bool, optional
        If True, the output raster will be overwritten if it already exists. Default: True.

    Returns
    -------
    str
        The path to the output raster.
    """
    utils_base.type_check(arr, [np.ndarray, np.ma.MaskedArray], "arr")
    utils_base.type_check(out_path, [str, None], "out_path")
    utils_base.type_check(pixel_size, [int, float, [int, float], tuple], "pixel_size")
    utils_base.type_check(x_min, [int, float], "x_min")
    utils_base.type_check(y_max, [int, float], "y_max")
    utils_base.type_check(projection, [int, str, gdal.Dataset, ogr.DataSource, osr.SpatialReference], "projection")
    utils_base.type_check(creation_options, [[str], None], "creation_options")
    utils_base.type_check(overwrite, [bool], "overwrite")

    assert arr.ndim in [2, 3], "Array must be 2 or 3 dimensional (3rd dimension considered bands.)"

    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]

    # Parse the driver
    driver_name = "GTiff" if out_path is None else utils_gdal._get_raster_driver_from_path(out_path)
    if driver_name is None:
        raise ValueError(f"Unable to parse filetype from path: {out_path}")

    driver = gdal.GetDriverByName(driver_name)
    if driver is None:
        raise ValueError(f"Error while creating driver from extension: {out_path}")

    output_name = None
    if out_path is None:
        output_name = utils_path._get_output_path("raster_from_array.tif", add_uuid=True)
    else:
        output_name = out_path

    utils_path._delete_if_required(output_name, overwrite)

    height, width, bands = arr.shape

    destination = driver.Create(
        output_name,
        width,
        height,
        bands,
        utils_translate._translate_str_to_gdal_dtype(arr.dtype.name),
        utils_gdal._get_default_creation_options(creation_options),
    )

    parsed_projection = utils_projection.parse_projection(projection, return_wkt=True)

    destination.SetProjection(parsed_projection)

    pixel_width = pixel_size if isinstance(pixel_size, (int,  float)) else pixel_size[0]
    pixel_height = pixel_size if isinstance(pixel_size, (int,  float)) else pixel_size[1]

    transform = [x_min, pixel_width, 0, y_max, 0, -pixel_height] # negative for north-up

    destination.SetGeoTransform(transform)

    nodata = None
    if isinstance(arr, np.ma.MaskedArray):
        nodata = arr.fill_value

    for idx in range(0, bands):
        dst_band = destination.GetRasterBand(idx + 1)
        dst_band.WriteArray(arr[:, :, idx])

        if nodata is not None:
            dst_band.SetNoDataValue(nodata)

    return output_name
