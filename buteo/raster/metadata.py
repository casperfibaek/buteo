"""### Read metadata from rasters. ###"""

# Standard library
from typing import List, Union

# External
from osgeo import gdal

# Internal
from buteo.utils import utils_io, utils_base
from buteo.core_raster.core_raster_info import get_metadata_raster



def _raster_to_metadata(
    raster: Union[str, gdal.Dataset],
) -> dict:
    """Internal."""
    utils_base._type_check(raster, [str, gdal.Dataset], "raster")

    metadata = get_metadata_raster(raster)

    return metadata


def raster_to_metadata(
    raster: Union[str, gdal.Dataset, List[str], List[gdal.Dataset]],
) -> Union[dict, List[dict]]:
    """Reads metadata from a raster dataset or a list of raster datasets, and returns a dictionary or a list of dictionaries
    containing metadata information for each raster.

    Bounding boxes are in the format `[xmin, xmax, ymin, ymax]`. These are the keys in the dictionary:

    - `path` (str): Path to the raster.
    - `basename` (str): Basename of the raster.
    - `name` (str): Name of the raster without extension.
    - `folder` (str): Folder of the raster.
    - `ext` (str): Extension of the raster.
    - `in_memory` (bool): Whether the raster is in memory or not.
    - `driver` (str): Driver of the raster.
    - `projection_osr` (osr.SpatialReference): Projection of the raster as an osr.SpatialReference object.
    - `projection_wkt` (str): Projection of the raster as a WKT string.
    - `geotransform` (tuple): Geotransform of the raster.
    - `size` (tuple): Size of the raster in pixels.
    - `shape` (list): Shape of the raster in pixels. (height, width, bands)
    - `height` (int): Height of the raster in pixels.
    - `width` (int): Width of the raster in pixels.
    - `pixel_size` (tuple): Pixel size of the raster in units of the projection.
    - `pixel_width` (float): Pixel width of the raster in units of the projection.
    - `pixel_height` (float): Pixel height of the raster in units of the projection.
    - `origin` (tuple): Origin of the raster in units of the projection.
    - `origin_x` (float): Origin x of the raster in units of the projection.
    - `origin_y` (float): Origin y of the raster in units of the projection.
    - `bbox` (list): Bounding box of the raster in units of the projection. `[xmin, xmax, ymin, ymax]`
    - `bbox_gdal` (list): Bounding box of the raster in GDAL format. `[xmin, ymin, xmax, ymax]`
    - `bbox_latlng` (list): Bounding box of the raster in latlng format. `[ymin, xmin, ymax, xmax]`
    - `bounds_latlng` (str): Bounding box of the raster in latlng format as a WKT string.
    - `x_min` (float): Minimum x of the raster in units of the projection.
    - `x_max` (float): Maximum x of the raster in units of the projection.
    - `y_min` (float): Minimum y of the raster in units of the projection.
    - `y_max` (float): Maximum y of the raster in units of the projection.
    - `bands` (int): Number of bands in the raster.
    - `dtype_gdal` (int): GDAL data type of the raster.
    - `dtype` (numpy.dtype): Numpy data type of the raster.
    - `dtype_name` (str): Name of the numpy data type of the raster.
    - `area_latlng` (float): Area of the raster in latlng units.
    - `area` (float): Area of the raster in units of the projection.

    Parameters
    ----------
    raster : str or gdal.Dataset or list
        A path to a raster or a gdal.Dataset, or a list of paths to rasters.

    Returns
    -------
    dict or list of dict
        A dictionary or a list of dictionaries containing metadata information for each raster.
    """
    utils_base._type_check(raster, [str, gdal.Dataset, [str, gdal.Dataset]], "raster")

    input_is_list = isinstance(raster, list)
    input_rasters = utils_io._get_input_paths(raster, "raster")

    metadata = []
    for in_raster in input_rasters:
        metadata.append(_raster_to_metadata(in_raster))

    if input_is_list:
        return metadata

    return metadata[0]
