"""
### Functions to turn rasters into rasters of coordinates. ###
"""

# Standard library
import sys; sys.path.append("../../")
from typing import Union, Tuple

# External
import numpy as np
from osgeo import gdal

# Internal
from buteo.utils import utils_base
from buteo.raster import core_raster


# TODO: Define how this is going to work.
def raster_create_grid_with_coordinates(
    raster: Union[str, gdal.Dataset],
    latlng: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a grid of coordinates from a raster. Format is (x, y, xy).

    Parameters
    ----------
    raster : str or gdal.Dataset
        The raster to create the grid from.

    Returns
    -------
    np.ndarray
        A NumPy array of shape (x, y, xy-coordinates).
    """
    utils_base._type_check(raster, [str, gdal.Dataset], "raster")

    meta = core_raster._get_basic_metadata_raster(raster)

    step_x = meta["pixel_width"]
    size_x = meta["width"]
    start_x = meta["x_min"]
    stop_x = meta["x_max"]

    step_y = -meta["pixel_height"]
    size_y = meta["height"]
    start_y = meta["y_max"]
    stop_y = meta["y_min"]

    x_adj = step_x / 2
    y_adj = step_y / 2

    x_vals = np.linspace(start_x + x_adj, stop_x - x_adj, size_x, dtype=np.float32)
    y_vals = np.linspace(start_y - y_adj, stop_y + y_adj, size_y, dtype=np.float32)

    xx, yy = np.meshgrid(x_vals, y_vals)
    grid = np.dstack((xx, yy))

    return grid
