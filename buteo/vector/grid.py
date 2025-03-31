"""Create vector grids from references or hardcode"""

# Standard library
from typing import Union, Optional, List

# External
from osgeo import ogr

# Internal
from buteo.utils import (
    utils_io,
    utils_base,
    utils_gdal,
    utils_path,
)
from buteo.core_vector.core_vector_read import open_vector as vector_open
from buteo.core_vector.core_vector_info import get_metadata_vector

# TODO: Implement grid functions
# TODO: create_grid, create_grid_points, create_grid_linestrings



def create_hexagonal_grid(extent, projection, cell_size, out_path=None):
    """Create a hexagonal grid in a given extent and projection.

    Parameters
    ----------
    extent : list
        [xmin, ymin, xmax, ymax]
    projection : str
        EPSG code or projection string
    cell_size : float
        Size of the hexagon
    out_path : str, optional
        Path to save the grid, by default None

    Returns
    -------
    str
        Path to the created grid
    """
    # Not implemented yet
    # vector = vector_open(extent)

    # vector.ExecuteSQL("CREATE TABLE hex_grid (id INTEGER PRIMARY KEY, geom GEOMETRY)")
    # vector.ExecuteSQL("SELECT AddGeometryColumn('hex_grid', 'geom', 4326, 'POLYGON', 2)")
    # vector.ExecuteSQL("CREATE INDEX hex_grid_geom_idx ON hex_grid USING GIST (geom)")
    # vector.ExecuteSQL("SELECT ST_Hex
