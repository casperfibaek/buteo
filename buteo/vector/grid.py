"""
Create vector grids from references or hardcode
"""
# Standard library
import sys; sys.path.append("../../")
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
from buteo.vector import core_vector
from buteo.vector.metadata import _vector_to_metadata


# TODO: Implement grid functions
# TODO: create_grid, create_grid_points, create_grid_linestrings


def create_hexagonal_grid(extend, projection, cell_size, out_path=None):
    """Create a hexagonal grid in a given extend and projection.

    Args:
        extend (list): [xmin, ymin, xmax, ymax]
        projection (str): EPSG code
        cell_size (float): size of the hexagon
        out_path (str): path to save the grid

    Returns:
        None
    """
    # vector = core_vector._vector_open(extend)

    # vector.ExecuteSQL("CREATE TABLE hex_grid (id INTEGER PRIMARY KEY, geom GEOMETRY)")
    # vector.ExecuteSQL("SELECT AddGeometryColumn('hex_grid', 'geom', 4326, 'POLYGON', 2)")
    # vector.ExecuteSQL("CREATE INDEX hex_grid_geom_idx ON hex_grid USING GIST (geom)")
    # vector.ExecuteSQL("SELECT ST_Hex