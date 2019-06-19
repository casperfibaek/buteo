import sys
import geopandas as gp
from geopandas.tools import sjoin
sys.path.append('../lib')

from utils import divide_steps


def intersecting_tile(shape):
    world_tiles = gp.GeoDataFrame.from_file('../geometry/sentinel2_tiles_world.shp')
    world_tiles.crs = {'init': 'epsg:4326', 'no_defs': True}
    test_shape = gp.GeoDataFrame.from_file(shape).to_crs(world_tiles.crs)

    return gp.sjoin(world_tiles, test_shape, how="inner", op='intersects')['Name'].values
