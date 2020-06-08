import sys; sys.path.append('..'); sys.path.append('../lib/')
from stats_zonal import calc_zonal, calc_shapes
from rasterstats import zonal_stats

raster_hot = "/home/cfi/data/hot.vrt"
raster_vol = "/home/cfi/data/vol.vrt"
vector = "/home/cfi/data/studyAreaBuildings.gpkg"

# calc_shapes(vector)
# calc_zonal(vector, [raster_hot], prefixes=["hot_"], stats=["mean"])
calc_zonal(vector, [raster_vol], prefixes=["vol_"], stats=["mean"])
