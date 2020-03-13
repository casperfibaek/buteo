import sys; sys.path.append('..')
import numpy as np
from time import time
from lib.raster_io import raster_to_array, array_to_raster
from lib.stats_filters import median_filter, mean_filter

# Merging aster and srtm to remove slope noise in srtm.
folder = '/mnt/c/users/caspe/desktop/data/DEM/'

srtm = folder + 'Ghana_dem_utm30_wgs84_srtm_clipped_align.tif'
aster = folder + 'Ghana_dem_utm30_wgs84_aster_clipped_align.tif'
srtm_arr = raster_to_array(srtm)
aster_arr = raster_to_array(aster)

merge = np.ma.array([srtm_arr, aster_arr], fill_value=srtm_arr.fill_value)

# import pdb; pdb.set_trace()

before = time()
array_to_raster(mean_filter(median_filter(merge, 11, iterations=2), 11, iterations=3), reference_raster=srtm, out_raster=folder + 'ghana_smooth_dem.tif')
print(time() - before)