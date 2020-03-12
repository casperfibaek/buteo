import sys; sys.path.append('..')
import numpy as np
from time import time
from lib.raster_io import raster_to_array, array_to_raster
from lib.stats_filters import median_filter, mean_filter

# Merging aster and srtm to remove slope noise in srtm.
folder = '/mnt/c/users/caspe/desktop/data/DEM/'

srtm = folder + 'Ghana_dem_utm30_wgs84_srtm_clipped_align.tif'
aster = folder + 'Ghana_dem_utm30_wgs84_aster_clipped_align.tif'
srtm_arr = raster_to_array(srtm, src_nodata=-32768)
aster_arr = raster_to_array(aster, src_nodata=-32768)

merge = np.ma.array([srtm_arr, aster_arr], fill_value=-32768)

# import pdb; pdb.set_trace()

before = time()
array_to_raster(median_filter(merge, 3, sigma=1), reference_raster=srtm, out_raster=folder + 'ghana_DEM_merge.tif')
print(time() - before)