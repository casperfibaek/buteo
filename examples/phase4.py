import sys; sys.path.append('..')
import numpy as np
import geopandas as gpd
from time import time
from glob import glob
from pathlib import Path
from lib.raster_io import raster_to_array, array_to_raster
from lib.stats_filters import fast_sum_filter, highest_filter

phase4_folder = '/mnt/c/users/caspe/desktop/Analysis/'

ref_path = phase4_folder + 'Data/s2_b04_10m_dry.tif'
ref = raster_to_array(ref_path)

# path = phase4_folder + 'Phase4/urban_raw_masked.tif'
# arr = raster_to_array(path).astype('float32')
# arr.fill_value = 0
# arr = arr.filled()
# dens = fast_sum_filter(arr, 201, weighted_distance=True, distance_calc='power')
# dens = np.ma.masked_where(ref.mask, dens)
# out = phase4_folder + 'Phase4/urban_raw_density.tif'
# array_to_raster(dens, reference_raster=ref_path, out_raster=out)


# path_dens = phase4_folder + 'Phase4/suburban_density.tif'
# path_prox = phase4_folder + 'Phase4/suburban_proximity.tif'

# dens_arr = np.ma.add(raster_to_array(path_dens), 1)
# prox_arr = np.ma.add(raster_to_array(path_prox), 1)

# merge = dens_arr / np.sqrt(prox_arr)
# merge = np.ma.masked_where(ref.mask, merge)

# out = phase4_folder + 'Phase4/suburban_merge_1km.tif'
# array_to_raster(merge, reference_raster=ref_path, out_raster=out)

# path_dens = phase4_folder + 'Phase4/urban_density.tif'
# path_prox = phase4_folder + 'Phase4/urban_proximity.tif'

# dens_arr = np.ma.add(raster_to_array(path_dens), 1)
# prox_arr = np.ma.add(raster_to_array(path_prox), 1)

# merge = dens_arr / np.sqrt(prox_arr)
# merge = np.ma.masked_where(ref.mask, merge)

# out = phase4_folder + 'Phase4/urban_merge_1km.tif'
# array_to_raster(merge, reference_raster=ref_path, out_raster=out)

# dense_urban = raster_to_array(phase4_folder + 'Phase4/dens_urban_merge_1km.tif')
# urban = raster_to_array(phase4_folder + 'Phase4/urban_merge_1km.tif')
# suburban = raster_to_array(phase4_folder + 'Phase4/suburban_merge_1km.tif')
# rural = raster_to_array(phase4_folder + 'Phase4/rural_outskirts_merge_1km.tif')
# hinterlands = raster_to_array(phase4_folder + 'Phase4/hinterlands_proximity_merge_1km.tif')

weights = np.array([1.0, 0.9, 0.8, 0.7, 0.5], dtype=np.double)
merged = np.ma.array([
    raster_to_array(phase4_folder + 'Phase4/dense_urban_density.tif'),
    raster_to_array(phase4_folder + 'Phase4/urban_density.tif'),
    raster_to_array(phase4_folder + 'Phase4/suburban_density.tif'),
    raster_to_array(phase4_folder + 'Phase4/rural_outskirts_density.tif'),
    raster_to_array(phase4_folder + 'Phase4/hinterlands_density.tif'),
])

highest = highest_filter(merged, weights)
array_to_raster(highest, reference_raster=ref_path, out_raster=phase4_folder + 'Phase4/merged_06.tif')
