import sys; sys.path.append('..')
import numpy as np
from glob import glob
from time import time
from pyproj import CRS
from lib.raster_clip import clip_raster
from lib.raster_reproject import reproject
from lib.raster_io import raster_to_array, array_to_raster
from lib.stats_filters import threshold_filter, median_filter, median_deviation_filter

clip_to = '/mnt/c/Users/caspe/Desktop/Analysis/Data/vector/ghana_5km_buffer_wgs84.shp'
# reference = 
src_folder = '/mnt/c/users/caspe/Desktop/Analysis/Data/nightlights/'
dst_folder = '/mnt/c/users/caspe/Desktop/Analysis/Data/nightlights/clipped/'

# files = glob(src_folder + '*.tif')

# for f in files:
#     clip_raster(f, out_raster=dst_folder + f.rsplit('/', 1)[1], cutline=clip_to, cutline_all_touch=True, crop_to_cutline=True)

files = glob(dst_folder + '*cf_cvg*.tif')
base = raster_to_array(files[0])
for i, f in enumerate(files):
    if i == 0: continue
    base = np.add(base, raster_to_array(f))

array_to_raster(base, out_raster=dst_folder + 'count.tif', reference_raster=files[0])

count = raster_to_array(dst_folder + 'count.tif')

files = glob(dst_folder + '*avg_rade9h*.tif')
base = files[0]
base_count = files[0].rsplit('.', 2)[0] + '.cf_cvg.tif'
base_weight = raster_to_array(base_count) / count
base = raster_to_array(base) * base_weight

# # import pdb; pdb.set_trace()

for i, f in enumerate(files):
    if i == 0: continue
    f_count = f.rsplit('.', 2)[0] + '.cf_cvg.tif'
    f_weight = raster_to_array(f_count) / count
    base = np.add(base, (raster_to_array(f) * f_weight))

array_to_raster(base, reference_raster=files[0], out_raster=dst_folder + 'avg_lights_5-9.tif')
    