import sys; sys.path.append('..'); sys.path.append('../lib/')
from lib.raster_io import raster_to_array, array_to_raster
from lib.raster_clip import clip_raster
from lib.raster_nodata import remove_nodata
from lib.stats_filters import mad_filter, median_deviation_filter, standard_deviation_filter, median_filter, truncate_filter, scale_to_range_filder
from glob import glob
import os
import numpy as np

# folder = "/mnt/c/Users/caspe/Desktop/Paper_2_StruturalDensity/aligned/"
# b4_path = folder + "s2_b04.tif"
# b8_path = folder + "s2_b08.tif"
# coh_path = folder + "s1_coherence_median.tif"

# array_to_raster(mad_filter(raster_to_array(b4_path), 5, holed=True), folder + "s2_b04_madh_5.tif", b4_path)
# array_to_raster(mad_filter(raster_to_array(b8_path), 5, holed=True), folder + "s2_b08_madh_5.tif", b8_path)
# # array_to_raster(median_filter(raster_to_array(coh_path), 3, distance_calc=False), folder + "s1_coherence_median_med_3.tif", coh_path)


# # b8_path = folder + "b08_s2_mosaic.tif"
# # b4_path = folder + "b04_s2_mosaic.tif"

# # b8 = raster_to_array(b8_path)
# # b4 = raster_to_array(b4_path)

# # a = b8 - b4
# # b = b8 + b4

# # d = np.divide(a, b, out=np.zeros_like(a, dtype="float"), where=b != 0)

# # # ndvi = truncate_filter(d, min_value=-1, max_value=1)

# # # ndvi_inv = ((ndvi * (-1)) + 1) / 2

# # # array_to_raster(d.astype('float32'), folder + "ndvi_inv_scaled_s2_mosaic_05.tif", b8_path)
# # array_to_raster((a).astype('uint8'), folder + "ndvi_inv_scaled_s2_mosaic_a.tif", b8_path)
# # array_to_raster((b).astype('uint8'), folder + "ndvi_inv_scaled_s2_mosaic_b.tif", b8_path)

# folder = "/mnt/c/Users/caspe/Desktop/Paper_2_StruturalDensity/"
# images = glob(folder + "aligned/*.tif")

# for img in images:
#     clip_raster(
#         img,
#         folder + 'silkeborg/clip_' + os.path.basename(img),
#         cutline=folder + 'silkeborg/silkeborg_municipality.gpkg',
#         cutline_all_touch=True,
#     )

# Rescale imagery
folder = "/mnt/c/Users/caspe/Desktop/Paper_2_StruturalDensity/silkeborg/"
path_bsc = folder + "clip_s1_bs_x_coh.tif"
path_msavi2 = folder + "clip_s2_msavi2.tif"
path_b8 = folder + "clip_s2_b08.tif"
path_b4 = folder + "clip_s2_b04.tif"

bsc = raster_to_array(path_bsc)
bsc = np.rint(scale_to_range_filder(remove_nodata(truncate_filter(bsc, 0, 1)), 0, 255)).astype('uint8')
array_to_raster(bsc, folder + "8bit/bsc_8bit.tif", path_bsc)

msavi2 = raster_to_array(path_msavi2)
msavi2 = np.rint(scale_to_range_filder(remove_nodata(truncate_filter(msavi2, -1, 1)), 0, 255)).astype('uint8')
array_to_raster(msavi2, folder + "8bit/msavi2_8bit.tif", path_msavi2)

b8 = raster_to_array(path_b8)
b8 = np.rint(scale_to_range_filder(remove_nodata(truncate_filter(b8, 0, 5000)), 0, 255)).astype('uint8')
array_to_raster(b8, folder + "8bit/b8_8bit.tif", path_b8)

b4 = raster_to_array(path_b4)
b4 = np.rint(scale_to_range_filder(remove_nodata(truncate_filter(b4, 0, 3000)), 0, 255)).astype('uint8')
array_to_raster(b4, folder + "8bit/b4_8bit.tif", path_b4)

import pdb; pdb.set_trace()



