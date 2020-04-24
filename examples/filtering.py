import sys; sys.path.append('..')
import numpy as np
from time import time
import cv2
from lib.raster_io import raster_to_array, array_to_raster
from lib.stats_filters import threshold_filter, median_filter, median_deviation_filter

folder = '/mnt/c/Users/caspe/Desktop/Analysis/Data/mosaic/aligned/'
b4_path = folder + 'B04.tif'
b8_path = folder + 'B08.tif'
bs_path = folder + 'BS.tif'
blur_path = folder + 'bs_blur.tif'

B4 = raster_to_array(b4_path)
B8 = raster_to_array(b8_path)

# bs_blur = cv2.GaussianBlur(raster_to_array(bs_path), (11, 11), 0)
# array_to_raster(np.sqrt(raster_to_array(blur_path)).astype('float32'), out_raster=folder + 'bs_blur_sqrt.tif', reference_raster=b4_path)

array_to_raster((B4 - B8) / (B4 + B8), out_raster=folder + 'inv_ndvi.tif', reference_raster=b4_path)
