import sys; sys.path.append('..')
from lib.raster_io import raster_to_array, array_to_raster
from lib.stats_kernel import create_kernel
from lib.stats_filters import fast_sum_filter
import numpy as np
import scipy.signal


folder = folder = '/mnt/c/users/caspe/desktop/Analysis/data/clipped/tests/'
img = folder + 's1_wet_mask_cropped.tif'
arr = raster_to_array(img).astype('float32')

kernel = create_kernel(5, distance_calc='power')

# bob = scipy.signal.fftconvolve(arr, kernel, mode='same')
carl = fast_sum_filter(arr, _kernel=kernel)

array_to_raster(carl, reference_raster=img, out_raster=folder + 's1_wet_mask_cropped_yellow_01.tif')
