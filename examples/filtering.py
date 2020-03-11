import sys; sys.path.append('..')
import numpy as np
from time import time
from lib.raster_io import raster_to_array, array_to_raster
from lib.stats_filters import threshold_filter

folder = '/mnt/c/users/caspe/desktop/data/'
in_path = folder + 'B04.jp2'
in_raster = raster_to_array(in_path).astype(np.double)

# array_to_raster(standardise_filter(in_raster).astype('float32'), out_raster=folder + 'b4_standard.tif', reference_raster=in_path)
# array_to_raster(standardise_filter(in_raster, True).astype('float32'), out_raster=folder + 'b4_standard_scaled.tif', reference_raster=in_path)
# array_to_raster(median_deviation_filter(in_raster, 5, absolute_value=True).astype('float32'), out_raster=folder + 'b4_meddev-holed.tif', reference_raster=in_path)
# array_to_raster(median_deviation_filter(in_raster, 5, absolute_value=True, holed=False).astype('float32'), out_raster=folder + 'b4_meddev-no-holed.tif', reference_raster=in_path)
# array_to_raster(mean_deviation_filter(in_raster, 5, absolute_value=True).astype('float32'), out_raster=folder + 'b4_meandev-holed.tif', reference_raster=in_path)
# array_to_raster(mean_deviation_filter(in_raster, 5, absolute_value=True, holed=False).astype('float32'), out_raster=folder + 'b4_meandev-no-holed.tif', reference_raster=in_path)
array_to_raster(threshold_filter(in_raster, min_value=200, max_value=600), out_raster=folder + 'b4_threshold.tif', reference_raster=in_path)
array_to_raster(threshold_filter(in_raster, min_value=200, max_value=600, invert=True), out_raster=folder + 'b4_threshold_inverted.tif', reference_raster=in_path)
