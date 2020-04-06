import sys; sys.path.append('..')
import numpy as np
from time import time
from lib.raster_io import raster_to_array, array_to_raster
from lib.stats_filters import threshold_filter, median_filter



folder = '/mnt/c/users/caspe/desktop/tests/'
in_path = folder + 'B02_30NVL.tif'
in_raster = raster_to_array(in_path).astype(np.double)

# array_to_raster(standardise_filter(in_raster).astype('float32'), out_raster=folder + 'b4_standard.tif', reference_raster=in_path)
# array_to_raster(standardise_filter(in_raster, True).astype('float32'), out_raster=folder + 'b4_standard_scaled.tif', reference_raster=in_path)
# array_to_raster(median_deviation_filter(in_raster, 5, absolute_value=True).astype('float32'), out_raster=folder + 'b4_meddev-holed.tif', reference_raster=in_path)
# array_to_raster(median_deviation_filter(in_raster, 5, absolute_value=True, holed=False).astype('float32'), out_raster=folder + 'b4_meddev-no-holed.tif', reference_raster=in_path)
# array_to_raster(mean_deviation_filter(in_raster, 5, absolute_value=True).astype('float32'), out_raster=folder + 'b4_meandev-holed.tif', reference_raster=in_path)
# array_to_raster(mean_deviation_filter(in_raster, 5, absolute_value=True, holed=False).astype('float32'), out_raster=folder + 'b4_meandev-no-holed.tif', reference_raster=in_path)
# array_to_raster(threshold_filter(in_raster, min_value=200, max_value=600), out_raster=folder + 'b4_threshold.tif', reference_raster=in_path)
# array_to_raster(threshold_filter(in_raster, min_value=200, max_value=600, invert=True), out_raster=folder + 'b4_threshold_inverted.tif', reference_raster=in_path)

array_to_raster(median_filter(in_raster, sigma=1, dtype='uint16'), reference_raster=in_path, out_raster=folder + 'B02_30NVL_filtered_sigma1.tif')