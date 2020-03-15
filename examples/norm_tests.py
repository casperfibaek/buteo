import sys; sys.path.append('..')
import numpy as np

from lib.raster_io import raster_to_array, array_to_raster
from lib.stats_filters import mean_filter, median_filter, standard_deviation_filter, cdef_filter

folder = '/mnt/c/users/caspe/desktop/data/satf_preprocess/'
b16_path = folder + '16-02-2019_crop.tif'
b22_path = folder + '22-02-2019_crop.tif'

b16 = raster_to_array(b16_path)
b22 = raster_to_array(b22_path)

maxz = raster_to_array(folder + '__max_z.tif')
array_to_raster(cdef_filter(maxz), reference_raster=b16_path, out_raster=folder + '__max_z_cdf.tif')

# b16_mean = mean_filter(b16, 15)
# b16_median = median_filter(b16, 15)
# b22_mean = mean_filter(b22, 15)
# b22_median = median_filter(b22, 15)
# b16_std = standard_deviation_filter(b16, 15)
# b22_std = standard_deviation_filter(b22, 15)
# b16_22_median_difference = np.abs(b16_median - b22_median)
# b16_mean_after_difference = b16_mean - (b16_22_median_difference - b16_mean)
# b22_mean_after_difference = b22_mean - (b16_22_median_difference - b22_mean)

# with np.errstate(divide='ignore', invalid='ignore'):
#     b16_zscore = np.abs((b16_mean - b16_mean_after_difference) / b16_std)
#     b16_zscore[b16_std == 0] = 0
    
#     b22_zscore = np.abs((b22_mean - b22_mean_after_difference) / b22_std)
#     b22_zscore[b22_std == 0] = 0
        
# b16_cdf = cdef_filter(b16_zscore)
# b22_cdf = cdef_filter(b22_zscore)

# min_score = np.min([b16_cdf, b22_cdf], axis=0)


# array_to_raster(b16_mean, reference_raster=b16_path, out_raster=folder + 'B16_mean.tif')
# array_to_raster(b16_median, reference_raster=b16_path, out_raster=folder + 'B16_median.tif')
# array_to_raster(b22_mean, reference_raster=b16_path, out_raster=folder + 'B22_mean.tif')
# array_to_raster(b22_median, reference_raster=b16_path, out_raster=folder + 'B22_median.tif')
# array_to_raster(b16_22_median_difference, reference_raster=b16_path, out_raster=folder + 'B16-22_median-difference.tif')
# array_to_raster(b16_mean_after_difference, reference_raster=b16_path, out_raster=folder + 'B16_mean_after_difference.tif')
# array_to_raster(b22_mean_after_difference, reference_raster=b16_path, out_raster=folder + 'B22_mean_after-difference.tif')
# array_to_raster(b16_std, reference_raster=b16_path, out_raster=folder + 'b16_std.tif')
# array_to_raster(b22_std, reference_raster=b16_path, out_raster=folder + 'b22_std.tif')
# array_to_raster(b16_zscore, reference_raster=b16_path, out_raster=folder + 'b16_zscore.tif')
# array_to_raster(b22_zscore, reference_raster=b16_path, out_raster=folder + 'b22_zscore.tif')
# array_to_raster(b16_cdf, reference_raster=b16_path, out_raster=folder + 'b16_cdf.tif')
# array_to_raster(b22_cdf, reference_raster=b16_path, out_raster=folder + 'b22_cdf.tif')
# array_to_raster(min_score, reference_raster=b16_path, out_raster=folder + 'min_score.tif')

