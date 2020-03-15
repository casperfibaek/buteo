import sys; sys.path.append('..')
import numpy as np
from time import time
from lib.raster_io import raster_to_array, array_to_raster
from lib.stats_filters import median_filter, mean_filter, threshold_array, truncate_filter, sum_filter

folder = '/mnt/c/users/caspe/desktop/Analysis/data/clipped/'

dry_perm2_mask = raster_to_array(folder + 's1_dry_perm2_mask.tif')
dry_perm2_mask.fill_value = 0
dry_perm2_mask = dry_perm2_mask.filled()

dry_perm2_trunc = raster_to_array(folder + 's1_dry_perm2_truncated.tif')
dry_perm2_trunc.fill_value = 0
dry_perm2_trunc = dry_perm2_trunc.filled()

array_to_raster(sum_filter(dry_perm2_mask, 201), reference_raster=folder + 's1_dry_perm2.tif', out_raster=folder + 's1_dry_perm2_mask-density.tif')
array_to_raster(sum_filter(dry_perm2_trunc, 201), reference_raster=folder + 's1_dry_perm2.tif', out_raster=folder + 's1_dry_perm2_truncated-density.tif')


# dry_perm2 = raster_to_array(folder + 's1_dry_perm2.tif')
# wet_perm2 = raster_to_array(folder + 's1_wet_perm2.tif')
# array_to_raster(threshold_array(raster_to_array(folder + 's1_dry_perm2.tif'), min_value=0.2), reference_raster=folder + 's1_dry_perm2.tif', out_raster=folder + 's1_dry_perm2_mask.tif')
# array_to_raster(threshold_array(raster_to_array(folder + 's1_wet_perm2.tif'), min_value=0.2), reference_raster=folder + 's1_wet_perm2.tif', out_raster=folder + 's1_wet_perm2_mask.tif')

# dry_truncate = truncate_filter(dry_perm2, min_value=0, max_value=1)
# wet_truncate = truncate_filter(wet_perm2, min_value=0, max_value=1)

# import pdb; pdb.set_trace()

# array_to_raster(dry_truncate, reference_raster=folder + 's1_dry_perm2.tif', out_raster=folder + 's1_dry_perm2_truncated.tif')
# array_to_raster(wet_truncate, reference_raster=folder + 's1_wet_perm2.tif', out_raster=folder + 's1_wet_perm2_truncated.tif')

# array_to_raster(sum_filter(raster_to_array(folder + 's1_dry_perm2_mask.tif'), 201), reference_raster=folder + 's1_dry_perm2.tif', out_raster=folder + 's1_dry_perm2_mask-density.tif')
# array_to_raster(sum_filter(raster_to_array(folder + 's1_dry_perm2_truncated.tif'), 201), reference_raster=folder + 's1_dry_perm2.tif', out_raster=folder + 's1_dry_perm2_truncated-density.tif')

# dry_coh2 = folder + 's1_dry_coh_pow2.tif'
# dry_coh2_arr = raster_to_array(dry_coh2)
# dry_sigma0 = folder + 's1_dry_sigma0.tif'
# dry_sigma0_arr = raster_to_array(dry_sigma0)

# wet_coh2 = folder + 's1_wet_coh_pow2.tif'
# wet_coh2_arr = raster_to_array(wet_coh2)
# wet_sigma0 = folder + 's1_wet_sigma0.tif'
# wet_sigma0_arr = raster_to_array(wet_sigma0)

# array_to_raster(dry_coh2_arr * np.ma.power(dry_sigma0_arr, 2), reference_raster=dry_coh2, out_raster=folder + 's1_dry_perm2.tif')
# array_to_raster(wet_coh2_arr * np.ma.power(wet_sigma0_arr, 2), reference_raster=wet_coh2, out_raster=folder + 's1_wet_perm2.tif')

# coh_jan = folder + 's1_dry_coh.tif'
# coh_june = folder + 's1_2019-06-15_coh.tif'

# coh_read_jan = raster_to_array(coh_jan)
# coh_read_june = raster_to_array(coh_june)
# coh_jan_arr = mean_filter(median_filter(coh_read_jan, 7), 5)
# coh_june_arr = mean_filter(median_filter(coh_read_june, 7), 5)
# coh_jan_arr_power = np.ma.power(coh_jan_arr, 2)
# coh_june_arr_power = np.ma.power(coh_june_arr, 2)

# array_to_raster(coh_june_arr_power, reference_raster=coh_june, out_raster=folder + 's1_2019-06-15_coh_preprocessed.tif')
# array_to_raster(coh_jan_arr_power, reference_raster=coh_jan, out_raster=folder + 's1_dry_coh_pow2.tif')

# jan = 's1_january_sigma0.tif'
# june = 's1_june_sigma0.tif'

# jan_arr = raster_to_array(folder + jan)
# june_arr = raster_to_array(folder + june)

# jan_dB = 10 * np.log10(np.abs(jan_arr))
# june_dB = 10 * np.log10(np.abs(june_arr))

# array_to_raster(jan_dB, reference_raster=folder + jan, out_raster=folder + 's1_january_sigma0_dB.tif')
# array_to_raster(june_dB, reference_raster=folder + june, out_raster=folder + 's1_june_sigma0_dB.tif')


# sigma0_16jan2019 = 's1_2019-01-16_sigma0.tif'
# sigma0_22jan2019 = 's1_2019-01-22_sigma0.tif'
# sigma0_09jun2019 = 's1_2019-06-09_sigma0.tif'
# sigma0_21jun2019 = 's1_2019-06-21_sigma0.tif'

# sigma0_22jan2019_arr = median_filter(raster_to_array(folder + sigma0_22jan2019))
# sigma0_16jan2019_arr = median_filter(raster_to_array(folder + sigma0_16jan2019))
# merged_jan2019 = np.ma.masked_equal([sigma0_16jan2019_arr, sigma0_22jan2019_arr], sigma0_22jan2019_arr.fill_value)

# sigma0_09jun2019_arr = raster_to_array(folder + sigma0_09jun2019)
# sigma0_21jun2019_arr = raster_to_array(folder + sigma0_21jun2019)
# merged_jun2019 = np.ma.masked_equal([sigma0_09jun2019_arr, sigma0_21jun2019_arr], sigma0_21jun2019_arr.fill_value)

# array_to_raster(median_filter(mean_filter(merged_jan2019, sigma=1)), reference_raster=folder + sigma0_16jan2019, out_raster=folder + 's1_january_sigma0.tif')
# array_to_raster(median_filter(mean_filter(merged_jun2019, sigma=1)), reference_raster=folder + sigma0_09jun2019, out_raster=folder + 's1_june_sigma0.tif')
