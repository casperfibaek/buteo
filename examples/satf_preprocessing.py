import sys; sys.path.append('..')
import numpy as np
from time import time
from lib.raster_io import raster_to_array, array_to_raster
from lib.stats_filters import median_filter

folder = '/mnt/c/users/caspe/desktop/data/satf_preprocess/'
sigma0_22jan2019 = 'Sigma0_VV_mst_22Jan2019.tif'
sigma0_16jan2019 = 'Sigma0_VV_slv2_16Jan2019.tif'
sigma0_09jun2019 = 'Sigma0_VV_slv3_09Jun2019.tif'
sigma0_21jun2019 = 'Sigma0_VV_slv4_21Jun2019.tif'

# sigma0_22jan2019_arr = median_filter(raster_to_array(folder + sigma0_22jan2019))
# sigma0_16jan2019_arr = median_filter(raster_to_array(folder + sigma0_16jan2019))
# merged_jan2019 = np.mean(np.array([sigma0_16jan2019_arr, sigma0_22jan2019_arr]), axis=0)

# sigma0_09jun2019_arr = raster_to_array(folder + sigma0_09jun2019)
# sigma0_21jun2019_arr = raster_to_array(folder + sigma0_21jun2019)
# merged_jun2019 = np.mean(np.array([sigma0_09jun2019_arr, sigma0_21jun2019_arr]), axis=0)

from scipy.stats import norm

def normalise_filter_2(in_raster):
    mi = np.nanmin(in_raster)
    return (in_raster + np.abs(mi))

# def normalise_filter(in_raster):
#     mi = np.nanmin(in_raster)
#     ma = np.nanmax(in_raster)
#     return ((in_raster - mi) / (ma - mi)).astype(in_raster.dtype)

a = np.array([[ 5.,  3.,  2.], [ 1., -10.,-1.], [-2., -3., -4.]])
b = np.array([[-5., -3., -2.], [-1.,  10.,  1.], [ 2.,  3.,  4.]])
c = np.array([a, b])
k = np.array([[.3,  .7,  .3 ], [.7,  .9,  .7 ], [.3,  .7,  .3]])
# a = b


import pdb; pdb.set_trace()
# diff = (a / a.sum()) - (b / b.sum())
# m_norm = np.sum(np.abs(diff))
# z_norm = norm(np.ravel(diff), 0)


# before = time()

# array_to_raster(median_filter(merged_jan2019, 3, sigma=2, iterations=2), out_raster=folder + 'jan2019_merge.tif', reference_raster=folder + sigma0_22jan2019)
# array_to_raster(median_filter(merged_jun2019, 3, sigma=2, iterations=2), out_raster=folder + 'jun2019_merge.tif', reference_raster=folder + sigma0_09jun2019)

# print(time() - before)
