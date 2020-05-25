import sys; sys.path.append('..')
import numpy as np
from time import time
import cv2
from lib.raster_io import raster_to_array, array_to_raster
from lib.stats_filters import standardise_filter
from lib.raster_merge import merge
from lib.stats_filters import threshold_filter, median_filter, median_deviation_filter, mad_filter, mode_filter

# folder = 'C:/Users/Caspe/Desktop/Ghana_data/'
folder = 'C:/Users/caspe/Desktop/analysis_p2/classification/'
c_path = folder + 'classification_01.tif'


# b8_path = folder + 'B08.tif'
# b8_tex_path = folder + 'B08_MAD.tif'
# bs_path = folder + 'BS.tif'
# coh_path = folder + 'COH.tif'
# coh_x_bs_path = folder + 'COH_x_BS.tif'
# coh_x_bs_blur_path = folder + 'COH_x_BS_blur.tif'
# ndvi_path = folder + 'NDVI.tif'
# nl_path = folder + 'NL.tif'
# b4_tex = folder + 'B04_MAD.tif'

# indvi_path = folder + 'inv_ndvi_s.tif'
# nl_path = folder + 'NL_norm.tif'
# ndvi_path = folder + 'NDVI.tif'

# bs = raster_to_array(bs_path)
# coh = raster_to_array(coh_path)
# indvi = raster_to_array(indvi_path)
# nl = raster_to_array(nl_path)

# ndvi = raster_to_array(ndvi_path)
# nl = raster_to_array(nl_path)
# ndvi = raster_to_array(indvi_path)
# coh = raster_to_array(coh_path)
# bs = raster_to_array(bs_path)
# out_folder = folder + 'normalized/'



array_to_raster(mode_filter(raster_to_array(c_path), width=201), reference_raster=c_path, out_raster=folder + 'classification_01_1km_majority_v2.tif')
# array_to_raster(mad_filter(raster_to_array(b8_path), width=7), reference_raster=b8_path, out_raster=folder + 'B08_MAD.tif')
# array_to_raster(np.abs(raster_to_array(folder + 'B04_md7.tif')), reference_raster=b4_path, out_raster=folder + 'B04_md7_abs.tif')

# array_to_raster(cv2.GaussianBlur(raster_to_array(bs_coh_path), (401, 401), 2), reference_raster=bs_coh_path, out_raster=folder + 'coh_x_bs_2km_guassian_blur.tif')

# array_to_raster(raster_to_array(bs_path) * raster_to_array(coh_path), reference_raster=bs_path, out_raster=folder + 'coh_x_bs.tif')


# array_to_raster(standardise_filter(raster_to_array(bs_path)).astype('float32'), reference_raster=bs_path, out_raster=out_folder + 'BS_norm.tif', dst_nodata=-999.9)
# array_to_raster(standardise_filter(raster_to_array(coh_path)).astype('float32'), reference_raster=coh_path, out_raster=out_folder + 'COH_norm.tif', dst_nodata=-999.9)
# array_to_raster(standardise_filter(raster_to_array(coh_x_bs_path)).astype('float32'), reference_raster=coh_x_bs_path, out_raster=out_folder + 'COH_x_BS_norm.tif', dst_nodata=-999.9)
# array_to_raster(standardise_filter(raster_to_array(coh_x_bs_blur_path)).astype('float32'), reference_raster=coh_x_bs_blur_path, out_raster=out_folder + 'COH_x_BS_blur_norm.tif', dst_nodata=-999.9)
# array_to_raster(standardise_filter(raster_to_array(ndvi_path)).astype('float32'), reference_raster=ndvi_path, out_raster=out_folder + 'NDVI_norm.tif', dst_nodata=-999.9)
# array_to_raster(standardise_filter(raster_to_array(nl_path)).astype('float32'), reference_raster=nl_path, out_raster=out_folder + 'NL_norm.tif', dst_nodata=-999.9)
# array_to_raster(standardise_filter(raster_to_array(b4_tex)).astype('float32'), reference_raster=b4_tex, out_raster=out_folder + 'B4_MAD_norm.tif', dst_nodata=-999.9)
# array_to_raster(standardise_filter(raster_to_array(b8_tex_path)).astype('float32'), reference_raster=b8_tex_path, out_raster=out_folder + 'B8_MAD_norm.tif', dst_nodata=-999.9)

# merge(out_folder + 'merged.vrt', [out_folder + 'B4_norm.tif', out_folder + 'B8_norm.tif'])

# nl_t = np.clip(nl, 0, 25)
# array_to_raster((nl_t - np.min(nl_t)) / (np.max(nl_t) - np.min(nl_t)), reference_raster=nl_path, out_raster=folder + 'NL_norm.tif')


# B4 = raster_to_array(b4_path)
# B8 = raster_to_array(b8_path)

# bs_blur = cv2.GaussianBlur(raster_to_array(bs_path), (11, 11), 0)
# array_to_raster(np.sqrt(raster_to_array(blur_path)).astype('float32'), out_raster=folder + 'bs_blur_sqrt.tif', reference_raster=b4_path)

# array_to_raster(np.clip(((-1 * ndvi) + 1), 0, 1).astype('float32'), out_raster=folder + 'inv_ndvi_s.tif', reference_raster=coh_path)
# array_to_raster((-1 * ndvi).astype('float32'), out_raster=folder + 'inv_ndvi.tif', reference_raster=coh_path)
