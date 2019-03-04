from array_to_raster import array_to_raster
from raster_to_array import raster_to_array
from raster_stats import raster_stats
from utils import scale_zscores
import numpy as np


def calc_zscores(layer, vector):
    org_stats = raster_stats(layer, cutline=vector, statistics=['mean', 'madstd'])
    print(org_stats)
    org_arr = raster_to_array(layer)

    return np.divide(np.subtract(org_arr, org_stats['mean']), org_stats['madstd'])

vector = '../geometry/roses.geojson'
nir_pca = '../raster/T32VNJ_20180727T104021_nir_pca_10m.tif'
nir_pca_tex = '../raster/T32VNJ_20180727T104021_nir_pca_10m_tex.tif'
swir_pca = '../raster/T32VNJ_20180727T104021_swir_pca_10m.tif'
swir_pca_tex = '../raster/T32VNJ_20180727T104021_swir_pca_10m_tex.tif'
vis_pca = '../raster/T32VNJ_20180727T104021_vis_pca_10m.tif'
vis_pca_tex = '../raster/T32VNJ_20180727T104021_vis_pca_10m_tex.tif'

# nir = np.add(calc_zscores(nir_pca, vector), calc_zscores(nir_pca_tex, vector))
# vis = np.add(calc_zscores(vis_pca, vector), calc_zscores(vis_pca_tex, vector))
# swir = np.add(calc_zscores(swir_pca, vector), calc_zscores(swir_pca_tex, vector))

# z = np.divide(np.add(np.add(nir, vis), swir), 3)

array_to_raster(calc_zscores(nir_pca_tex, vector), reference_raster=nir_pca, out_raster='../raster/nir_tex_z_roses.tif')
