import sys; sys.path.append('..')
import numpy as np
import geopandas as gpd
from time import time
from glob import glob
from pathlib import Path
from lib.stats_zonal import calc_zonal, calc_shapes
from lib.orfeo_toolbox import meanshift_segmentation
from lib.utils_core import step_function

# phase1_folder = 'C:/users/caspe/desktop/Analysis/Phase1/'

# test_grid = {
#     'spatialr': [3, 5, 7, 9],
#     'ranger': [0.2, 0.3, 0.4],
#     'thres': [0.1, 0.25, 0.5],
#     'minsize': [50, 100, 200]
# }

# step_function(
#     meanshift_segmentation,
#     phase1_folder + 'phase1_layers.vrt',
#     grid=test_grid,
#     outfile=True,
#     outfile_arg=1,
#     outfile_prefix=phase1_folder + 'segmentations/segmentation_',
#     outfile_suffix='.shp',
# )


folder = '/mnt/c/users/caspe/desktop/Analysis/'

# in_vector = folder + 'Phase2/vector/phase2_segmentation_zonal.gpkg'
# df = gpd.read_file(in_vector)

# ipq_std = df.loc[:,"ipq"].std()
# ipq_mean = df.loc[:,"ipq"].mean()
# area_std = df.loc[:,"area"].std()
# area_mean = df.loc[:,"area"].mean()
# peri_std = df.loc[:,"perimeter"].std()
# peri_mean = df.loc[:, "perimeter"].mean()

# df["ipq_std"] = (df["ipq"] - ipq_mean) / ipq_std
# df["area_std"] = (df["area"] - area_mean) / area_std
# df["peri_std"] = (df["perimeter"] - peri_mean) / peri_std

# df.to_file(folder + "Phase2/vector/phase2_segmentation_zonal_std.gpkg", layer='phase2_segmentation_zonal_std', driver="GPKG")

# import pdb; pdb.set_trace()
# before = time()
# calc_shapes(in_vector)
# print('Calculation took: ', time() - before, ' seconds')

in_vector = folder + 'Phase2/vector/phase2_segmentation_zonal_std.gpkg'

in_rasters = [
    # folder + 'Data/standardized/' + 'dem_slope_std.tif',
    # folder + 'Data/standardized/' + 'nightlights_std.tif',
    # folder + 'Data/standardized/' + 'roads_merge_1km_std.tif',
    # folder + 'Data/standardized/' + 's1_dry_coh_std.tif',
    # folder + 'Data/standardized/' + 's1_dry_perm_std.tif',
    # folder + 'Data/standardized/' + 's1_dry_sigma0_std.tif',
    # folder + 'Data/standardized/' + 's1_wet_coh_std.tif',
    # folder + 'Data/standardized/' + 's1_wet_perm_std.tif',
    # folder + 'Data/standardized/' + 's1_wet_sigma0_std.tif',
    # folder + 'Data/standardized/' + 's2_b04_10m_dry_std.tif',
    # folder + 'Data/standardized/' + 's2_b04_10m_dry_tex_std.tif',
    # folder + 'Data/standardized/' + 's2_b04_10m_wet_std.tif',
    # folder + 'Data/standardized/' + 's2_b04_10m_wet_tex_std.tif',
    # folder + 'Data/standardized/' + 's2_b08_10m_dry_std.tif',
    # folder + 'Data/standardized/' + 's2_b08_10m_dry_tex_std.tif',
    # folder + 'Data/standardized/' + 's2_b08_10m_wet_std.tif',
    # folder + 'Data/standardized/' + 's2_b08_10m_wet_tex_std.tif',
    folder + 'Data/standardized/' + 's2_b12_10m_dry_std.tif',
    folder + 'Data/standardized/' + 's2_b12_10m_wet_std.tif',
]

prefixes = [
    # 'slop_',
    # 'nigh_',
    # 'road_',
    # 'dcoh_',
    # 'dper_',
    # 'dsig_',
    # 'wcoh_',
    # 'wper_',
    # 'wsig_',
    # 'b4d_',
    # 'b4dt_',
    # 'b4w_',
    # 'b4wt_',
    # 'b8d_',
    # 'b8dt_',
    # 'b8w_',
    # 'b8wt_',
    'b12d_',
    'b12w_',
]

before = time()
calc_zonal(in_vector, in_rasters, prefixes=prefixes)
print('Calculation took: ', time() - before, ' seconds')