import sys
sys.path.append('../lib')

from local_statistics import local_statistics2

in_raster = 'C:\\Users\\CFI\\Desktop\\road_experiment\\roads_10m.tif'
out_raster = 'C:\\Users\\CFI\\Desktop\\road_experiment\\road_density_1km2_circ.tif'
# out_raster_smooth = 'C:\\Users\\CFI\\Desktop\\satf_segmentation\\dry_surf_coh2_clip_med_smooth.tif'
# out_raster_smooth = 'D:\\PhD\\Projects\\SavingsAtTheFrontiers\\pilot_analysis_v02\\data\\processed\\surf_max_abs_density_100m_med_smooth.tif'
# in_raster = 'C:\\Users\\CFI\\Desktop\\road_experiment\\roads_10m.tif'
# out_raster = 'C:\\Users\\CFI\\Desktop\\road_experiment\\road_density_200m.tif'
# out_raster_smooth = 'C:\\Users\\CFI\\Desktop\\road_experiment\\road_density_200m_smooth.tif'

local_statistics2(in_raster, stat='sum', out_raster=out_raster, dtype='uint16', radius=100, dst_nodata=None)
# local_statistics2(out_raster, stat='mean', out_raster=out_raster_smooth, radius=10, dst_nodata=None)
