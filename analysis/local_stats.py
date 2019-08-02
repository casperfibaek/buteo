import sys
sys.path.append('../lib')

from local_statistics import local_statistics2

in_raster = 'C:\\Users\\CFI\\Desktop\\road_experiment\\roads_10m.tif'
out_raster = 'C:\\Users\\CFI\\Desktop\\road_experiment\\road_density_200m.tif'
out_raster_smooth = 'C:\\Users\\CFI\\Desktop\\road_experiment\\road_density_200m_smooth.tif'

local_statistics2(in_raster, stat='sum', out_raster=out_raster, radius=20, dst_nodata=None)
local_statistics2(out_raster, stat='mean', out_raster=out_raster_smooth, radius=10, dst_nodata=None)
