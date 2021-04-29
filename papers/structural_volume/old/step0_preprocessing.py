import sys; sys.path.append('c:/Users/caspe/desktop/yellow')
from lib.raster_io import raster_to_array, array_to_raster
from lib.stats_filters import mad_std_filter, scale_to_range_filter, truncate_filter

project_folder = 'c:/Users/caspe/desktop/C:\Users\caspe\Desktop\Paper_2_StructuralVolume/'

in_files = ['b04_spring.tif', 'b08_spring.tif']

for f in in_files:
  ref_path = project_folder + f
  ref = raster_to_array(ref_path)
  filtered = mad_std_filter(ref, 5)
  array_to_raster(filtered, project_folder + f.split('.')[0] + '_madstd5.tif', ref_path)


