import sys; sys.path.append('c:/Users/caspe/desktop/yellow')
import os
from glob import glob
from lib.stats_zonal import calc_zonal

folder = 'C:/Users/caspe/Desktop/Paper_2_StructuralVolume/'
rasters = glob(folder + '*.tif')
prefixes = []

for raster in rasters:
    name = os.path.splitext(os.path.basename(raster))[0] 
    prefixes.append(name + "_")

calc_zonal(folder + '160x160m_all.gpkg', rasters, prefixes, ['mean', 'med', 'std', 'min', 'max'])