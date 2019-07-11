import sys
import os
import numpy as np
import time
from glob import glob

sys.path.append('../lib')
from raster_stats import raster_stats
from raster_to_array import raster_to_array
from array_to_raster import array_to_raster

before = time.time()

base = 'D:\\PhD\\Projects\\SavingsAtTheFrontiers\\pilot_analysis\\data\\MSI\\'
out_folder = 'D:\\PhD\\Projects\\SavingsAtTheFrontiers\\pilot_analysis\\data\\MSI\\'

in_raster = f"{base}R10m_B12.tif"

stats = raster_stats(in_raster, statistics=['min', 'q02', 'q98'])
shifted = np.add(raster_to_array(in_raster), abs(stats['min']))
new_max = stats['q98'] + abs(stats['min'])
norm_ratio = 1000 / new_max
new_arr = np.multiply(shifted, norm_ratio)
array_to_raster(new_arr, reference_raster=in_raster, out_raster=f"{base}R10m_B12_scaled.tif")
