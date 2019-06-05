import sys
import os
import numpy as np
import time
from glob import glob

sys.path.append('../lib')
from array_to_raster import array_to_raster
from raster_to_array import raster_to_array

before = time.time()

coh = 'D:\\PhD\\Projects\\Byggemodning\\_temp_\\coh_align.tif'
db = 'D:\\PhD\\Projects\\Byggemodning\\_temp_\\db_align.tif'

coh_arr = raster_to_array(coh)
db_arr = raster_to_array(db)

mult = np.multiply(coh_arr, db_arr)
array_to_raster(mult, reference_raster=coh, out_raster='D:\\PhD\\Projects\\Byggemodning\\_temp_\\db_times_coh_test2019_01.tif')
