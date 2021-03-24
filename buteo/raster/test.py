import sys; sys.path.append('../../')
from glob import glob
import numpy as np

folder = "C:/Users/caspe/Desktop/test/"

from buteo.raster.io import raster_to_array, array_to_raster

raster = glob(folder + "*.tif")

arr = raster_to_array(raster[0])

import pdb; pdb.set_trace()