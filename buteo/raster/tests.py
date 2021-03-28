import sys; sys.path.append('../../')
from glob import glob
import numpy as np


# from buteo.raster.warp import warp_raster
from buteo.raster.align import align_rasters
from buteo.raster.io import raster_to_array, array_to_raster
# from buteo.raster.clip import clip_raster
# from buteo.vector.io import vector_to_reference

folder = "C:/Users/caspe/Desktop/test/align/comp/"
rasters = glob(folder + "*.tif")

raster = raster_to_array(rasters[2])

import pdb; pdb.set_trace()
