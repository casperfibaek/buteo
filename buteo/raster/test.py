import sys; sys.path.append('../../')

folder = "C:/Users/caspe/Desktop/test/"

from buteo.raster.io import raster_to_memory

raster = folder + "fyn_vv.tif"

mem = raster_to_memory(raster)
