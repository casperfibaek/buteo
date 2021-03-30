import sys; sys.path.append('../../')
from glob import glob
import numpy as np


# from buteo.raster.warp import warp_raster
from buteo.raster.io import raster_to_metadata, raster_to_memory
# from buteo.raster.clip import clip_raster
from buteo.vector.io import vector_to_metadata, vector_to_memory

folder = "C:/Users/caspe/Desktop/test/"

vector_path = folder + "odense.gpkg"
vector_memory = vector_to_memory(vector_path)

# raster_path = folder + "fyn.tif"
# raster_memory = raster_to_memory(raster_path)

# raster_to_metadata(raster_path)
# raster_to_metadata(raster_memory)

# vector_to_metadata(vector_path)
vector_to_metadata(vector_memory)
