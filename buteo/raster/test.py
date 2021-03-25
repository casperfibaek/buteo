import sys; sys.path.append('../../')
from glob import glob
import numpy as np

folder = "C:/Users/caspe/Desktop/test/"

from buteo.raster.clip import clip_raster
from buteo.machine_learning.patch_extraction import extract_patches

raster = folder + "fyn.tif"
target = folder + "odense.gpkg"

# vector = vector_to_reference(target)

extract_patches(
    raster,
    folder,
    clip_to_vector=target,
    offsets=[(16, 16)]
)

# import pdb; pdb.set_trace()