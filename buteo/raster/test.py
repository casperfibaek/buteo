import sys; sys.path.append('../../')
import numpy as np

from buteo.machine_learning.patch_extraction import extract_patches, test_extraction

folder = "C:/Users/caspe/Desktop/test/"
raster = folder + "dtm.tif"
patches = folder + "dtm_patches.npy"
geom = folder + "patches_64_patches.gpkg"
# target = folder + "odense.gpkg"

# vector = vector_to_reference(target)

# extract_patches(
#     raster,
#     folder,
#     size=64,
#     output_geom=True,
# )


test_extraction(raster, patches, geom)
import pdb; pdb.set_trace()


# from matplotlib import pyplot as plt

# en [1] == en [3]
