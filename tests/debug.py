# Standard library
import sys; sys.path.append("../")

import os

import numpy as np
from buteo import (
    raster_to_array,
    array_to_raster,
    patches_to_array,
    array_to_patches,
)


FOLDER = r"C:/Users/casper.fibaek/OneDrive - ESA/Desktop/Unicef/"

# raster = os.path.join(FOLDER, "GHS_SMOD_E2020_GLOBE_R2022A_54009_1000_V1_0.tif")
# arr = raster_to_array(raster)

# blocks, og_coords, og_shape = array_to_patches(
#     arr,
#     tile_size=64,
#     offset=[0, 0],
# )

TILE_SIZE = 64
OFFSETS = 3


offsets = get_offsets(TILE_SIZE, OFFSETS, OFFSETS)
print(offsets)
# import pdb ; pdb.set_trace()
