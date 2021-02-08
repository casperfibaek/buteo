yellow_follow = 'C:/Users/caspe/Desktop/yellow/lib/'

import sys; sys.path.append(yellow_follow) 
from patch_extraction import extract_patches
from raster_clip import clip_raster
from glob import glob
import os

folder = "C:/Users/caspe/Desktop/Paper_2_StructuralVolume/"
ref = folder + "clip/reference.tif"

images = glob(folder + '*.tif')
for image in images:
    name = os.path.splitext(os.path.basename(image))[0]

    clipped = clip_raster(image, reference_raster=ref)

    extract_patches(
        clipped,
        folder + f"clip/{name}.npy",
        size=16,
        # overlaps=[(8, 0), (8, 8), (0, 8)],
        # fill_value=0,
    )