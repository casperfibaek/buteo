yellow_follow = 'C:/Users/caspe/Desktop/yellow/lib/'

import sys; sys.path.append(yellow_follow) 
from patch_extraction import extract_patches
from raster_clip import clip_raster
from glob import glob
import os

folder = "C:/Users/caspe/Desktop/Paper_2_StructuralVolume/imagery_unscaled/"
ref = folder + "clip_file_silkeborg.gpkg"

images = glob(folder + 'bob_*.tif')
for image in images:
    name = os.path.splitext(os.path.basename(image))[0]

    clip_raster(image, crop_to_cutline=True, cutline=ref, out_raster=folder + f"/clipped/{name}.tif")
