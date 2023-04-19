""" This is a debug script, used for ad-hoc testing. """

# Standard library
import sys; sys.path.append("../")
import os

from buteo.vector.split import split_vector_by_fid
from buteo.raster import raster_to_array, array_to_raster, clip_raster

FOLDER = "C:/Users/casper.fibaek/OneDrive - ESA/Desktop/ccai_tutorial/gaza_israel/"
FOLDER_OUT = FOLDER + "patches/"

split_files = split_vector_by_fid(
    os.path.join(FOLDER, "mask.gpkg"),
)

mask_raster_path = os.path.join(FOLDER, "labels_10m.tif")

for idx, split_file in enumerate(split_files):
    mask_clipped = clip_raster(
        mask_raster_path,
        split_file,
        adjust_bbox=True,
    )

    mask_arr = raster_to_array(mask_clipped, filled=True, fill_value=0)

    array_to_raster(
        mask_arr,
        reference=mask_clipped,
        out_path=os.path.join(FOLDER_OUT, f"label_{idx}.tif"),
    )

import pdb; pdb.set_trace()
