""" This is a debug script, used for ad-hoc testing. """
# disable all of pylint for this file only.
# pylint: disable-all

# Standard library
import sys; sys.path.append("../")
import os
from glob import glob
import buteo as beo

FOLDER = "./features/"

raster_utm = os.path.join(FOLDER, "test_image_rgb_8bit.tif")

bbox_latlng = [116.206538452, 116.243267680, 6.165653284, 6.198924241]

bbox = beo.reproject_bbox(bbox_latlng, "EPSG:4326", raster_utm)
# read = beo.raster_to_array(raster_utm, bbox=bbox)

import pdb; pdb.set_trace()

# beo.array_to_raster(
#     read,
#     bbox=bbox_latlng,
#     out_path=os.path.join(FOLDER, "test_image_rgb_8bit_bbox_latlng.tif"),
#     reference=raster_latlng,
# )
