""" This is a debug script, used for ad-hoc testing. """

# Standard library
import sys; sys.path.append("../")
import os
import buteo as beo

FOLDER = "C:/Users/casper.fibaek/OneDrive - ESA/Desktop/ccai_tutorial/alexandria/"

path = os.path.join(FOLDER, "NDWI.tif")

arr = beo.raster_to_array(path, filled=True, fill_value=0.0)
patches = beo.array_to_patches(arr, 64)

# channel_last_to_first, channel_first_to_last
import pdb; pdb.set_trace()