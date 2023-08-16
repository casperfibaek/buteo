""" This is a debug script, used for ad-hoc testing. """
# disable all of pylint for this file only.
# pylint: disable-all

# Standard library
import sys; sys.path.append("../")
import os
import numpy as np
import buteo as beo

FOLDER = "C:/Users/casper.fibaek/OneDrive - ESA/Desktop/test_data/"

raster = os.path.join(FOLDER, "GHA1_51_s2.tif")
for v in beo.raster_to_metadata(raster).items():
    print(v)
