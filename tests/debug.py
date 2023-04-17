""" This is a debug script, used for ad-hoc testing. """

# Standard library
import sys; sys.path.append("../")
import os

from buteo.eo.s2_utils import s2_l2a_get_metadata
from buteo.raster.patches import array_to_patches

FOLDER = "C:/Users/casper.fibaek/OneDrive - ESA/Desktop/s2_data/"

zip_file = os.path.join(FOLDER, "S2A_MSIL2A_20220117T090321_N0301_R007_T35TPE_20220117T120927.zip")

metadata = s2_l2a_get_metadata(zip_file)
bands_10m = metadata["bands_10m"]()

patches = array_to_patches(bands_10m, tile_size=128, offsets_x=1, offsets_y=1, border_check=True)

final = (patches / 10000.0).astype("float32")
