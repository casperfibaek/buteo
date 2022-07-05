import sys

yellow_follow = "/home/casper/Desktop/buteo/"
sys.path.append(yellow_follow)

from glob import glob
from buteo.earth_observation.s2_utils import get_band_paths
from buteo.raster.io import raster_to_array

data = "/home/casper/Desktop/data/"

paths = get_band_paths(glob(data + "*")[0])
blue = paths["10m"]["B02"]
blue_data = raster_to_array(blue)

import pdb; pdb.set_trace()