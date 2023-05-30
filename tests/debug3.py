# pylint: disable-all
import sys; sys.path.append("../")

import os
import numpy as np
import buteo as beo
from osgeo import gdal

FOLDER = "./features/"

path = os.path.join(FOLDER, "test_image_rgb_8bit.tif")

arr = beo.raster_to_array(path)
resampled = beo.raster_resample(path, 20.0, resample_alg="bilinear")

import pdb; pdb.set_trace()