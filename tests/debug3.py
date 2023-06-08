# pylint: disable-all
import sys; sys.path.append("../")

import os
import numpy as np
import buteo as beo
from osgeo import gdal

FOLDER = "./features/"

path = os.path.join(FOLDER, "test_image_rgb_8bit.tif")

arr = beo.raster_to_array(path)
arr_filtered = beo.filter_median(arr, 2)
