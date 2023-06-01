# pylint: disable-all
import sys; sys.path.append("../")

import os
import numpy as np
import buteo as beo
from osgeo import gdal

FOLDER = "./features/"

path = os.path.join(FOLDER, "test_image_rgb_8bit.tif")

arr1 = beo.raster_to_array(path)
raster = beo.array_to_raster(arr1, reference=path, out_path="/vsimem/test.tif")
arr2 = beo.raster_to_array(raster)
import pdb; pdb.set_trace()
