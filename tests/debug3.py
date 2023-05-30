# pylint: disable-all
import sys; sys.path.append("../")

import os
import numpy as np
import buteo as beo
from osgeo import gdal

FOLDER = "./features/"

path = os.path.join(FOLDER, "test_image_rgb_8bit.tif")

arr = beo.raster_to_array(path)
reprojected = beo.raster_reproject(path, path)
resampled = beo.raster_resample(reprojected, path, resample_alg="bilinear")
clipped = beo.raster_clip(resampled, path)