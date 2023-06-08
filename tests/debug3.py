# pylint: disable-all
import sys; sys.path.append("../")

import os
import numpy as np
import buteo as beo
from osgeo import gdal

FOLDER = "./features/"

path = os.path.join(FOLDER, "prediction_2.tif")

arr = beo.raster_to_array(path)
arr_filtered = beo.filter_median(arr, radius=2)

beo.array_to_raster(
    arr_filtered,
    reference=path,
    out_path=os.path.join(FOLDER, "prediction_2_median_5x5.tif"),
)
