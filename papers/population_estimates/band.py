yellow_follow = "C:/Users/caspe/Desktop/buteo/"
import sys
import os

sys.path.append(yellow_follow)

import numpy as np
from glob import glob
from osgeo import gdal
from buteo.raster.io import stack_rasters, raster_to_array
from buteo.machine_learning.patch_extraction_v2 import predict_raster
from buteo.vector.rasterize import rasterize_vector
from buteo.raster.clip import clip_raster
from buteo.raster.io import (
    raster_to_array,
    array_to_raster,
    stack_rasters_vrt,
)
from buteo.orfeo_toolbox import merge_rasters, obt_bandmath

folder = "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/data/ghana/predictions/"

obt_bandmath(
    [folder + "Ghana_classification_float32_v2.tif", folder + "Ghana_float32_v9.tif"],
    "((im1b2 + im1b3 + im1b4) == 0) ? im2b1 : (im1b2 / (im1b2 + im1b3 + im1b4)) * im2b1",
    folder + "area_residential.tif",
    ram=32000,
)

obt_bandmath(
    [folder + "Ghana_classification_float32_v2.tif", folder + "Ghana_float32_v9.tif"],
    "((im1b2 + im1b3 + im1b4) == 0) ? 0.0 : (im1b3 / (im1b2 + im1b3 + im1b4)) * im2b1",
    folder + "area_industrial.tif",
    ram=32000,
)

obt_bandmath(
    [folder + "Ghana_classification_float32_v2.tif", folder + "Ghana_float32_v9.tif"],
    "((im1b2 + im1b3 + im1b4) == 0) ? 0.0 : (im1b4 / (im1b2 + im1b3 + im1b4)) * im2b1",
    folder + "area_slum.tif",
    ram=32000,
)
