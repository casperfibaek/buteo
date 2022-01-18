import sys

sys.path.append("../")
sys.path.append("../../")
import numpy as np
from buteo.raster.io import (
    raster_to_array,
    array_to_raster,
)
from buteo.raster.clip import clip_raster
from buteo.raster.resample import resample_raster

folder = "C:/Users/caspe/Desktop/egypt_accuracy_test/population/"

v7_10m = folder + "egypt_v7_unweighted_pop.tif"

resample_raster(
    v7_10m,
    100,
    out_path=folder + "egypt_v7_unweighted_pop_100m.tif",
    resample_alg="sum",
)

v13_10m = folder + "egypt_v13_unweighted_pop.tif"

resample_raster(
    v13_10m,
    100,
    out_path=folder + "egypt_v13_unweighted_pop_100m.tif",
    resample_alg="sum",
)
