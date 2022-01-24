from doctest import master
import sys
import numpy as np


sys.path.append("../")
sys.path.append("../../")
np.set_printoptions(suppress=True)

from buteo.raster.resample import resample_raster
from buteo.raster.io import raster_to_array, array_to_raster
from buteo.filters.convolutions import filter_array
from buteo.raster.align import align_rasters
from buteo.raster.clip import clip_raster

folder = "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/results/"

pop = folder + "egypt_population_unweighted.tif"

resample_raster(
    pop,
    100,
    out_path=folder + "egypt_population_unweighted_100m.tif",
    resample_alg="sum",
    postfix="",
)

pop = folder + "ghana_population_unweighted.tif"

resample_raster(
    pop,
    100,
    out_path=folder + "ghana_population_unweighted_100m.tif",
    resample_alg="sum",
    postfix="",
)
