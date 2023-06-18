import os
import sys; sys.path.append("../")
from glob import glob
from tqdm import tqdm

import buteo as beo
import numpy as np


FOLDER = "C:/Users/casper.fibaek/OneDrive - ESA/Desktop/s12_buildings/data_raw/israel_gaza_2/"

ref = os.path.join(FOLDER, "labels.tif")
ref_resampled = beo.raster_resample(ref, 10.0)

images = glob(os.path.join(FOLDER, "*.tif"))

beo.raster_align(
    images,
    reference=ref_resampled,
    out_path=FOLDER + "aligned/",
    target_nodata=None,
)
