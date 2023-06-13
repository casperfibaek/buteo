# pylint: disable-all
import sys; sys.path.append("../")

import os
import numpy as np
import buteo as beo
from osgeo import gdal
from glob import glob
from tqdm import tqdm


VRT = "E:/data/terrain/"
REF = "C:/Users/casper.fibaek/OneDrive - ESA/Desktop/projects/unicef/data/albedo/albedo_2019_bb_54009_500m.tif" 
OUT = "E:/data/terrain_500m/"

images = glob(VRT + "*.tif")
for img in tqdm(images, total=len(images)):
    try:
        reprojected = beo.raster_reproject(img, REF, resample_alg="average")
        resampled = beo.raster_resample(
            reprojected,
            REF,
            resample_alg="average",
            out_path=os.path.join(OUT, os.path.basename(img)),
        )
        beo.delete_dataset_if_in_memory(reprojected)
    except:
        print(f"Failed on {img}")
        continue

import pdb; pdb.set_trace()