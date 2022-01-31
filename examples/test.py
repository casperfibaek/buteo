from doctest import master
import sys
from turtle import distance
import numpy as np
from osgeo import gdal


sys.path.append("../")
sys.path.append("../../")
np.set_printoptions(suppress=True)

from buteo.raster.resample import resample_raster
from buteo.raster.io import raster_to_array, array_to_raster
from buteo.filters.convolutions import filter_array
from buteo.raster.align import align_rasters
from buteo.raster.clip import clip_raster
from buteo.raster.reproject import reproject_raster

folder = "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/data/tanzania_kilimanjaro/patches/"

bob = np.load(folder + "kili_compressed.npz")
label = bob["label"]
rgbn = bob["rgbn"]
reswir = bob["reswir"]
sar = bob["sar"]

bob = None

import pdb; pdb.set_trace()

# label = np.load(folder + "label_area.npy")
# label_masked = label.sum(axis=(1, 2))[:, 0]

# label_mask = np.arange(label_masked.shape[0])

# above = label_mask[label_masked > 0]
# below = label_mask[label_masked == 0][:above.shape[0] // 10]
# merged = np.concatenate([above, below])
# shuffle_mask = np.random.permutation(merged.shape[0])
# mask = merged[shuffle_mask]

# np.savez_compressed(
#     folder + "kili_compressed.npz",
#     label=label[mask],
#     rgbn=np.load(folder + "RGBN.npy")[mask],
#     reswir=np.load(folder + "RESWIR.npy")[mask],
#     sar=np.load(folder + "SAR.npy")[mask],
# )
