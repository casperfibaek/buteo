import os
import sys; sys.path.append("../")
from glob import glob
from tqdm import tqdm

import buteo as beo
import numpy as np

FOLDER = "D:/data/B8_2_B8A_model/images/"
DST = "D:/data/B8_2_B8A_model/"
PROP = 0.2


all_patches = []

images = glob(FOLDER + "*B08.tif")
for idx, img in enumerate(tqdm(images, total=len(images))):
    path_b08 = img
    path_b8a = img.replace("B08", "B8A")

    arr = beo.raster_to_array([path_b08, path_b8a], filled=True, fill_value=0, cast="uint16")

    patches = beo.array_to_patches(arr, 128)
    random_indices = np.random.permutation(patches.shape[0])[:int(PROP * patches.shape[0])]
    patches = patches[random_indices]

    all_patches.append(patches)

all_patches = np.concatenate(all_patches, axis=0)
shuffle_mask = np.random.permutation(all_patches.shape[0])
all_patches = all_patches[shuffle_mask]

np.save(os.path.join(DST, "patches_b08-b8a.npy"), all_patches)
