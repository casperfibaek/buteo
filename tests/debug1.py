""" This is a debug script, used for ad-hoc testing. """
# disable all of pylint for this file only.
# pylint: disable-all

# Standard library
import sys; sys.path.append("../")
import os
import numpy as np
import buteo as beo

FOLDER = "C:/Users/casper.fibaek/OneDrive - ESA/Desktop/test_data/"

raster = os.path.join(FOLDER, "landcover.tif")
classes = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100])

arr = beo.raster_to_array(raster, fill_value=80, filled=True, cast=np.uint8)
dst = np.zeros((arr.shape[0], arr.shape[1], len(classes)), dtype=np.float32)

for i, c in enumerate(classes):
    feathered = beo.filter_operation(
        arr,
        15, # count weighted occurances
        radius=2, # 5x5
        func_value=int(c),
        normalised=False,
    )
    dst[:, :, i] = feathered[:, :, 0]

dst = dst / np.sum(dst, axis=2, keepdims=True)
beo.array_to_raster(dst, reference=raster, out_path=os.path.join(FOLDER, "feathered.tif"))

# Kernel Distance weighing Parameters
# method : int
#     Method to use for weighting.
#     0. linear
#     1. sqrt
#     2. power
#     3. gaussian
#     4. constant

# decay : float
#     Decay rate for distance weighted kernels. Only used if `distance_weighted` is True.

# sigma : float
#     Sigma for gaussian distance weighted kernels. Only used if `distance_weighted` is True and `method` is 3.
