yellow_follow = "C:/Users/caspe/Desktop/buteo/"
import sys

sys.path.append(yellow_follow)

# from buteo.machine_learning.patch_extraction import predict_raster
# import tensorflow as tf
from buteo.raster.io import raster_to_array, array_to_raster
import numpy as np
import h5py


folder = "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/upsampling/"

merge_10 = [
    raster_to_array(folder + "32UMF_B02_10m.tif"),
    raster_to_array(folder + "32UMF_B03_10m.tif"),
    raster_to_array(folder + "32UMF_B04_10m.tif"),
]

for idx in range(len(merge_10)):
    merge_10[idx] = (((merge_10[idx] - merge_10[idx].min()) / (merge_10[idx].max() - merge_10[idx].min()))).astype("float32")

merge_10 = np.mean(np.dstack(merge_10), axis=2)

import pdb; pdb.set_trace()



lumi = (np.sqrt((0.299 * ((merge_10[2] / 65534) ** 2)) + (0.587 * ((merge_10[1]) /  65534) ** 2) + (0.114 * ((merge_10[0] /  65534) ** 2)) *  65534)).astype("float32")

# model = folder + "upsampling_model_10epochs_norm.h5"
# loaded = tf.keras.models.load_model(model)
# band_11 = raster_to_array(folder + "32UMF_B11_20m.tif")
# band_11_max = band_11.max()
# band_11 = ((band_11 / band_11_max) * 1000).astype("float32")
# band_11 = array_to_raster(band_11, reference=folder + "32UMF_B11_20m.tif")

# rgb_stack = np.dstack(merge_10)
# rgb = array_to_raster(rgb_stack, reference=folder + "32UMF_B02_10m.tif")

# path = predict_raster(
#     [rgb, band_11],
#     loaded,
#     out_path=folder + "upsample_07_norm.tif",
#     offsets=[[(32, 32), (64, 64), (96, 96)], [(16, 16), (32, 32), (48, 48)]],
#     # offsets=[[], []],
#     batch_size=64,
#     mirror=False,
#     rotate=False,
#     device="gpu",
# )

array_to_raster(bob, reference=folder + "32UMF_B02_10m.tif", out_path=folder + "lumi_03.tif")

# array_to_raster(
#     np.rint((raster_to_array(folder + "upsample_07_norm.tif") / 1000) * band_11_max).astype("uint16"),
#     folder + "upsample_07_norm.tif",
#     out_path=folder + "upsample_07_norm_uint16.tif",
# )