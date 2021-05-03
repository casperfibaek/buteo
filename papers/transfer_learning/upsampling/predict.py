yellow_follow = "C:/Users/caspe/Desktop/buteo/"
import sys

sys.path.append(yellow_follow)

from buteo.machine_learning.patch_extraction import predict_raster
from buteo.raster.io import raster_to_array, array_to_raster
import tensorflow as tf
import h5py


folder = "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/upsampling/"

merge_10 = [
    folder + "32UMF_B02_10m.tif",
    folder + "32UMF_B03_10m.tif",
    folder + "32UMF_B04_10m.tif",
]


model = folder + "upsampling_model_new_10epochs.h5"
loaded = tf.keras.models.load_model(model)
band_11 = folder + "32UMF_B11_20m.tif"

rgb = array_to_raster(raster_to_array(merge_10), reference=merge_10[0])

path = predict_raster(
    [rgb, band_11],
    loaded,
    out_path=folder + "upsample_02.tif",
    offsets=[[(32, 32), (64, 64), (96, 96)], [(16, 16), (32, 32), (48, 48)]],
    batch_size=64,
    mirror=False,
    rotate=False,
    device="gpu",
)
