import numpy as np
from utils import (
    preprocess_optical,
    preprocess_sar,
    random_scale_noise,
)
from glob import glob


folder = "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/data/ghana/raster/patches/"

rgbn = np.stack(
    [
        np.load(folder + "2021_B02_10m.npy"),
        np.load(folder + "2021_B03_10m.npy"),
        np.load(folder + "2021_B04_10m.npy"),
        np.load(folder + "2021_B08_10m.npy"),
    ],
    axis=3,
)[:, :, :, :, 0]
rgbn = preprocess_optical(random_scale_noise(rgbn))

swir = np.stack(
    [
        np.load(folder + "2021_B11_20m.npy"),
        np.load(folder + "2021_B12_20m.npy"),
    ],
    axis=3,
)[:, :, :, :, 0]
swir = preprocess_optical(random_scale_noise(swir))

sar = np.stack(
    [
        np.load(folder + "2021_VH_10m.npy"),
        np.load(folder + "2021_VV_10m.npy"),
    ],
    axis=3,
)[:, :, :, :, 0]
sar = preprocess_sar(random_scale_noise(sar))

area = np.load(folder + "area.npy")

sample_arr = [True, False]
shuffle_mask = np.random.permutation(swir.shape[0])
flip = np.random.choice(sample_arr, size=swir.shape[0])
norm = ~flip

np.save(
    folder + "001_RGBN.npy",
    np.concatenate(
        [
            np.rot90(rgbn[flip], k=2, axes=(1, 2)),
            rgbn[norm],
        ]
    )[shuffle_mask],
)

np.save(
    folder + "001_SWIR.npy",
    np.concatenate(
        [
            np.rot90(swir[flip], k=2, axes=(1, 2)),
            swir[norm],
        ]
    )[shuffle_mask],
)

np.save(
    folder + "001_SAR.npy",
    np.concatenate(
        [
            np.rot90(sar[flip], k=2, axes=(1, 2)),
            sar[norm],
        ]
    )[shuffle_mask],
)

np.save(
    folder + "001_LABEL_AREA.npy",
    np.concatenate(
        [
            np.rot90(area[flip], k=2, axes=(1, 2)),
            area[norm],
        ]
    )[shuffle_mask],
)
