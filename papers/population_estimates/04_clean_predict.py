yellow_follow = "C:/Users/caspe/Desktop/buteo/"
import sys

sys.path.append(yellow_follow)

import os
import time
import numpy as np

import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.callbacks import (
    LearningRateScheduler,
    ModelCheckpoint,
    EarlyStopping,
)
from tensorflow.keras import backend as K
from buteo.machine_learning.ml_utils import create_step_decay, mse, tpe
from buteo.utils import timing

from model_trio_down import model_trio_down

np.set_printoptions(suppress=True)
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
mixed_precision.set_global_policy("mixed_float16")

folder = "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/data/"
outdir = folder + f"models/"
# place = "dojo"
place = "ghana"

x_train = [
    np.load(folder + f"{place}/patches/RGBN.npy"),
    np.load(folder + f"{place}/patches/SAR.npy"),
    np.load(folder + f"{place}/patches/RESWIR.npy"),
]

y_train = np.load(folder + f"{place}/patches/label_area.npy")

# x_train = [
#     np.concatenate(
#         [
#             np.load(folder + f"{place}/patches/dar_RGBN.npy"),
#             np.load(folder + f"{place}/patches/kampala_RGBN.npy"),
#             np.load(folder + f"{place}/patches/kilimanjaro_RGBN.npy"),
#             np.load(folder + f"{place}/patches/mwanza_RGBN.npy"),
#         ]
#     ),
#     np.concatenate(
#         [
#             np.load(folder + f"{place}/patches/dar_SAR.npy"),
#             np.load(folder + f"{place}/patches/kampala_SAR.npy"),
#             np.load(folder + f"{place}/patches/kilimanjaro_SAR.npy"),
#             np.load(folder + f"{place}/patches/mwanza_SAR.npy"),
#         ]
#     ),
#     np.concatenate(
#         [
#             np.load(folder + f"{place}/patches/dar_RESWIR.npy"),
#             np.load(folder + f"{place}/patches/kampala_RESWIR.npy"),
#             np.load(folder + f"{place}/patches/kilimanjaro_RESWIR.npy"),
#             np.load(folder + f"{place}/patches/mwanza_RESWIR.npy"),
#         ]
#     ),
# ]

# y_train = np.concatenate(
#     [
#         np.load(folder + f"{place}/patches/dar_label_area.npy"),
#         np.load(folder + f"{place}/patches/kampala_label_area.npy"),
#         np.load(folder + f"{place}/patches/kilimanjaro_label_area.npy"),
#         np.load(folder + f"{place}/patches/mwanza_label_area.npy"),
#     ]
# )

model = tf.keras.models.load_model(
    outdir + "check_ghana_01_06", custom_objects={"tpe": tpe}
)

y_pred = model.predict(x_train, verbose=1, batch_size=512)
y_true = y_train

y_pred_sum = (y_pred.astype("float32") + 1).sum(axis=(1, 2))[:, 0]
y_true_sum = (y_true.astype("float32") + 1).sum(axis=(1, 2))[:, 0]

dif = np.abs(y_pred_sum - y_true_sum)
taep = dif / y_true_sum

mask = taep < np.quantile(taep, 0.99)

import pdb

pdb.set_trace()

np.save(folder + "dojo/ghana_RGBN.npy", x_train[0][mask].astype("float32"))
np.save(folder + "dojo/ghana_SAR.npy", x_train[1][mask].astype("float32"))
np.save(folder + "dojo/ghana_RESWIR.npy", x_train[2][mask].astype("float32"))
np.save(folder + "dojo/ghana_label_area.npy", y_train[mask].astype("float32"))
