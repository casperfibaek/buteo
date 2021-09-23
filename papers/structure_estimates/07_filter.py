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

folder = "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/data/denmark/"
model_dir = folder + f"models/"

x_train = [
    np.load(folder + "patches/RGBN.npy"),
    np.load(folder + "patches/SAR.npy"),
    np.load(folder + "patches/RESWIR.npy"),
]

y_train = np.load(folder + "patches/label_area.npy")
volume = np.load(folder + "patches/label_volume.npy")
people = np.load(folder + "patches/label_people.npy")

# model = tf.keras.models.load_model(
#     model_dir + "big_model_dk_08_54", custom_objects={"tpe": tpe}
# )

# y_pred = model.predict(x_train, verbose=1, batch_size=512)
y_pred = np.load(folder + "patches/y_pred.npy")

# np.save(folder + "patches/y_pred.npy", y_pred)

y_true = y_train

y_pred_sum = (y_pred.astype("float32") + 1).sum(axis=(1, 2))[:, 0]
y_true_sum = (y_true.astype("float32") + 1).sum(axis=(1, 2))[:, 0]

dif = np.abs(y_pred_sum - y_true_sum)
taep = dif / y_true_sum

mask = taep < np.quantile(taep, 0.99)

# np.save(folder + "patches/extra_RGBN.npy", x_train[0][mask])
# np.save(folder + "patches/extra_SAR.npy", x_train[1][mask])
# np.save(folder + "patches/extra_RESWIR.npy", x_train[2][mask])
# np.save(folder + "patches/extra_label_area.npy", y_train[mask])
np.save(folder + "patches/extra_label_volume.npy", volume[mask])
np.save(folder + "patches/extra_label_people.npy", people[mask])
