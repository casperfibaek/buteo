yellow_follow = "C:/Users/caspe/Desktop/buteo/"
import sys

sys.path.append(yellow_follow)

import os
import math
import numpy as np

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Conv2DTranspose,
    Concatenate,
    AveragePooling2D,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras import mixed_precision

np.set_printoptions(suppress=True)
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

from buteo.machine_learning.ml_utils import load_mish

load_mish()

policy = tf.keras.mixed_precision.experimental.Policy("mixed_float16")
tf.keras.mixed_precision.experimental.set_policy(policy)

folder = "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/upsampling/"

out_dir = folder + "out/"

merge_10 = [
    folder + "out/" + "32UMF_B02_10m_patches.npy",
    folder + "out/" + "32UMF_B03_10m_patches.npy",
    folder + "out/" + "32UMF_B04_10m_patches.npy",
]

target_20m = folder + "out/" + "32UMF_B08_20m_patches.npy"

target_10m = folder + "out/" + "32UMF_B08_10m_patches.npy"

band_11 = folder + "out/" + "32UMF_B11_20m.tif"

stack = []
for arr in merge_10:
    loaded = np.load(arr)
    stack.append((loaded / loaded.max()) * 1000)

stacked = np.stack(stack, axis=3)[:, :, :, :, 0]
stack = None

target_10m = np.load(target_10m)
target_10m = (target_10m / target_10m.max()) * 1000

target_20m = np.load(target_20m)
target_20m = (target_20m / target_20m.max()) * 1000

y = target_10m

x_10m = stacked
x_20m = target_20m

# Shuffle the training dataset
np.random.seed(42)
shuffle_mask = np.random.permutation(len(y))
y = y[shuffle_mask]
x_10m = x_10m[shuffle_mask]
x_20m = x_20m[shuffle_mask]

split = int(y.shape[0] * 0.75)

y_train = y[0:split]
y_test = y[split:]

x_10m_train = x_10m[0:split]
x_10m_test = x_10m[split:]

x_20m_train = x_20m[0:split]
x_20m_test = x_20m[split:]
