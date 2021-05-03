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


def define_model(
    shape_10m,
    shape_20m,
    name,
    activation="swish",
    # filter_size=32,
    filter_size=[32, 48, 64],
    skip_connections=True,
    max_pooling=False,
    kernel_initializer="glorot_normal",
):
    input_10m = Input(shape=shape_10m, name=name + "_10m")
    input_20m = Input(shape=shape_20m, name=name + "_20m")

    # -------------- 10m -------------------
    model_10m = Conv2D(
        filter_size[0],
        kernel_size=3,
        padding="same",
        activation=activation,
        kernel_initializer=kernel_initializer,
    )(input_10m)

    if max_pooling:
        model_10m = MaxPooling2D(padding="same")(model_10m)
    else:
        model_10m = AveragePooling2D(padding="same")(model_10m)

    model_10m_skip_64 = Conv2D(
        filter_size[1],
        kernel_size=3,
        padding="same",
        activation=activation,
        kernel_initializer=kernel_initializer,
    )(model_10m)

    if max_pooling:
        model_10m = MaxPooling2D(padding="same")(model_10m_skip_64)
    else:
        model_10m = AveragePooling2D(padding="same")(model_10m_skip_64)

    model_10m = Conv2D(
        filter_size[2],
        kernel_size=3,
        padding="same",
        activation=activation,
        kernel_initializer=kernel_initializer,
    )(model_10m)

    model_10m = Conv2DTranspose(
        filter_size[2],
        kernel_size=3,
        strides=(2, 2),
        activation=activation,
        kernel_initializer=kernel_initializer,
        padding="same",
    )(model_10m)

    if skip_connections:
        model_10m = Concatenate()([model_10m, model_10m_skip_64])

    # -------------- 20m -------------------
    model_20m_skip_64 = Conv2D(
        filter_size[1],
        kernel_size=3,
        padding="same",
        activation=activation,
        kernel_initializer=kernel_initializer,
    )(input_20m)

    if max_pooling:
        model_20m = MaxPooling2D(padding="same")(model_20m_skip_64)
    else:
        model_20m = AveragePooling2D(padding="same")(model_20m_skip_64)

    model_20m = Conv2D(
        filter_size[2],
        kernel_size=3,
        padding="same",
        activation=activation,
        kernel_initializer=kernel_initializer,
    )(model_20m)

    model_20m = Conv2DTranspose(
        filter_size[2],
        kernel_size=3,
        strides=(2, 2),
        activation=activation,
        kernel_initializer=kernel_initializer,
        padding="same",
    )(model_20m)

    if skip_connections:
        model_20m = Concatenate()([model_20m, model_20m_skip_64])

    # -------------- merged -------------------
    model = Concatenate()([model_10m, model_20m])

    model = Conv2D(
        filter_size[2],
        kernel_size=3,
        padding="same",
        activation=activation,
        kernel_initializer=kernel_initializer,
    )(model)

    model = Conv2DTranspose(
        filter_size[2],
        kernel_size=3,
        strides=(2, 2),
        activation=activation,
        kernel_initializer=kernel_initializer,
        padding="same",
    )(model)

    model = Conv2D(
        filter_size[2],
        kernel_size=3,
        padding="same",
        activation=activation,
        kernel_initializer=kernel_initializer,
    )(model)

    output = Conv2D(
        1,
        kernel_size=3,
        padding="same",
        activation="relu",
        kernel_initializer=kernel_initializer,
    )(model)

    return Model(inputs=[input_10m, input_20m], outputs=output)


lr = 0.001
bs = 32
epochs = 10


def step_decay(epoch):
    initial_lrate = lr
    drop = 0.5
    epochs_drop = 3
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate


with tf.device("/device:GPU:0"):

    model = define_model(
        x_10m_train.shape[1:],
        x_20m_train.shape[1:],
        "Upsampling",
        kernel_initializer="glorot_normal",
    )

    model.compile(
        optimizer=Adam(learning_rate=lr, name="Adam"),
        loss="log_cosh",
        metrics=["mse", "mae"],
    )

    print(model.summary())

    model.fit(
        x=[x_10m_train, x_20m_train],
        y=y_train,
        epochs=10,
        verbose=1,
        batch_size=bs,
        validation_split=0.2,
        callbacks=[
            LearningRateScheduler(step_decay),
            EarlyStopping(
                monitor="val_loss",
                patience=10,
                min_delta=0.1,
                restore_best_weights=True,
            ),
        ],
        use_multiprocessing=True,
        workers=0,
        shuffle=True,
    )

    print(f"Batch_size: {str(bs)}")
    loss, mse, mae = model.evaluate(
        x=[x_10m_test, x_20m_test],
        y=y_test,
        verbose=1,
        batch_size=bs,
        use_multiprocessing=True,
    )

    print(f"Mean Square Error:      {round(mse, 3)}")
    print(f"Mean Absolute Error:    {round(mae, 3)}")
    # print(f"log_cosh:               {round(loss, 3)}")
    print("")


import pdb

pdb.set_trace()

import h5py

model.save(folder + "upsampling_model_10epochs_norm.h5")

