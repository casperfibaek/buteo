yellow_follow = "C:/Users/caspe/Desktop/buteo/"
import sys

sys.path.append(yellow_follow)

import os
import time
import numpy as np
import datetime

import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras import Model, Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    AveragePooling2D,
    Conv2DTranspose,
    Concatenate,
)
from tensorflow.keras.callbacks import (
    LearningRateScheduler,
    ModelCheckpoint,
    TensorBoard,
)
from tensorflow.keras import mixed_precision
from buteo.machine_learning.ml_utils import load_mish, create_step_decay
from buteo.utils import timing

np.set_printoptions(suppress=True)
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
mixed_precision.set_global_policy("mixed_float16")


folder = "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/data/ghana/"

load_mish()

start = time.time()


def reduction_block(
    inputs,
    size=64,
    activation="Mish",
    kernel_initializer="glorot_normal",
    name=None,
):
    track1 = AveragePooling2D(
        pool_size=(2, 2),
        padding="same",
        name=name + "_reduction_t1",
    )(inputs)
    track2 = Conv2D(
        size,
        kernel_size=3,
        padding="same",
        strides=(2, 2),
        activation=activation,
        kernel_initializer=kernel_initializer,
        name=name + "_reduction_t2",
    )(inputs)
    track3 = Conv2D(
        size - 16,
        kernel_size=1,
        padding="same",
        strides=(1, 1),
        activation=activation,
        kernel_initializer=kernel_initializer,
        name=name + "_track3_1",
    )(inputs)
    track3 = Conv2D(
        size - 8,
        kernel_size=3,
        padding="same",
        strides=(1, 1),
        activation=activation,
        kernel_initializer=kernel_initializer,
        name=name + "_track3_2",
    )(track3)
    track3 = Conv2D(
        size,
        kernel_size=3,
        padding="same",
        strides=(2, 2),
        activation=activation,
        kernel_initializer=kernel_initializer,
        name=name + "_track3_3",
    )(track3)

    return Concatenate(name=name + "_concatenate")(
        [
            track1,
            track2,
            track3,
        ]
    )


def expansion_block(
    inputs, size=64, activation="Mish", kernel_initializer="glorot_normal", name=None
):
    track1 = Conv2DTranspose(
        size,
        kernel_size=3,
        strides=(2, 2),
        kernel_initializer=kernel_initializer,
        activation=activation,
        padding="same",
        name=name + "_track1",
    )(inputs)

    return track1


def inception_block(
    inputs, size=64, activation="Mish", kernel_initializer="glorot_normal", name=None
):
    track1 = MaxPooling2D(
        pool_size=2,
        strides=1,
        padding="same",
        name=name + "_track1",
    )(inputs)
    track2 = Conv2D(
        size,
        kernel_size=1,
        padding="same",
        strides=(1, 1),
        activation=activation,
        kernel_initializer=kernel_initializer,
        name=name + "_track2",
    )(inputs)
    track3 = Conv2D(
        size - 8,
        kernel_size=1,
        padding="same",
        strides=(1, 1),
        activation=activation,
        kernel_initializer=kernel_initializer,
        name=name + "_track3_1",
    )(inputs)
    track3 = Conv2D(
        size,
        kernel_size=3,
        padding="same",
        strides=(1, 1),
        activation=activation,
        kernel_initializer=kernel_initializer,
        name=name + "_track3_2",
    )(track3)
    track4 = Conv2D(
        size - 16,
        kernel_size=1,
        padding="same",
        strides=(1, 1),
        activation=activation,
        kernel_initializer=kernel_initializer,
        name=name + "_track4_1",
    )(inputs)
    track4 = Conv2D(
        size - 8,
        kernel_size=3,
        padding="same",
        strides=(1, 1),
        activation=activation,
        kernel_initializer=kernel_initializer,
        name=name + "_track4_2",
    )(track4)
    track4 = Conv2D(
        size,
        kernel_size=3,
        padding="same",
        strides=(1, 1),
        activation=activation,
        kernel_initializer=kernel_initializer,
        name=name + "_track4_3",
    )(track4)

    return Concatenate(name=name + "_concatenate")(
        [
            track1,
            track2,
            track3,
            track4,
        ]
    )


version = 1

x_train_rgbn = np.load(folder + "raster/patches/001_RGBN.npy")
x_train_swir = np.load(folder + "raster/patches/001_SWIR.npy")
x_train_sar = np.load(folder + "raster/patches/001_SAR.npy")

y_train = np.load(folder + "raster/patches/001_LABEL_AREA.npy")

mask = (y_train.sum(axis=(1, 2)) > 100)[:, 0]

x_train_rgbn = x_train_rgbn[mask]
x_train_swir = x_train_swir[mask]
x_train_sar = x_train_sar[mask]
y_train = y_train[mask]


dk_model = tf.keras.models.load_model(folder + "/models/denmark_base_01")
for idx, layer in enumerate(dk_model.layers):
    if idx > (len(dk_model.layers) - 1) - 6:
        continue
    layer.trainable = False

model = Conv2D(
    64,
    kernel_size=3,
    padding="same",
    strides=(1, 1),
    activation="Mish",
    kernel_initializer="glorot_normal",
    name="new_conv2_0",
)(dk_model.layers[-3].output)

model = Conv2D(
    64,
    kernel_size=3,
    padding="same",
    strides=(1, 1),
    activation="Mish",
    kernel_initializer="glorot_normal",
    name="new_conv2_1",
)(model)

model = inception_block(model, name="new_inception_block_01")
model = inception_block(model, name="new_inception_block_02")

model = reduction_block(model, name="new_reduction_block_01")

model = inception_block(model, name="new_inception_block_03")
model = inception_block(model, name="new_inception_block_04")

model = reduction_block(model, name="new_reduction_block_02")

model = inception_block(model, name="new_inception_block_05")
model = inception_block(model, name="new_inception_block_06")

model = expansion_block(model, name="new_expansion_block_01")

model = inception_block(model, name="new_inception_block_07")
model = inception_block(model, name="new_inception_block_08")

model = expansion_block(model, name="new_expansion_block_02")

model = inception_block(model, name="new_inception_block_09")
model = inception_block(model, name="new_inception_block_10")
model = inception_block(model, name="new_inception_block_11")
model = inception_block(model, name="new_inception_block_12")
model = inception_block(model, name="new_inception_block_13")
model = inception_block(model, name="new_inception_block_14")

model = Conv2D(
    64,
    kernel_size=1,
    padding="same",
    activation="Mish",
    kernel_initializer="glorot_normal",
    name="new_conv2_2_pca",
)(model)

output = Conv2D(
    1,
    kernel_size=3,
    padding="same",
    activation="relu",
    kernel_initializer="glorot_normal",
    name="output",
)(model)

model = Model(inputs=dk_model.inputs, outputs=output)

learning_rate = 0.0001

optimizer = tf.keras.optimizers.Adam(
    learning_rate=learning_rate,
    epsilon=1e-07,
    amsgrad=False,
    name="Adam",
)

model.compile(
    optimizer=optimizer,
    loss="mse",
    metrics=["mse", "mae"],
)


with tf.device("/device:GPU:0"):
    model.fit(
        x=[x_train_rgbn, x_train_swir, x_train_sar],
        y=y_train,
        validation_split=0.2,
        shuffle=True,
        epochs=25,
        initial_epoch=0,
        verbose=1,
        batch_size=4,
        use_multiprocessing=True,
        workers=0,
        callbacks=[
            LearningRateScheduler(
                create_step_decay(
                    learning_rate=learning_rate,
                    drop_rate=0.8,
                    epochs_per_drop=3,
                )
            ),
        ],
    )

    model.save(folder + "models/ghana_02", save_format="tf")

timing(start)
