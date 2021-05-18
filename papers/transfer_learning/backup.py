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
    BatchNormalization,
)
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras import mixed_precision
from buteo.machine_learning.ml_utils import load_mish, mean_standard, mad_standard

import matplotlib

matplotlib.use("Qt5Agg")

from matplotlib import pyplot as plt


def plot_figures(figures, nrows=1, ncols=1, vmin=[0, 0], vmax=[1, 1000]):
    """Plot a dictionary of figures.

    Parameters
    ----------
    figures : <title, figure> dictionary
    ncols : number of columns of subplots wanted in the display
    nrows : number of rows of subplots wanted in the figure
    """

    _fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows)
    for idx, title in zip(range(len(figures)), figures):
        axeslist.ravel()[idx].imshow(figures[title], vmin=vmin[idx], vmax=vmax[idx])
        axeslist.ravel()[idx].set_title(title)
        axeslist.ravel()[idx].set_axis_off()

    plt.tight_layout()


load_mish()

np.set_printoptions(suppress=True)
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
mixed_precision.set_global_policy("mixed_float16")

folder = "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/data/denmark/"

optical_2020_10m = [
    # (np.load(folder + "2021_B02_10m.npy") / 3000).astype("float32"),
    # (np.load(folder + "2021_B03_10m.npy") / 3500).astype("float32"),
    (np.load(folder + "2021_B04_10m.npy") / 4000).astype("float32"),
    (np.load(folder + "2021_B08_10m.npy") / 5500).astype("float32"),
]

labels = np.load(folder + "area.npy")

training_data = np.stack(optical_2020_10m, axis=3)[:, :, :, :, 0]
optical_2020_10m = None

# Shuffle the training dataset
np.random.seed(42)
shuffle_mask = np.random.permutation(len(labels))
labels = labels[shuffle_mask]

training_data = training_data[shuffle_mask]

split = int(labels.shape[0] * 0.75)

labels_train = labels[0:split]
labels_test = labels[split:]
labels = None

training_data_train = training_data[0:split]
training_data_test = training_data[split:]
training_data = None

np.save(folder + "training_data_train.npy", training_data_train)
np.save(folder + "training_data_test.npy", training_data_test)

np.save(folder + "labels_train.npy", labels_train)
np.save(folder + "labels_test.npy", labels_test)

# training_data_train = np.load(folder + "training_data_train.npy")
# training_data_test = np.load(folder + "training_data_test.npy")
# labels_train = np.load(folder + "labels_train.npy")
# labels_test = np.load(folder + "labels_test.npy")

# mask = labels_train.sum(axis=(1, 2, 3)) > 25000
# training_data_train = training_data_train[mask]
# labels_train = labels_train[mask]


import pdb

pdb.set_trace()


def define_model(
    shape_10m_optical,
    name,
    activation="Mish",
    kernel_initializer="glorot_normal",
    sizes=[32, 48, 64],
):
    model_input = Input(shape=shape_10m_optical, name=name)
    model_skip1 = Conv2D(
        sizes[0],
        kernel_size=3,
        padding="same",
        activation=activation,
        kernel_initializer=kernel_initializer,
    )(model_input)
    model = Conv2D(
        sizes[0],
        kernel_size=3,
        padding="same",
        activation=activation,
        kernel_initializer=kernel_initializer,
    )(model_skip1)

    model = MaxPooling2D(pool_size=(2, 2))(model)

    model_skip2 = Conv2D(
        sizes[1],
        kernel_size=3,
        padding="same",
        activation=activation,
        kernel_initializer=kernel_initializer,
    )(model)
    model = Conv2D(
        sizes[1],
        kernel_size=3,
        padding="same",
        activation=activation,
        kernel_initializer=kernel_initializer,
    )(model_skip2)

    model = MaxPooling2D(pool_size=(2, 2))(model)

    model = Conv2D(
        sizes[2],
        kernel_size=3,
        padding="same",
        activation=activation,
        kernel_initializer=kernel_initializer,
    )(model)
    model = Conv2D(
        sizes[2],
        kernel_size=3,
        padding="same",
        activation=activation,
        kernel_initializer=kernel_initializer,
    )(model)

    model = Conv2DTranspose(
        sizes[1],
        kernel_size=3,
        strides=(2, 2),
        kernel_initializer=kernel_initializer,
        activation=activation,
        padding="same",
    )(model)
    model = Concatenate()([model_skip2, model])

    model = Conv2DTranspose(
        sizes[0],
        kernel_size=3,
        strides=(2, 2),
        kernel_initializer=kernel_initializer,
        activation=activation,
        padding="same",
    )(model)
    model = Concatenate()([model_skip1, model])

    model = Conv2D(
        sizes[2],
        kernel_size=3,
        padding="same",
        activation=activation,
        kernel_initializer=kernel_initializer,
    )(model)
    model = Conv2D(
        sizes[2],
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

    return Model(inputs=[model_input], outputs=output)


def create_model(
    optical_10m,
    name="investigating",
    kernel_initializer="normal",
    activation="relu",
    learning_rate=0.001,
):
    model = define_model(
        optical_10m.shape[1:],
        name,
        kernel_initializer=kernel_initializer,
        activation=activation,
    )

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

    return model


def step_decay(epoch):
    initial_lrate = lr
    drop = 0.5
    epochs_drop = 10
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate


with tf.device("/device:GPU:0"):
    lr = 0.001
    epochs = [10, 40]
    bs = [32, 16]

    model = create_model(
        training_data_train,
        kernel_initializer="glorot_normal",
        activation="relu",
        learning_rate=lr,
    )

    print(model.summary())

    model.fit(
        x=training_data_train,
        y=labels_train,
        epochs=epochs[0],
        verbose=1,
        batch_size=bs[0],
        validation_split=0.2,
        use_multiprocessing=True,
        workers=0,
        shuffle=True,
    )

    model.fit(
        x=training_data_train,
        y=labels_train,
        epochs=epochs[1],
        initial_epoch=10,
        verbose=1,
        batch_size=bs[1],
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
        x=training_data_test,
        y=labels_test,
        verbose=1,
        batch_size=bs[0],
        use_multiprocessing=True,
    )

    print(f"Mean Square Error:      {round(mse, 3)}")
    print(f"Mean Absolute Error:    {round(mae, 3)}")
    print("")


model.save(folder + "models/denmark_01", save_format="tf")
import pdb

pdb.set_trace()
