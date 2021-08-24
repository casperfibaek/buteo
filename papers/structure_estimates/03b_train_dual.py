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
from buteo.machine_learning.ml_utils import create_step_decay
from buteo.utils import timing

from model_dual_down import model_dual_down

np.set_printoptions(suppress=True)
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
mixed_precision.set_global_policy("mixed_float16")

folder = "C:/Users/caspe/Desktop/paper_2_Structural_Volume/data/"
outdir = folder + "tmp/"

model_name = "rgbn_swir"

for idx, val in enumerate(["area", "volume", "people"]):
    label = val
    x_train_rgbn = np.load(folder + "patches/RGBN.npy")
    x_train_swir = np.load(folder + "patches/RESWIR.npy")

    y_train = np.load(folder + f"patches/label_{label}.npy")

    area_limit = 500
    tile_limit = 15000

    mask = np.load(folder + f"patches/label_area.npy")
    mask = (mask.sum(axis=(1, 2)) > area_limit)[:, 0]

    x_train_rgbn = x_train_rgbn[mask]
    x_train_swir = x_train_swir[mask]
    y_train = y_train[mask]

    x_train_rgbn = x_train_rgbn[:tile_limit]
    x_train_swir = x_train_swir[:tile_limit]
    y_train = y_train[:tile_limit]

    if label == "people":
        y_train = y_train / 10
    elif label == "area":
        y_train = y_train / 100
    elif label == "volume":
        y_train = y_train / 1000
    else:
        raise Exception("Wrong label used.")

    with tf.device("/device:GPU:0"):
        lr = 0.001
        epochs = [15, 15, 5]
        bs = [64, 32, 16]

        model = model_dual_down(
            x_train_rgbn.shape[1:],
            x_train_swir.shape[1:],
            kernel_initializer="glorot_normal",
            activation="relu",
            inception_blocks=2,
            name=model_name,
        )

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr,
            epsilon=1e-07,
            amsgrad=False,
            name="Adam",
        )

        model.compile(
            optimizer=optimizer,
            loss="mse",
            metrics=["mse", "mae"],
        )

        if idx == 0:
            print(model.summary())

        model_checkpoint_callback = ModelCheckpoint(
            filepath=f"tmp/{model_name}_" + "{epoch:02d}",
            save_best_only=True,
        )

        start = time.time()

        for phase in range(len(bs)):
            use_epoch = np.cumsum(epochs)[phase]
            use_bs = bs[phase]
            initial_epoch = np.cumsum(epochs)[phase - 1] if phase != 0 else 0

            model.fit(
                x=[
                    x_train_rgbn,
                    x_train_swir,
                ],
                y=y_train,
                validation_split=0.2,
                shuffle=True,
                epochs=use_epoch,
                initial_epoch=initial_epoch,
                verbose=1,
                batch_size=use_bs,
                use_multiprocessing=True,
                workers=0,
                callbacks=[
                    LearningRateScheduler(
                        create_step_decay(
                            learning_rate=lr,
                            drop_rate=0.65,
                            epochs_per_drop=5,
                        )
                    ),
                    model_checkpoint_callback,
                    EarlyStopping(monitor="val_loss", patience=3, min_delta=0.05),
                ],
            )

    timing(start)
