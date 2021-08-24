yellow_follow = "C:/Users/caspe/Desktop/buteo/"
import sys

sys.path.append(yellow_follow)

import os
import time
from glob import glob
import numpy as np

import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.callbacks import (
    LearningRateScheduler,
    ModelCheckpoint,
    EarlyStopping,
)
from buteo.machine_learning.ml_utils import create_step_decay, mse
from buteo.utils import timing

from model_single import model_single

np.set_printoptions(suppress=True)
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
mixed_precision.set_global_policy("mixed_float16")

folder = "C:/Users/caspe/Desktop/paper_2_Structural_Volume/data/"
outdir = folder + "tmp/"

layers = {
    "rgb": np.load(folder + "patches/RGB.npy"),
    "rgbn": np.load(folder + "patches/RGBN.npy"),
    "VVa": np.load(folder + "patches/vv_asc_v2.npy"),
    "VVa_VHa": np.concatenate(
        [
            np.load(folder + "patches/vv_asc_v2.npy"),
            np.load(folder + "patches/vh_asc_v2.npy"),
        ],
        axis=3,
    ),
    "VVa_VVd": np.concatenate(
        [
            np.load(folder + "patches/vv_asc.npy"),
            np.load(folder + "patches/vv_desc.npy"),
        ],
        axis=3,
    ),
    "VVa_COHa": np.concatenate(
        [
            np.load(folder + "patches/vv_asc.npy"),
            np.load(folder + "patches/coh_asc.npy"),
        ],
        axis=3,
    ),
    "VVa_VVd_COHa_COHd": np.concatenate(
        [
            np.load(folder + "patches/vv_asc.npy"),
            np.load(folder + "patches/vv_desc.npy"),
            np.load(folder + "patches/coh_asc.npy"),
            np.load(folder + "patches/coh_desc.npy"),
        ],
        axis=3,
    ),
    "VVa_VHa_COHa": np.concatenate(
        [
            np.load(folder + "patches/vv_asc_v2.npy"),
            np.load(folder + "patches/vh_asc_v2.npy"),
            np.load(folder + "patches/coh_asc.npy"),
        ],
        axis=3,
    ),
}

for model_name in [
    "rgb",
    "rgbn",
    "VVa",
    "VVa_VHa",
    "VVa_VVd",
    "VVa_COHa",
    "VVa_VVd_COHa_COHd",
    "VVa_VHa_COHa",
]:
    timings = []

    for idx, val in enumerate(["area", "volume", "people"]):
        label = val
        x_train = layers[model_name]
        y_train = np.load(folder + f"patches/label_{label}.npy")

        area_limit = 250
        tile_limit = 15000

        mask = np.load(folder + f"patches/label_area.npy")
        mask = (mask.sum(axis=(1, 2)) > area_limit)[:, 0]

        x_train = x_train[mask]
        y_train = y_train[mask]

        x_train = x_train[:tile_limit]
        y_train = y_train[:tile_limit]

        if label == "area":
            lr = 0.001
            min_delta = 0.05
        elif label == "volume":
            lr = 0.0001
            min_delta = 0.5
        elif label == "people":
            lr = 0.00001
            min_delta = 0.025
        else:
            raise Exception("Wrong label used.")

        with tf.device("/device:GPU:0"):
            epochs = [15, 15, 5]
            bs = [64, 32, 16]

            model = model_single(
                x_train.shape[1:],
                kernel_initializer="glorot_normal",
                activation="relu",
                inception_blocks=2,
                name=f"{model_name.lower()}_{label}",
            )

            optimizer = tf.keras.optimizers.Adam(
                learning_rate=lr,
                epsilon=1e-07,
                amsgrad=False,
                name="Adam",
            )

            loss = "mse"
            if label != "area":
                loss = mse

            model.compile(
                optimizer=optimizer,
                loss=loss,
                metrics=["mse", "mae"],
            )

            if label == "area":
                print(model.summary())
            elif label == "volume":
                area_model = sorted(glob(outdir + f"{model_name.lower()}_area_*"))[-1]
                donor_model = tf.keras.models.load_model(area_model)
                model.set_weights(donor_model.get_weights())

            elif label == "people":
                area_model = sorted(glob(outdir + f"{model_name.lower()}_volume_*"))[-1]
                donor_model = tf.keras.models.load_model(area_model)

                model.set_weights(donor_model.get_weights())

            donor_model = None

            model_checkpoint_callback = ModelCheckpoint(
                filepath=f"{outdir}{model_name.lower()}_{label}_" + "{epoch:02d}",
                save_best_only=True,
            )

            start = time.time()

            for phase in range(len(bs)):
                use_epoch = np.cumsum(epochs)[phase]
                use_bs = bs[phase]
                initial_epoch = np.cumsum(epochs)[phase - 1] if phase != 0 else 0

                model.fit(
                    x=x_train,
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
                                drop_rate=0.8,
                                epochs_per_drop=3,
                            )
                        ),
                        model_checkpoint_callback,
                        EarlyStopping(
                            monitor="val_loss", patience=3, min_delta=min_delta
                        ),
                    ],
                )

        timings.append([label, timing(start)])

    with open(folder + f"logs/{model_name}.txt", "w") as f:
        for time in timings:
            f.write(f"{time[0]} - {time[1]}\n")
