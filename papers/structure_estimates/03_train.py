yellow_follow = "C:/Users/caspe/Desktop/buteo/"
from papers.structure_estimates.model_trio_end import model_trio_down
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
from model_dual_down import model_dual_down
from model_trio_down import model_dual_down
from model_trio_end import model_trio_end
from training_utils import get_layer

np.set_printoptions(suppress=True)
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
mixed_precision.set_global_policy("mixed_float16")

folder = "C:/Users/caspe/Desktop/paper_2_Structural_Volume/data/"
outdir = folder + "tmp/"


for model_name in [
    # "rgb",
    # "rgbn",
    # "VVa",
    # "VVa_VHa",
    # "VVa_VVd",
    # "VVa_COHa",
    # "VVa_VVd_COHa_COHd",
    # "VVa_VHa_COHa",
    # "RGBN_SWIR",
    # "RGBN_RE",
    # "RGBN_RESWIR",
    "merge_test_join",
    "merge_test_down",
    "merge_test_end",
]:
    timings = []

    for idx, val in enumerate(["area", "volume", "people"]):
        label = val
        x_train = get_layer(folder, model_name)
        y_train = np.load(folder + f"patches/label_{label}.npy")

        area_limit = 250
        tile_limit = 15000

        mask = np.load(folder + f"patches/label_area.npy")
        mask = (mask.sum(axis=(1, 2)) > area_limit)[:, 0]

        if model_name in [
            "RGBN_SWIR",
            "RGBN_RE",
            "RGBN_RESWIR",
            "merge_test_join",
            "merge_test_down",
            "merge_test_end",
        ]:
            for idx in range(len(x_train)):
                x_train[idx] = x_train[idx][mask]
                x_train[idx] = x_train[idx][:tile_limit]
        else:
            y_train = y_train[mask]
            y_train = y_train[:tile_limit]

        if label == "area":
            lr = 0.001
            min_delta = 0.05
        elif label == "volume":
            lr = 0.0001
            min_delta = 0.5
            continue
        elif label == "people":
            lr = 0.00001
            min_delta = 0.025
            continue
        else:
            raise Exception("Wrong label used.")

        with tf.device("/device:GPU:0"):
            # epochs = [15, 15, 5]
            epochs = [15]
            bs = [64, 32, 16]
            inception_blocks = 2
            activation = "relu"
            initializer = "glorot_normal"

            if model_name in ["RGBN_SWIR", "RGBN_RE", "RGBN_RESWIR", "merge_test_join"]:
                model = model_dual_down(
                    x_train[0].shape[1:],
                    x_train[1].shape[1:],
                    kernel_initializer=initializer,
                    activation=activation,
                    inception_blocks=inception_blocks,
                    name=f"{model_name.lower()}_{label}",
                )
            elif model_name in ["merge_test_down"]:
                model = model_trio_down(
                    x_train[0].shape[1:],
                    x_train[1].shape[1:],
                    x_train[2].shape[1:],
                    kernel_initializer=initializer,
                    activation=activation,
                    inception_blocks=inception_blocks,
                    name=f"{model_name.lower()}_{label}",
                )
            elif model_name in ["merge_test_end"]:
                model = model_trio_end(
                    x_train[0].shape[1:],
                    x_train[1].shape[1:],
                    x_train[2].shape[1:],
                    kernel_initializer=initializer,
                    activation=activation,
                    inception_blocks=inception_blocks,
                    name=f"{model_name.lower()}_{label}",
                )
            else:
                model = model_single(
                    x_train.shape[1:],
                    kernel_initializer=initializer,
                    activation=activation,
                    inception_blocks=inception_blocks,
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
        for t in timings:
            f.write(f"{t[0]} - {t[1]}\n")
