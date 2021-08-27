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
from buteo.machine_learning.ml_utils import create_step_decay, mse
from buteo.utils import timing

from model_single import model_single
from model_dual_down import model_dual_down
from model_trio_down import model_trio_down
from training_utils import get_layer

np.set_printoptions(suppress=True)
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
mixed_precision.set_global_policy("mixed_float16")

for tile_size in [
    # "128x128",
    # "64x64",
    "32x32",
    # "16x16",
]:

    folder = "C:/Users/caspe/Desktop/paper_2_Structural_Volume/data/"
    outdir = folder + f"tmp/"

    for model_name in [
        # "RGB",
        # "RGBN",
        # "RGBN_RE",
        # "RGBN_SWIR",
        # "RGBN_RESWIR",
        # "VVa",
        # "VVa_VHa",
        # "VVa_VVd",
        # "VVa_COHa",
        # "VVa_VHa_COHa",
        # "VVa_VVd_COHa_COHd",
        # "RGBN_RESWIR_VVa_VHa",
        "RGBN_RESWIR_VVa_VVd_COHa_COHd",
    ]:
        timings = []

        for idx, val in enumerate(
            [
                "area",
                "volume",
                "people",
            ]
        ):
            print(f"Processing: {model_name} - {val}")

            label = val
            x_train = get_layer(folder, model_name, tile_size=tile_size)
            y_train = get_layer(folder, label, tile_size=tile_size)

            area_limit = 250
            tile_limit = 15000 * 4

            mask = get_layer(folder, "area", tile_size=tile_size)
            mask = (mask.sum(axis=(1, 2)) > area_limit)[:, 0]

            if model_name in [
                "RGBN_RE",
                "RGBN_SWIR",
                "RGBN_RESWIR",
                "RGBN_RESWIR_VVa_VHa",
                "RGBN_RESWIR_VVa_VVd_COHa_COHd",
            ]:
                x_train_holder = []
                for idx in range(len(x_train)):
                    x_train_masked = x_train[idx][mask]
                    x_train_limited = x_train_masked[:tile_limit]
                    x_train_holder.append(x_train_limited)
                x_train_reduced = x_train_holder
            else:
                x_train_reduced = x_train_reduced[mask]
                x_train_reduced = x_train_reduced[:tile_limit]

            y_train_reduced = y_train[mask]
            y_train_reduced = y_train_reduced[:tile_limit]

            if label == "area":
                lr = 0.001
                min_delta = 0.05
                y_train = y_train_reduced
                x_train = x_train_reduced
            elif label == "volume":
                lr = 0.0001
                min_delta = 0.5
                y_train = y_train_reduced
                x_train = x_train_reduced
            elif label == "people":
                lr = 0.0001
                min_delta = 0.25

                # Ensures the model does not optimize to zero.
                y_train = y_train * 100
            else:
                raise Exception("Wrong label used.")

            with tf.device("/device:GPU:0"):
                # if tile_size == "128x128":
                #     bs = [16, 8]  # for 128x126
                # elif tile_size == "64x64":
                #     bs = [64, 32]  # for 64x64
                # elif tile_size == "32x32":
                #     bs = [256, 128]  # for 32x32
                # elif tile_size == "16x16":
                #     bs = [1024, 512]  # for 16x16

                # for big_model
                epochs = [25, 25, 10]
                bs = [256, 128, 64]

                # for testing
                # epochs = [20, 5]
                # bs = [64, 32]

                # for big_model
                inception_blocks = 3
                # inception_blocks = 3  # model model
                activation = "relu"
                initializer = "glorot_normal"

                # Triple Model
                if model_name in [
                    "RGBN_RESWIR_VVa_VHa",
                    "RGBN_RESWIR_VVa_VVd_COHa_COHd",
                ]:
                    model = model_trio_down(
                        x_train[0].shape[1:],
                        x_train[1].shape[1:],
                        x_train[2].shape[1:],
                        kernel_initializer=initializer,
                        activation=activation,
                        inception_blocks=inception_blocks,
                        name=f"{model_name.lower()}_{label}",
                    )

                # Dual model
                elif model_name in [
                    "RGBN_RE",
                    "RGBN_SWIR",
                    "RGBN_RESWIR",
                ]:
                    model = model_dual_down(
                        x_train[0].shape[1:],
                        x_train[1].shape[1:],
                        kernel_initializer=initializer,
                        activation=activation,
                        inception_blocks=inception_blocks,
                        name=f"{model_name.lower()}_{label}",
                    )

                # Single Model
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

                # Cast two 64 bit loss function for area.
                # Ensures that the loss doesn't go NaN
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
                    area_model = f"{outdir}{model_name.lower()}_area"
                    donor_model = tf.keras.models.load_model(area_model)
                    model.set_weights(donor_model.get_weights())
                elif label == "people":
                    volume_model = f"{outdir}{model_name.lower()}_volume"
                    donor_model = tf.keras.models.load_model(volume_model)
                    model.set_weights(donor_model.get_weights())

                donor_model = None

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
                                    drop_rate=0.80,
                                    epochs_per_drop=5,
                                )
                            ),
                            ModelCheckpoint(
                                filepath=f"{outdir}{model_name.lower()}_{label}_"
                                + "{epoch:02d}",
                                save_best_only=True,
                            ),
                            EarlyStopping(
                                monitor="val_loss",
                                patience=3,
                                min_delta=min_delta,
                                restore_best_weights=True,
                            ),
                        ],
                    )

                model.save(f"{outdir}{model_name.lower()}_{label}")

            timings.append([label, timing(start)])

        with open(folder + f"logs/{model_name}.txt", "w") as f:
            for t in timings:
                f.write(f"{t[0]} - {t[1]}\n")
