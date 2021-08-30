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

from model_trio_down import model_trio_down

np.set_printoptions(suppress=True)
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
mixed_precision.set_global_policy("mixed_float16")

model_name = "big_model_dk_01"

folder = "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/data/denmark/"
outdir = folder + f"models/"

x_train = [
    np.load(folder + "patches/RGBN.npy"),
    np.load(folder + "patches/SAR.npy"),
    np.load(folder + "patches/RESWIR.npy"),
]
y_train = np.load(folder + "patches/label_area.npy")

lr = 0.001
min_delta = 0.05

with tf.device("/device:GPU:0"):
    epochs = [25, 25, 10]
    bs = [256, 128, 64]
    inception_blocks = 3
    activation = "relu"
    initializer = "glorot_normal"

    model = model_trio_down(
        x_train[0].shape[1:],
        x_train[1].shape[1:],
        x_train[2].shape[1:],
        kernel_initializer=initializer,
        activation=activation,
        inception_blocks=inception_blocks,
        name=f"{model_name.lower()}",
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr,
        epsilon=1e-07,
        amsgrad=False,
        name="Adam",
    )

    model.compile(
        optimizer=optimizer,
        loss=mse,
        metrics=["mse", "mae"],
    )

    print(model.summary())

    start = time.time()

    for phase in range(len(bs)):
        use_epoch = np.cumsum(epochs)[phase]
        use_bs = bs[phase]
        initial_epoch = np.cumsum(epochs)[phase - 1] if phase != 0 else 0

        model.fit(
            x=x_train,
            y=y_train,
            validation_split=0.1,
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
                    filepath=f"{outdir}{model_name.lower()}_" + "{epoch:02d}",
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

    model.save(f"{outdir}{model_name.lower()}")

    timing(start)
