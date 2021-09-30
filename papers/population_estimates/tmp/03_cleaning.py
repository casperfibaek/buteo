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

model_name = "check_ghana_01"

folder = "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/data/"
outdir = folder + f"models/"
# place = "dojo"
place = "ghana"

x_train = [
    np.load(folder + f"{place}/patches/RGBN.npy"),
    np.load(folder + f"{place}/patches/SAR.npy"),
    np.load(folder + f"{place}/patches/RESWIR.npy"),
]

y_train = np.load(folder + f"{place}/patches/label_area.npy")

# x_train = [
#     np.concatenate(
#         [
#             np.load(folder + f"{place}/patches/dar_RGBN.npy"),
#             np.load(folder + f"{place}/patches/kampala_RGBN.npy"),
#             np.load(folder + f"{place}/patches/kilimanjaro_RGBN.npy"),
#             np.load(folder + f"{place}/patches/mwanza_RGBN.npy"),
#         ]
#     ),
#     np.concatenate(
#         [
#             np.load(folder + f"{place}/patches/dar_SAR.npy"),
#             np.load(folder + f"{place}/patches/kampala_SAR.npy"),
#             np.load(folder + f"{place}/patches/kilimanjaro_SAR.npy"),
#             np.load(folder + f"{place}/patches/mwanza_SAR.npy"),
#         ]
#     ),
#     np.concatenate(
#         [
#             np.load(folder + f"{place}/patches/dar_RESWIR.npy"),
#             np.load(folder + f"{place}/patches/kampala_RESWIR.npy"),
#             np.load(folder + f"{place}/patches/kilimanjaro_RESWIR.npy"),
#             np.load(folder + f"{place}/patches/mwanza_RESWIR.npy"),
#         ]
#     ),
# ]

# y_train = np.concatenate(
#     [
#         np.load(folder + f"{place}/patches/dar_label_area.npy"),
#         np.load(folder + f"{place}/patches/kampala_label_area.npy"),
#         np.load(folder + f"{place}/patches/kilimanjaro_label_area.npy"),
#         np.load(folder + f"{place}/patches/mwanza_label_area.npy"),
#     ]
# )

shuffle_mask = np.random.permutation(y_train.shape[0])

for idx in range(len(x_train)):
    x_train[idx] = x_train[idx][shuffle_mask]

y_train = y_train[shuffle_mask]

for idx in range(len(x_train)):
    x_train[idx] = x_train[idx][:200000]

y_train = y_train[:200000]

lr = 0.0001
min_delta = 0.005

with tf.device("/device:GPU:0"):
    epochs = [10, 10]
    bs = [128, 256]
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
        metrics=["mse", "mae", tpe],
    )

    # print(model.summary())

    # transfer weights
    donor_model_path = outdir + "ghana_area_06_06"
    donor_model = tf.keras.models.load_model(
        donor_model_path, custom_objects={"tpe": tpe}
    )
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
                        drop_rate=0.90,
                        epochs_per_drop=3,
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
