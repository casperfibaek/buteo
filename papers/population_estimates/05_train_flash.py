yellow_follow = "C:/Users/caspe/Desktop/buteo/"
import sys

sys.path.append(yellow_follow)

import os
import numpy as np
import random

import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.callbacks import (
    LearningRateScheduler,
    EarlyStopping,
)
from buteo.machine_learning.ml_utils import create_step_decay, tpe

from model_trio_down import model_trio_down

np.set_printoptions(suppress=True)
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
mixed_precision.set_global_policy("mixed_float16")

folder = "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/data/"
outdir = folder + f"models/"
place = "dojo"

x_train_base = [
    np.load(folder + f"{place}/all_half_start_RGBN.npy"),
    np.load(folder + f"{place}/all_half_start_SAR.npy"),
    np.load(folder + f"{place}/all_half_start_RESWIR.npy"),
]

y_train_base = np.load(folder + f"{place}/all_half_start_label_area.npy")

x_val = [
    np.load(folder + f"{place}/all_half_val_RGBN.npy"),
    np.load(folder + f"{place}/all_half_val_SAR.npy"),
    np.load(folder + f"{place}/all_half_val_RESWIR.npy"),
]

y_val = np.load(folder + f"{place}/all_half_val_label_area.npy")

for idx in range(len(x_val)):
    x_val[idx] = x_val[idx][:20000]

y_val = y_val[:20000]

limit = 50000
flashes = int(y_train_base.shape[0] // limit)

for iteration in range(flashes):
    print(f"Iteration {iteration + 1} out of {flashes}")
    model_name = f"flash_run3_{iteration}"
    out_path = f"{outdir}{model_name}"

    # ------------- Shuffle ------------
    shuffle_mask = np.random.permutation(y_train_base.shape[0])

    for idx in range(len(x_train_base)):
        x_train_base[idx] = x_train_base[idx][shuffle_mask]

    y_train_base = y_train_base[shuffle_mask]
    # ------------- / Shuffle ----------

    # ------------- Limit ------------
    x_train = []
    for idx in range(len(x_train_base)):
        x_train.append(x_train_base[idx][:limit])

    y_train = y_train_base[:limit]
    # ------------- / Limit ----------

    lr = random.choice([0.001, 0.0001, 0.00001])
    min_delta = 0.005

    with tf.device("/device:GPU:0"):
        epochs = [random.choice([4, 8, 12])]
        bs = [random.choice([64, 128, 256])]
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
            loss="mse",
            metrics=["mse", "mae", tpe],  # tpe
        )

        if iteration == 0:
            donor_model_path = outdir + "flash_run2_0"
        else:
            donor_name = f"flash_run3_{iteration - 1}"
            donor_model_path = f"{outdir}{donor_name}"

        donor_model = tf.keras.models.load_model(
            donor_model_path, custom_objects={"tpe": tpe}
        )
        model.set_weights(donor_model.get_weights())
        donor_model = None

        for phase in range(len(bs)):
            use_epoch = np.cumsum(epochs)[phase]
            use_bs = bs[phase]
            initial_epoch = np.cumsum(epochs)[phase - 1] if phase != 0 else 0

            model.fit(
                x=x_train,
                y=y_train,
                validation_data=(x_val, y_val),
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
                            drop_rate=random.choice([0.5, 0.8, 0.95]),
                            epochs_per_drop=3,
                        )
                    ),
                    EarlyStopping(
                        monitor="val_loss",
                        patience=2,
                        min_delta=min_delta,
                        restore_best_weights=True,
                    ),
                ],
            )

        model.save(out_path)

# first round
# val_mse: 20.8017
# val_mae: 0.8449
# val_tpe: -1.8925
