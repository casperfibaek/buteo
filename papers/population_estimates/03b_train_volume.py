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
from buteo.machine_learning.ml_utils import create_step_decay, tpe, SaveBestModel
from buteo.utils import timing

from model_trio_down_volume import model_trio_down_volume

np.set_printoptions(suppress=True)
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
mixed_precision.set_global_policy("mixed_float16")

model_name = "volume_for_ghana_04"

folder = "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/data/"
outdir = folder + f"models/"
place = "dojo"


# train_limit = 100000
# val_limit = 10000
# tst_limit = 20000

x_train = [
    np.concatenate(
        [
            np.load(folder + f"{place}/volume_v3_RGBN.npy"),
            np.load(folder + f"{place}/volume_v3_noise_RGBN.npy"),
        ]
    ),
    np.concatenate(
        [
            np.load(folder + f"{place}/volume_v3_SAR.npy"),
            np.load(folder + f"{place}/volume_v3_noise_SAR.npy"),
        ]
    ),
    np.concatenate(
        [
            np.load(folder + f"{place}/volume_v3_RESWIR.npy"),
            np.load(folder + f"{place}/volume_v3_noise_RESWIR.npy"),
        ]
    ),
]

y_train = np.concatenate(
    [
        np.load(folder + f"{place}/volume_v3_label_volume.npy"),
        np.load(folder + f"{place}/volume_v3_noise_label_volume.npy"),
    ]
)

# x_test = [
#     dk_rgbn[-tst_limit:],
#     dk_sar[-tst_limit:],
#     dk_reswir[-tst_limit:],
# ]

# y_test = dk_label[-tst_limit:]

shuffle_mask = np.random.permutation(y_train.shape[0])

for idx in range(len(x_train)):
    x_train[idx] = x_train[idx][shuffle_mask]

y_train = y_train[shuffle_mask]

# x_val = []
# for idx in range(len(x_train)):
#     x_val.append(x_train[idx][-val_limit:])
#     x_train[idx] = x_train[idx][:-val_limit]

# y_val = y_train[-val_limit:]
# y_train = y_train[:-val_limit]

# for idx in range(len(x_train)):
#     x_train[idx] = x_train[idx][:train_limit]

# y_train = y_train[:train_limit]


lr = 0.000001
min_delta = 0.005

with tf.device("/device:GPU:0"):
    epochs = [5, 5, 10, 10]
    bs = [32, 64, 128, 256]
    # epochs = [2, 4]
    # bs = [16, 32]
    # epochs = [20, 10, 5]
    # bs = [256, 128, 64]
    inception_blocks = 3
    activation = "relu"
    output_activation = "relu"
    initializer = "glorot_normal"

    # relu relu mse
    # val_mse: 794.7561 - val_mae: 4.2011 - val_tpe: -2.3732
    # relu relu log_cosh
    # val_mse: 1186.1876 - val_mae: 4.2638 - val_tpe: -14.5024

    # donor_model_path = outdir + "student2_v4"
    # donor_model = tf.keras.models.load_model(
    #     donor_model_path, custom_objects={"tpe": tpe}
    # )

    model = tf.keras.models.load_model(
        outdir + "volume_for_ghana_03", custom_objects={"tpe": tpe}
    )

    # model = model_trio_down_volume(
    #     x_train[0].shape[1:],
    #     x_train[1].shape[1:],
    #     x_train[2].shape[1:],
    #     kernel_initializer=initializer,
    #     output_activation=output_activation,
    #     activation=activation,
    #     inception_blocks=inception_blocks,
    #     name=f"{model_name.lower()}",
    # )

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

    # for idx, _ in enumerate(model.layers[:-37]):
    #     model.layers[idx].set_weights(donor_model.layers[idx].get_weights())
    #     model.layers[idx].trainable = False

    start = time.time()

    save_best_model = SaveBestModel(save_best_metric="val_loss")

    for phase in range(len(bs)):
        use_epoch = np.cumsum(epochs)[phase]
        use_bs = bs[phase]
        initial_epoch = np.cumsum(epochs)[phase - 1] if phase != 0 else 0

        model.fit(
            x=x_train,
            y=y_train,
            validation_split=0.1,
            # validation_data=(x_val, y_val),
            shuffle=True,
            epochs=use_epoch,
            initial_epoch=initial_epoch,
            verbose=1,
            batch_size=use_bs,
            use_multiprocessing=True,
            workers=0,
            callbacks=[
                # LearningRateScheduler(
                #     create_step_decay(
                #         learning_rate=lr,
                #         drop_rate=0.75,
                #         epochs_per_drop=3,
                #     )
                # ),
                # ModelCheckpoint(
                #     filepath=f"{outdir}{model_name.lower()}_" + "{epoch:02d}",
                #     save_best_only=True,
                # ),
                EarlyStopping(
                    monitor="val_loss",
                    patience=3,
                    min_delta=min_delta,
                ),
                save_best_model,
            ],
        )

    model.set_weights(save_best_model.best_weights)
    model.save(f"{outdir}{model_name.lower()}")

    timing(start)

# volume_01
# val_mse: 786.2783 - val_mae: 4.0503 - val_tpe: -2.0579

# volume_02
# val_mse: 641.7850 - val_mae: 3.8456 - val_tpe: -2.8259

# volume_03
# val_loss: 746.9427 - val_mse: 746.9427 - val_mae: 4.0033 - val_tpe: -2.4683
