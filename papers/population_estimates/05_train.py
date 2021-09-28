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
from buteo.machine_learning.ml_utils import create_step_decay, tpe, mse_biased
from buteo.utils import timing

from model_trio_down import model_trio_down

np.set_printoptions(suppress=True)
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
mixed_precision.set_global_policy("mixed_float16")

model_name = "student_01"

folder = "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/data/"
outdir = folder + f"models/"
place = "dojo"

# x_train = [
#     np.concatenate(
#         [
#             np.load(folder + f"{place}/all_noise_RGBN.npy"),
#             np.load(folder + f"{place}/teacher_RGBN.npy"),
#         ]
#     ),
#     np.concatenate(
#         [
#             np.load(folder + f"{place}/all_noise_SAR.npy"),
#             np.load(folder + f"{place}/teacher_SAR.npy"),
#         ]
#     ),
#     np.concatenate(
#         [
#             np.load(folder + f"{place}/all_noise_RESWIR.npy"),
#             np.load(folder + f"{place}/teacher_RESWIR.npy"),
#         ]
#     ),
# ]

# y_train = np.concatenate(
#     [
#         np.load(folder + f"{place}/all_noise_label_area.npy"),
#         np.load(folder + f"{place}/teacher_label_area.npy"),
#     ]
# )

x_test = [
    np.load(folder + f"{place}/test_RGBN.npy"),
    np.load(folder + f"{place}/test_SAR.npy"),
    np.load(folder + f"{place}/test_RESWIR.npy"),
]

y_test = np.load(folder + f"{place}/test_label_area.npy")

# shuffle_mask = np.random.permutation(y_train.shape[0])

# for idx in range(len(x_train)):
#     x_train[idx] = x_train[idx][shuffle_mask]

# y_train = y_train[shuffle_mask]

# limit = y_train.shape[0] // 8

# for idx in range(len(x_train)):
#     x_train[idx] = x_train[idx][:limit]

# y_train = y_train[:limit]

lr = 0.0001
min_delta = 0.005

with tf.device("/device:GPU:0"):
    epochs = [16, 8, 4, 2]
    bs = [256, 128, 64, 32]
    # bs = [16, 8, 4]
    inception_blocks = 3
    activation = "relu"
    initializer = "glorot_normal"

    donor_model_path = outdir + "student_01"
    model = tf.keras.models.load_model(donor_model_path, custom_objects={"tpe": tpe})

    # model = model_trio_down(
    #     x_train[0].shape[1:],
    #     x_train[1].shape[1:],
    #     x_train[2].shape[1:],
    #     kernel_initializer=initializer,
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
        # loss="log_cosh",
        loss="mse",
        # loss=mse_biased,
        metrics=["mse", "mae", tpe],  # tpe
    )

    # transfer weights
    # donor_model_path = outdir + "teacher_06"
    # donor_model = tf.keras.models.load_model(
    #     donor_model_path, custom_objects={"tpe": tpe}
    # )
    # model.set_weights(donor_model.get_weights())
    # donor_model = None

    start = time.time()

    # for layer in model.layers[:-7]:
    #     layer.trainable = False

    # print(model.summary())

    print("Evaluating:")
    model.evaluate(x=x_test, y=y_test, batch_size=1024)
    print("")

    exit()

    for phase in range(len(bs)):
        use_epoch = np.cumsum(epochs)[phase]
        use_bs = bs[phase]
        initial_epoch = np.cumsum(epochs)[phase - 1] if phase != 0 else 0

        model.fit(
            x=x_train,
            y=y_train,
            # validation_split=0.1,
            validation_data=(x_test, y_test),
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
                        drop_rate=0.75,
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
