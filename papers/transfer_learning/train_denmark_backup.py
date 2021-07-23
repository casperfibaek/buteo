yellow_follow = "C:/Users/caspe/Desktop/buteo/"
import sys

sys.path.append(yellow_follow)

import os
import time
import numpy as np
import datetime

import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras import Model, Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    AveragePooling2D,
    Conv2DTranspose,
    Concatenate,
)
from tensorflow.keras import mixed_precision
from buteo.machine_learning.ml_utils import load_mish
from buteo.utils import timing

np.set_printoptions(suppress=True)
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
mixed_precision.set_global_policy("mixed_float16")

folder = "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/data/machine_learning_data/"

load_mish()

start = time.time()


def reduction_block(
    inputs,
    size=32,
    activation="Mish",
    kernel_initializer="glorot_normal",
    name=None,
):
    track1 = AveragePooling2D(
        pool_size=(2, 2),
        padding="same",
        name=name + "_reduction_t1_0",
    )(inputs)
    track2 = Conv2D(
        size,
        kernel_size=3,
        padding="same",
        strides=(2, 2),
        activation=activation,
        kernel_initializer=kernel_initializer,
        name=name + "_reduction_t2_0",
    )(inputs)
    track3 = Conv2D(
        size - 16,
        kernel_size=1,
        padding="same",
        strides=(1, 1),
        activation=activation,
        kernel_initializer=kernel_initializer,
        name=name + "_reduction_t3_0",
    )(inputs)
    track3 = Conv2D(
        size - 8,
        kernel_size=3,
        padding="same",
        strides=(1, 1),
        activation=activation,
        kernel_initializer=kernel_initializer,
        name=name + "_reduction_t3_1",
    )(track3)
    track3 = Conv2D(
        size,
        kernel_size=3,
        padding="same",
        strides=(2, 2),
        activation=activation,
        kernel_initializer=kernel_initializer,
        name=name + "_reduction_t3_2",
    )(track3)

    return Concatenate()(
        [
            track1,
            track2,
            track3,
        ]
    )


def expansion_block(
    inputs,
    size=32,
    activation="Mish",
    kernel_initializer="glorot_normal",
    name=None,
):
    track1 = Conv2DTranspose(
        size,
        kernel_size=3,
        strides=(2, 2),
        kernel_initializer=kernel_initializer,
        activation=activation,
        padding="same",
        name=name + "_expansion_t1_0",
    )(inputs)

    return track1


def inception_block(
    inputs,
    size=32,
    activation="Mish",
    kernel_initializer="glorot_normal",
    name=None,
):
    track1 = MaxPooling2D(
        pool_size=2,
        strides=1,
        padding="same",
        name=name + "_inception_t1_0",
    )(inputs)
    track2 = Conv2D(
        size,
        kernel_size=1,
        padding="same",
        strides=(1, 1),
        activation=activation,
        kernel_initializer=kernel_initializer,
        name=name + "_inception_t2_0",
    )(inputs)
    track3 = Conv2D(
        size - 8,
        kernel_size=1,
        padding="same",
        strides=(1, 1),
        activation=activation,
        kernel_initializer=kernel_initializer,
        name=name + "_inception_t3_0",
    )(inputs)
    track3 = Conv2D(
        size,
        kernel_size=3,
        padding="same",
        strides=(1, 1),
        activation=activation,
        kernel_initializer=kernel_initializer,
        name=name + "_inception_t3_1",
    )(track3)
    track4 = Conv2D(
        size - 16,
        kernel_size=1,
        padding="same",
        strides=(1, 1),
        activation=activation,
        kernel_initializer=kernel_initializer,
        name=name + "_inception_t4_0",
    )(inputs)
    track4 = Conv2D(
        size - 8,
        kernel_size=3,
        padding="same",
        strides=(1, 1),
        activation=activation,
        kernel_initializer=kernel_initializer,
        name=name + "_inception_t4_1",
    )(track4)
    track4 = Conv2D(
        size,
        kernel_size=3,
        padding="same",
        strides=(1, 1),
        activation=activation,
        kernel_initializer=kernel_initializer,
        name=name + "_inception_t4_2",
    )(track4)

    return Concatenate()(
        [
            track1,
            track2,
            track3,
            track4,
        ]
    )


def define_model(
    shape_rgbn,
    shape_swir,
    shape_sar,
    activation="Mish",
    kernel_initializer="glorot_normal",
    sizes=[48, 48, 48],
):
    # ----------------- RGBN ------------------------
    rgbn_input = Input(shape=shape_rgbn, name="rgbn_input")
    rgbn = Conv2D(
        sizes[0],
        kernel_size=5,
        padding="same",
        activation=activation,
        kernel_initializer=kernel_initializer,
        name="initial_conv_rgbn",
    )(rgbn_input)
    rgbn = inception_block(rgbn, size=sizes[0])
    rgbn_skip1 = inception_block(rgbn, size=sizes[0])
    rgbn = reduction_block(rgbn, size=sizes[0])
    rgbn = inception_block(rgbn, size=sizes[0])
    rgbn_skip2 = inception_block(rgbn, size=sizes[1])
    rgbn = reduction_block(rgbn_skip2, size=sizes[1])
    rgbn = inception_block(rgbn, size=sizes[0])
    rgbn = inception_block(rgbn, size=sizes[2])
    rgbn = expansion_block(rgbn, size=sizes[1])
    rgbn = Concatenate()([rgbn_skip2, rgbn])
    rgbn = inception_block(rgbn, size=sizes[1])

    # ----------------- SWIR ------------------------
    swir_input = Input(shape=shape_swir, name="swir_input")
    swir = Conv2D(
        sizes[1],
        kernel_size=3,
        padding="same",
        activation=activation,
        kernel_initializer=kernel_initializer,
        name="initial_conv_swir",
    )(swir_input)
    swir = inception_block(swir, size=sizes[2])
    swir_skip1 = inception_block(swir, size=sizes[1])
    swir = reduction_block(swir_skip1, size=sizes[1])
    swir = inception_block(swir, size=sizes[2])
    swir = inception_block(swir, size=sizes[2])
    swir = expansion_block(swir, size=sizes[1])
    swir = Concatenate()([swir_skip1, swir])
    swir = inception_block(swir, size=sizes[1])

    # CONCATENATE
    model = Concatenate()([rgbn, swir])
    model = inception_block(model, size=sizes[1])
    model = expansion_block(model, size=sizes[0])
    model = Concatenate()([rgbn_skip1, model])
    model = inception_block(model, size=sizes[0])

    # ----------------- SAR -------------------------
    sar_input = Input(shape=shape_sar, name="sar_input")
    sar = Conv2D(
        sizes[0],
        kernel_size=5,
        padding="same",
        activation=activation,
        kernel_initializer=kernel_initializer,
        name="initial_conv_sar",
    )(sar_input)
    sar = inception_block(sar, size=sizes[0])
    sar_skip1 = inception_block(sar, size=sizes[0])
    sar = reduction_block(sar_skip1, size=sizes[0])
    sar = inception_block(sar, size=sizes[0])
    sar_skip2 = inception_block(sar, size=sizes[1])
    sar = reduction_block(sar_skip2, size=sizes[1])
    sar = inception_block(sar, size=sizes[0])
    sar = inception_block(rgbn, size=sizes[2])
    sar = expansion_block(rgbn, size=sizes[1])
    sar = Concatenate()([sar_skip2, rgbn])
    sar = inception_block(sar, size=sizes[0])
    sar = inception_block(sar, size=sizes[0])
    sar = expansion_block(sar, size=sizes[0])
    sar = Concatenate()([sar_skip1, sar])
    sar = inception_block(sar, size=sizes[0])
    sar = inception_block(sar, size=sizes[0])

    # ----------------- TAIL ------------------------
    model = Concatenate()([model, sar])

    model = inception_block(model, size=sizes[0])
    model = inception_block(model, size=sizes[0])
    model = inception_block(model, size=sizes[0])
    model = inception_block(model, size=sizes[0])
    model = inception_block(model, size=sizes[0])

    model = Conv2D(
        sizes[0],
        kernel_size=3,
        padding="same",
        activation=activation,
        kernel_initializer=kernel_initializer,
    )(model)

    model = Conv2D(
        sizes[0],
        kernel_size=3,
        padding="same",
        activation=activation,
        kernel_initializer=kernel_initializer,
    )(model)

    model = Conv2D(
        sizes[0],
        kernel_size=1,
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

    return Model(inputs=[rgbn_input, swir_input, sar_input], outputs=output)


version = 1

# x_train_rgbn = np.load(folder + f"00{version}_RGBN.npy")
# x_train_swir = np.load(folder + f"00{version}_SWIR.npy")
# x_train_sar = np.load(folder + f"00{version}_SAR.npy")

# y_train = np.load(folder + f"00{version}_LABEL_AREA.npy")[:, :, :, 0]
# y_train = np.load(folder + f"00{version}_LABEL_AREA.npy")

# init_mask = y_train.sum(axis=(1, 2)) > 1000

# x_train_rgbn = x_train_rgbn[init_mask][0:5000]
# x_train_swir = x_train_swir[init_mask][0:5000]
# x_train_sar = x_train_sar[init_mask][0:5000]

# y_train = y_train[init_mask][0:5000]

# np.save(folder + "003_RGBN.npy", x_train_rgbn)
# np.save(folder + "003_SWIR.npy", x_train_swir)
# np.save(folder + "003_SAR.npy", x_train_sar)
# np.save(folder + "003_LABEL_AREA.npy", y_train)

x_test_rgbn = preprocess_optical(np.load(folder + "10830461_RGBN.npy"))
x_test_swir = preprocess_optical(np.load(folder + "10830461_SWIR.npy"))
x_test_sar = preprocess_sar(np.load(folder + "10830461_SAR.npy"))
y_test = np.load(folder + "10830461_LABEL_AREA.npy")[:, :, :, 0]
# y_test = np.load(folder + "10830461_LABEL_AREA.npy")


# import pdb

# pdb.set_trace()


def create_model(
    shape_rgbn,
    shape_swir,
    shape_sar,
    kernel_initializer="normal",
    activation="relu",
    learning_rate=0.001,
):
    model = define_model(
        shape_rgbn,
        shape_swir,
        shape_sar,
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


with tf.device("/device:GPU:0"):
    lr = 0.00001
    epochs = [10]
    # epochs = [10]
    bs = [16]
    # bs = [32]

    # model = create_model(
    #     (64, 64, 4),
    #     (32, 32, 2),
    #     (64, 64, 2),
    #     kernel_initializer="glorot_normal",
    #     activation="Mish",
    #     learning_rate=lr,
    # )

    model = tf.keras.models.load_model(folder + "/models/denmark_base_06")

    # print(model.summary())

    # model_checkpoint_callback = ModelCheckpoint(
    #     filepath=folder + "tmp/model_check_advanced_final-{epoch:02d}",
    # )

    # for phase in range(len(bs)):
    #     use_epoch = epochs[phase]
    #     use_bs = bs[phase]
    #     initial_epoch = epochs[phase - 1] if phase != 0 else 0

    #     model.fit(
    #         x=[x_train_rgbn, x_train_swir, x_train_sar],
    #         y=y_train,
    #         validation_split=0.05,
    #         shuffle=True,
    #         validation_data=([x_test_rgbn, x_test_swir, x_test_sar], y_test),
    #         epochs=use_epoch,
    #         initial_epoch=initial_epoch,
    #         verbose=1,
    #         batch_size=use_bs,
    #         use_multiprocessing=True,
    #         workers=0,
    #         callbacks=[
    #             model_checkpoint_callback,
    #             LearningRateScheduler(
    #                 create_step_decay(
    #                     learning_rate=lr,
    #                     drop_rate=0.8,
    #                     epochs_per_drop=3,
    #                 )
    #             ),
    #             # tensorboard_callback,
    #         ],
    #     )

    # model.save(folder + "models/denmark_01", save_format="tf")

    print(f"Batch_size: {str(bs)}")
    loss, mse, mae = model.evaluate(
        x=[x_test_rgbn, x_test_swir, x_test_sar],
        y=y_test,
        verbose=1,
        batch_size=32,
        use_multiprocessing=True,
    )

    print(f"Mean Square Error:      {round(mse, 3)}")
    print(f"Mean Absolute Error:    {round(mae, 3)}")
    print("")

timing(start)


# Epoch 7/10
# 13133/13133 [==============================] - 5528s 421ms/step - loss: 16.4366 - mse: 16.4365 - mae: 0.7040 - val_loss: 17.0241 - val_mse: 17.0240 - val_mae: 0.7045

# 120s 213ms/step - loss: 12.0723 - mse: 12.0723 - mae: 0.7749 - val_loss: 61.4837 - val_mse: 61.4804 - val_mae: 1.6726

# best val
# val_loss: 58.2583 - val_mse: 58.2562 - val_mae: 1.6984

# model5
# Mean Square Error:      64.323
# Mean Absolute Error:    2.418

# Processing took: 0h 1m 3.66s


# Mean Square Error:      144.236
# Mean Absolute Error:    3.824
