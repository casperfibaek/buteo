yellow_follow = "C:/Users/caspe/Desktop/buteo/"
import sys

sys.path.append(yellow_follow)

import os
import time
import numpy as np

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    AveragePooling2D,
    Conv2DTranspose,
    Concatenate,
)
from tensorflow.keras import mixed_precision
from tensorflow.keras.callbacks import (
    LearningRateScheduler,
)
from buteo.machine_learning.ml_utils import load_mish, create_step_decay


np.set_printoptions(suppress=True)
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
mixed_precision.set_global_policy("mixed_float16")

load_mish()

start = time.time()


def reduction_block(
    inputs,
    size=32,
    activation="Mish",
    kernel_initializer="glorot_normal",
    kernel_regularizer=None,
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
        name=name + "_reduction_t2_0",
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
    )(inputs)
    track3 = Conv2D(
        size - 16,
        kernel_size=1,
        padding="same",
        strides=(1, 1),
        activation=activation,
        name=name + "_reduction_t3_0",
        kernel_initializer=kernel_initializer,
    )(inputs)
    track3 = Conv2D(
        size - 8,
        kernel_size=3,
        padding="same",
        strides=(1, 1),
        activation=activation,
        name=name + "_reduction_t3_1",
        kernel_initializer=kernel_initializer,
    )(track3)
    track3 = Conv2D(
        size,
        kernel_size=3,
        padding="same",
        strides=(2, 2),
        activation=activation,
        name=name + "_reduction_t3_2",
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
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
        activation=activation,
        padding="same",
        name=name + "_expansion_t1_0",
        kernel_initializer=kernel_initializer,
    )(inputs)

    return track1


def inception_block(
    inputs,
    size=32,
    activation="Mish",
    kernel_initializer="glorot_normal",
    name=None,
    kernel_regularizer=None,
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
        name=name + "_inception_t2_0",
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
    )(inputs)
    track3 = Conv2D(
        size - 8,
        kernel_size=1,
        padding="same",
        strides=(1, 1),
        activation=activation,
        name=name + "_inception_t3_0",
        kernel_initializer=kernel_initializer,
    )(inputs)
    track3 = Conv2D(
        size,
        kernel_size=3,
        padding="same",
        strides=(1, 1),
        activation=activation,
        name=name + "_inception_t3_1",
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
    )(track3)
    track4 = Conv2D(
        size - 16,
        kernel_size=1,
        padding="same",
        strides=(1, 1),
        activation=activation,
        name=name + "_inception_t4_0",
        kernel_initializer=kernel_initializer,
    )(inputs)
    track4 = Conv2D(
        size - 8,
        kernel_size=3,
        padding="same",
        strides=(1, 1),
        activation=activation,
        name=name + "_inception_t4_1",
        kernel_initializer=kernel_initializer,
    )(track4)
    track4 = Conv2D(
        size,
        kernel_size=3,
        padding="same",
        strides=(1, 1),
        activation=activation,
        name=name + "_inception_t4_2",
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
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
    sizes=[24, 32, 40],
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
    rgbn = inception_block(rgbn, size=sizes[0], name="rgbn_ib_00")
    rgbn_skip1 = inception_block(rgbn, size=sizes[0], name="rgbn_ib_01")
    rgbn = reduction_block(rgbn, size=sizes[0], name="rgbn_rb_00")
    rgbn = inception_block(rgbn, size=sizes[0], name="rgbn_ib_02")
    rgbn_skip2 = inception_block(rgbn, size=sizes[1], name="rgbn_ib_03")
    rgbn = reduction_block(rgbn_skip2, size=sizes[1], name="rgbn_rb_02")
    rgbn = inception_block(rgbn, size=sizes[2], name="rgbn_ib_04")
    rgbn = expansion_block(rgbn, size=sizes[1], name="rgbn_eb_00")
    rgbn = Concatenate()([rgbn_skip2, rgbn])
    rgbn = inception_block(rgbn, size=sizes[1], name="rgbn_ib_05")

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
    swir = inception_block(swir, size=sizes[1], name="swir_ib_00")
    swir_skip1 = inception_block(swir, size=sizes[1], name="swir_ib_01")
    swir = reduction_block(swir_skip1, size=sizes[1], name="swir_rb_00")
    swir = inception_block(swir, size=sizes[2], name="swir_ib_02")
    swir = expansion_block(swir, size=sizes[1], name="swir_eb_00")
    swir = Concatenate()([swir_skip1, swir])
    swir = inception_block(swir, size=sizes[1], name="swir_ib_03")

    # CONCATENATE
    model = Concatenate()([rgbn, swir])
    model = inception_block(model, size=sizes[1], name="s2_ib_00")
    model = expansion_block(model, size=sizes[0], name="s2_eb_00")
    model = Concatenate()([rgbn_skip1, model])
    model = inception_block(model, size=sizes[0], name="s2_ib_01")

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
    sar = inception_block(sar, size=sizes[0], name="sar_ib_00")
    sar_skip1 = inception_block(sar, size=sizes[0], name="sar_ib_01")
    sar = reduction_block(sar_skip1, size=sizes[0], name="sar_rb_00")
    sar = inception_block(sar, size=sizes[0], name="sar_ib_02")
    sar_skip2 = inception_block(sar, size=sizes[1], name="sar_ib_03")
    sar = reduction_block(sar_skip2, size=sizes[1], name="sar_rb_01")
    sar = inception_block(rgbn, size=sizes[2], name="sar_ib_04")
    sar = expansion_block(rgbn, size=sizes[1], name="sar_eb_00")
    sar = Concatenate()([sar_skip2, rgbn])
    sar = inception_block(sar, size=sizes[0], name="sar_ib_05")
    sar = expansion_block(sar, size=sizes[0], name="sar_eb_01")
    sar = Concatenate()([sar_skip1, sar])
    sar = inception_block(
        sar, size=sizes[0], kernel_regularizer="l1_l2", name="sar_ib_06"
    )

    # ----------------- TAIL ------------------------
    model = Concatenate()([model, sar])

    model = inception_block(model, size=sizes[0], name="tail_ib_00")
    model = inception_block(model, size=sizes[0], name="tail_ib_01")

    model = Conv2D(
        sizes[0],
        kernel_size=3,
        padding="same",
        activation=activation,
        kernel_initializer=kernel_initializer,
        name="tail_conv_00",
    )(model)

    output = Conv2D(
        1,
        kernel_size=3,
        padding="same",
        activation="relu",
        kernel_initializer=kernel_initializer,
        name="tail_conv_01",
    )(model)

    return Model(inputs=[rgbn_input, swir_input, sar_input], outputs=output)


def create_subset(folder, train_or_test="train"):
    y_area = np.load(folder + f"LABEL_AREA_{train_or_test}.npy")

    limit = 25000
    area_limit = 1000

    mask = y_area.sum(axis=(1, 2))[:, 0] > area_limit

    y_area = y_area[mask]
    y_area = y_area[0:limit]
    np.save(folder + f"subset_{limit}_AREA_{train_or_test}", y_area)

    x_rgbn = np.load(folder + f"RGBN_{train_or_test}.npy")[mask]
    x_rgbn = x_rgbn[0:limit]
    np.save(folder + f"subset_{limit}_RGBN_{train_or_test}", x_rgbn)

    x_swir = np.load(folder + f"SWIR_{train_or_test}.npy")[mask]
    x_swir = x_swir[0:limit]
    np.save(folder + f"subset_{limit}_SWIR_{train_or_test}", x_swir)

    x_sar = np.load(folder + f"SAR_{train_or_test}.npy")[mask]
    x_sar = x_sar[0:limit]
    np.save(folder + f"subset_{limit}_SAR_{train_or_test}", x_sar)

    x_vol = np.load(folder + f"LABEL_VOLUME_{train_or_test}.npy")[mask]
    x_vol = x_vol[0:limit]
    np.save(folder + f"subset_{limit}_VOLUME_{train_or_test}", x_vol)

    x_pop = np.load(folder + f"LABEL_PEOPLE_{train_or_test}.npy")[mask]
    x_pop = x_pop[0:limit]
    np.save(folder + f"subset_{limit}_PEOPLE_{train_or_test}", x_pop)


def create_model(
    shape_rgbn,
    shape_swir,
    shape_sar,
    kernel_initializer="normal",
    activation="Mish",
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
    lr = 0.0001
    epochs = [10]
    bs = [64]

    model = create_model(
        (64, 64, 4),
        (32, 32, 2),
        (64, 64, 2),
        kernel_initializer="glorot_normal",
        activation="Mish",
        learning_rate=lr,
    )

    print(model.summary())

    folder = "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/analysis/denmark/"

    x_train_rgbn = folder + "subset_25000_RGBN_train.npy"
    x_train_swir = folder + "subset_25000_SWIR_train.npy"
    x_train_sar = folder + "subset_25000_SAR_train.npy"
    y_train = folder + "subset_25000_AREA_train.npy"

    for phase in range(len(bs)):
        use_epoch = epochs[phase]
        use_bs = bs[phase]
        initial_epoch = epochs[phase - 1] if phase != 0 else 0

        model.fit(
            x=[x_train_rgbn, x_train_swir, x_train_sar],
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
                        drop_rate=0.75,
                        epochs_per_drop=3,
                    )
                ),
            ],
        )

    x_train_rgbn = None
    x_train_swir = None
    x_train_sar = None
    y_train = None

    x_test_rgbn = folder + "RGBN_test.npy"
    x_test_swir = folder + "SWIR_test.npy"
    x_test_sar = folder + "SAR_test.npy"
    y_test = folder + "area_test.npy"

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


# TODO:
# Make small samples to tune model
# Optimize simple model, get < 65 on train
# 215ms/step with 10000
