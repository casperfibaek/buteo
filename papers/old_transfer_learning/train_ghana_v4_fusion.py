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
    ModelCheckpoint,
    EarlyStopping,
)
from buteo.machine_learning.ml_utils import create_step_decay
from buteo.utils import timing


np.set_printoptions(suppress=True)
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
mixed_precision.set_global_policy("mixed_float16")

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
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        name=name + "_reduction_t2_0",
    )(inputs)
    track3 = Conv2D(
        size - 8,
        kernel_size=1,
        padding="same",
        strides=(1, 1),
        activation=activation,
        kernel_initializer=kernel_initializer,
        name=name + "_reduction_t3_0",
    )(inputs)
    track3 = Conv2D(
        size - 4,
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
        kernel_regularizer=kernel_regularizer,
        name=name + "_reduction_t3_2",
    )(track3)

    return Concatenate(name=f"{name}_reduction_concat")(
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
        kernel_initializer=kernel_initializer,
        name=name + "_expansion_t1_0",
    )(inputs)

    return track1


def inception_block(
    inputs,
    size=32,
    activation="Mish",
    name=None,
    kernel_initializer="glorot_normal",
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
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        name=name + "_inception_t2_0",
    )(inputs)
    track3 = Conv2D(
        size - 4,
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
        kernel_regularizer=kernel_regularizer,
        name=name + "_inception_t3_1",
    )(track3)
    track4 = Conv2D(
        size - 8,
        kernel_size=1,
        padding="same",
        strides=(1, 1),
        activation=activation,
        kernel_initializer=kernel_initializer,
        name=name + "_inception_t4_0",
    )(inputs)
    track4 = Conv2D(
        size - 4,
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
        kernel_regularizer=kernel_regularizer,
        name=name + "_inception_t4_2",
    )(track4)

    return Concatenate(name=f"{name}_inception_concat")(
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
    sizes=[40, 48, 56],
):
    # ----------------- RGBN ------------------------
    rgbn_input = Input(
        shape=shape_rgbn,
        name="gh_rgbn_input",
    )
    rgbn = Conv2D(
        sizes[0],
        kernel_size=5,
        padding="same",
        activation=activation,
        kernel_initializer=kernel_initializer,
        name="gh_initial_conv_rgbn",
    )(rgbn_input)
    rgbn = inception_block(
        rgbn,
        size=sizes[0],
        name="gh_rgbn_ib_00",
        activation=activation,
        kernel_initializer=kernel_initializer,
    )
    rgbn_skip1 = inception_block(
        rgbn,
        size=sizes[0],
        name="gh_rgbn_ib_01",
        activation=activation,
        kernel_initializer=kernel_initializer,
    )
    rgbn = reduction_block(
        rgbn,
        size=sizes[0],
        name="gh_rgbn_rb_00",
        activation=activation,
        kernel_initializer=kernel_initializer,
    )
    rgbn = inception_block(
        rgbn,
        size=sizes[0],
        name="gh_rgbn_ib_02",
        activation=activation,
        kernel_initializer=kernel_initializer,
    )
    rgbn_skip2 = inception_block(
        rgbn,
        size=sizes[1],
        name="gh_rgbn_ib_03",
        activation=activation,
        kernel_initializer=kernel_initializer,
    )
    rgbn = reduction_block(
        rgbn_skip2,
        size=sizes[1],
        name="gh_rgbn_rb_02",
        activation=activation,
        kernel_initializer=kernel_initializer,
    )
    rgbn = inception_block(
        rgbn,
        size=sizes[2],
        name="gh_rgbn_ib_04",
        activation=activation,
        kernel_initializer=kernel_initializer,
    )
    rgbn = expansion_block(
        rgbn,
        size=sizes[1],
        name="gh_rgbn_eb_00",
        activation=activation,
        kernel_initializer=kernel_initializer,
    )
    rgbn = Concatenate(name="gh_concat_00")([rgbn_skip2, rgbn])
    rgbn = inception_block(
        rgbn,
        size=sizes[1],
        name="gh_rgbn_ib_05",
        activation=activation,
        kernel_initializer=kernel_initializer,
    )

    # ----------------- SWIR ------------------------
    swir_input = Input(shape=shape_swir, name="gh_swir_input")
    swir = Conv2D(
        sizes[1],
        kernel_size=3,
        padding="same",
        activation=activation,
        kernel_initializer=kernel_initializer,
        name="gh_initial_conv_swir",
    )(swir_input)
    swir = inception_block(
        swir,
        size=sizes[1],
        name="gh_swir_ib_00",
        activation=activation,
        kernel_initializer=kernel_initializer,
    )
    swir_skip1 = inception_block(
        swir,
        size=sizes[1],
        name="gh_swir_ib_01",
        activation=activation,
        kernel_initializer=kernel_initializer,
    )
    swir = reduction_block(
        swir_skip1,
        size=sizes[1],
        name="gh_swir_rb_00",
        activation=activation,
        kernel_initializer=kernel_initializer,
    )
    swir = inception_block(
        swir,
        size=sizes[2],
        name="gh_swir_ib_02",
        activation=activation,
        kernel_initializer=kernel_initializer,
    )
    swir = expansion_block(
        swir,
        size=sizes[1],
        name="gh_swir_eb_00",
        activation=activation,
        kernel_initializer=kernel_initializer,
    )
    swir = Concatenate(name="gh_concat_01")([swir_skip1, swir])
    swir = inception_block(
        swir,
        size=sizes[1],
        name="gh_swir_ib_03",
        activation=activation,
        kernel_initializer=kernel_initializer,
    )

    # CONCATENATE
    model = Concatenate(name="gh_concat_02")([rgbn, swir])
    model = inception_block(
        model,
        size=sizes[1],
        name="gh_s2_ib_00",
        activation=activation,
        kernel_initializer=kernel_initializer,
    )
    model = expansion_block(
        model,
        size=sizes[0],
        name="gh_s2_eb_00",
        activation=activation,
        kernel_initializer=kernel_initializer,
    )
    model = Concatenate(name="gh_concat_03")([rgbn_skip1, model])
    model = inception_block(
        model,
        size=sizes[0],
        name="gh_s2_ib_01",
        activation=activation,
        kernel_initializer=kernel_initializer,
    )

    # ----------------- SAR -------------------------
    sar_input = Input(shape=shape_sar, name="gh_sar_input")
    sar = Conv2D(
        sizes[0],
        kernel_size=5,
        padding="same",
        activation=activation,
        kernel_initializer=kernel_initializer,
        name="gh_initial_conv_sar",
    )(sar_input)
    sar = inception_block(
        sar,
        size=sizes[0],
        name="gh_sar_ib_00",
        activation=activation,
        kernel_initializer=kernel_initializer,
    )
    sar_skip1 = inception_block(
        sar,
        size=sizes[0],
        name="gh_sar_ib_01",
        activation=activation,
        kernel_initializer=kernel_initializer,
    )
    sar = reduction_block(
        sar_skip1,
        size=sizes[0],
        name="gh_sar_rb_00",
        activation=activation,
        kernel_initializer=kernel_initializer,
    )
    sar = inception_block(
        sar,
        size=sizes[0],
        name="gh_sar_ib_02",
        activation=activation,
        kernel_initializer=kernel_initializer,
    )
    sar_skip2 = inception_block(
        sar,
        size=sizes[1],
        name="gh_sar_ib_03",
        activation=activation,
        kernel_initializer=kernel_initializer,
    )
    sar = reduction_block(
        sar_skip2,
        size=sizes[1],
        name="gh_sar_rb_01",
        activation=activation,
        kernel_initializer=kernel_initializer,
    )
    sar = inception_block(
        rgbn,
        size=sizes[2],
        name="gh_sar_ib_04",
        activation=activation,
        kernel_initializer=kernel_initializer,
    )
    sar = expansion_block(
        rgbn,
        size=sizes[1],
        name="gh_sar_eb_00",
        activation=activation,
        kernel_initializer=kernel_initializer,
    )
    sar = Concatenate(name="gh_concat_04")([sar_skip2, rgbn])
    sar = inception_block(
        sar,
        size=sizes[0],
        name="gh_sar_ib_05",
        activation=activation,
        kernel_initializer=kernel_initializer,
    )
    sar = expansion_block(
        sar,
        size=sizes[0],
        name="gh_sar_eb_01",
        activation=activation,
        kernel_initializer=kernel_initializer,
    )
    sar = Concatenate(name="gh_concat_05")([sar_skip1, sar])
    sar = inception_block(
        sar,
        size=sizes[0],
        name="gh_sar_ib_06",
        activation=activation,
        kernel_initializer=kernel_initializer,
    )

    # ----------------- JOIN ------------------------
    model = Concatenate(name="gh_concat_06")([model, sar])

    model = inception_block(
        model,
        size=32,
        name="gh_tail_ib_00",
        activation=activation,
        kernel_initializer=kernel_initializer,
    )
    model = inception_block(
        model,
        size=32,
        name="gh_tail_ib_01",
        activation=activation,
        kernel_initializer=kernel_initializer,
    )

    base_output = Conv2D(
        40,
        kernel_size=3,
        padding="same",
        activation=activation,
        kernel_initializer=kernel_initializer,
        name="gh_tail_conv_base",
    )(model)

    base_model = Model(inputs=[rgbn_input, swir_input, sar_input], outputs=base_output)

    # --------------- TRANSFER ----------------------
    folder_dk = "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/analysis/denmark/"

    dk_model = tf.keras.models.load_model(folder_dk + "/models/area_advanced_v10_05")
    dk_new_output = dk_model.layers[-2].output
    dk_model = Model(inputs=dk_model.input, outputs=dk_new_output)

    for layer in dk_model.layers:
        layer.trainable = False

    for layer in dk_model.layers[-10:]:
        layer.trainable = True

    merged = Concatenate(name="merged_concat_00")([base_model.output, dk_model.output])

    merged = inception_block(
        merged,
        size=40,
        name="gh_tail_ib_02",
        activation=activation,
        kernel_initializer=kernel_initializer,
    )
    merged = inception_block(
        merged,
        size=40,
        name="gh_tail_ib_03",
        activation=activation,
        kernel_initializer=kernel_initializer,
    )

    merged = Conv2D(
        32,
        kernel_size=3,
        padding="same",
        activation=activation,
        kernel_initializer=kernel_initializer,
        name="gh_tail_conv_01",
    )(merged)

    output = Conv2D(
        1,
        kernel_size=3,
        padding="same",
        activation="relu",
        kernel_initializer=kernel_initializer,
        name="gh_tail_output",
    )(merged)

    return Model(inputs=[base_model.input, dk_model.input], outputs=output)


def create_subset(folder, train_or_test="train"):
    y_area = np.load(folder + f"LABEL_AREA_{train_or_test}.npy")

    limit = 25000
    area_limit = 2500

    mask = y_area.sum(axis=(1, 2))[:, 0] > area_limit

    y_area = y_area[mask]
    y_area = y_area[0:limit]
    np.save(folder + f"subset_{limit}_LABEL_AREA_{train_or_test}", y_area)

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
    np.save(folder + f"subset_{limit}_LABEL_VOLUME_{train_or_test}", x_vol)

    x_pop = np.load(folder + f"LABEL_PEOPLE_{train_or_test}.npy")[mask]
    x_pop = x_pop[0:limit]
    np.save(folder + f"subset_{limit}_LABEL_PEOPLE_{train_or_test}", x_pop)


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
    epochs = [10, 50, 50]
    bs = [96, 64, 32]

    # model_name = "subset_25000_area_simple_01"
    model_name = "ghana_fusion_01"

    # folder_dk = "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/analysis/denmark/"
    folder = "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/analysis/ghana/vector/grid_cells/patches/merged/"

    test_size = 5000

    x_train_rgbn = np.load(folder + "RGBN.npy")
    x_train_swir = np.load(folder + "SWIR.npy")
    x_train_sar = np.load(folder + "SAR.npy")
    y_train = np.load(folder + "LABEL_AREA.npy")

    x_test_rgbn = x_train_rgbn[:test_size]
    x_test_swir = x_train_swir[:test_size]
    x_test_sar = x_train_sar[:test_size]
    y_test = y_train[:test_size]

    x_train_rgbn = x_train_rgbn[test_size:]
    x_train_swir = x_train_swir[test_size:]
    x_train_sar = x_train_sar[test_size:]
    y_train = y_train[test_size:]

    # import pdb

    # pdb.set_trace()

    # create_subset(folder, "train")
    # x_train_rgbn = np.load(folder + "subset_25000_RGBN_train.npy")
    # x_train_swir = np.load(folder + "subset_25000_SWIR_train.npy")
    # x_train_sar = np.load(folder + "subset_25000_SAR_train.npy")
    # y_train = np.load(folder + "subset_25000_LABEL_AREA_train.npy")

    # donor_model = tf.keras.models.load_model(folder_dk + "/models/area_advanced_v07_01")

    model = create_model(
        (64, 64, 4),
        (32, 32, 2),
        (64, 64, 2),
        kernel_initializer="glorot_normal",
        activation="relu",
        learning_rate=lr,
    )

    # model.set_weights(donor_model.get_weights())

    # model = tf.keras.models.load_model(folder + "/models/ghana_05_fusion_40")

    print(model.summary())

    model_checkpoint_callback = ModelCheckpoint(
        filepath=folder + f"models/{model_name}_" + "{epoch:02d}" + "_best",
        save_best_only=True,
    )

    for phase in range(len(bs)):
        use_epoch = np.cumsum(epochs)[phase]
        use_bs = bs[phase]
        initial_epoch = np.cumsum(epochs)[phase - 1] if phase != 0 else 0

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
                        drop_rate=0.9,
                        epochs_per_drop=5,
                    )
                ),
                model_checkpoint_callback,
                EarlyStopping(monitor="val_loss", patience=3, min_delta=0.05),
            ],
        )

    x_train_rgbn = None
    x_train_swir = None
    x_train_sar = None
    y_train = None

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

    model.save(folder + f"/models/{model_name}_final")

timing(start)
