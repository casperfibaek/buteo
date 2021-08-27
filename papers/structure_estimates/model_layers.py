from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    AveragePooling2D,
    Conv2DTranspose,
    Concatenate,
)


def reduction_block(
    inputs,
    size=32,
    reduction_01=4,
    reduction_02=8,
    activation="relu",
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
        size - reduction_02,
        kernel_size=1,
        padding="same",
        strides=(1, 1),
        activation=activation,
        kernel_initializer=kernel_initializer,
        name=name + "_reduction_t3_0",
    )(inputs)
    track3 = Conv2D(
        size - reduction_01,
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
    track4 = MaxPooling2D(
        pool_size=(2, 2),
        padding="same",
        name=name + "_reduction_t4_0",
    )(inputs)

    return Concatenate(name=f"{name}_reduction_concat")(
        [
            track1,
            track2,
            track3,
            track4,
        ]
    )


def expansion_block(
    inputs,
    size=32,
    activation="relu",
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
    reduction_01=4,
    reduction_02=8,
    activation="relu",
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
        size - reduction_01,
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
        size - reduction_02,
        kernel_size=1,
        padding="same",
        strides=(1, 1),
        activation=activation,
        kernel_initializer=kernel_initializer,
        name=name + "_inception_t4_0",
    )(inputs)
    track4 = Conv2D(
        size - reduction_01,
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
