from model_layers import inception_block, expansion_block, reduction_block
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (
    Conv2D,
    Concatenate,
)


def model_single(
    shape,
    activation="relu",
    kernel_initializer="glorot_normal",
    sizes=[32, 40, 48],
    inception_blocks=2,
    name="denmark",
):
    model_input = Input(shape=shape, name=f"{name}_single_input")
    model = Conv2D(
        sizes[0],
        kernel_size=5,
        padding="same",
        activation=activation,
        kernel_initializer=kernel_initializer,
        name=f"{name}_initial_conv_single",
    )(model_input)

    for idx in range(inception_blocks):
        model = inception_block(
            model,
            size=sizes[0],
            activation=activation,
            kernel_initializer=kernel_initializer,
            name=f"{name}_single_ib_00_{idx}",
        )

    model_skip1 = inception_block(
        model,
        size=sizes[0],
        activation=activation,
        kernel_initializer=kernel_initializer,
        name=f"{name}_single_ib_01",
    )

    model = reduction_block(
        model_skip1,
        size=sizes[0],
        activation=activation,
        kernel_initializer=kernel_initializer,
        name=f"{name}_single_rb_00",
    )

    for idx in range(inception_blocks):
        model = inception_block(
            model,
            size=sizes[1],
            activation=activation,
            kernel_initializer=kernel_initializer,
            name=f"{name}_single_ib_02_{idx}",
        )

    model_skip2 = inception_block(
        model,
        size=sizes[1],
        activation=activation,
        kernel_initializer=kernel_initializer,
        name=f"{name}_single_ib_03",
    )

    model = reduction_block(
        model_skip2,
        size=sizes[1],
        activation=activation,
        kernel_initializer=kernel_initializer,
        name=f"{name}_single_rb_02",
    )

    for idx in range(inception_blocks):
        model = inception_block(
            model,
            size=sizes[2],
            activation=activation,
            kernel_initializer=kernel_initializer,
            name=f"{name}_single_ib_04_{idx}",
        )

    model = expansion_block(
        model,
        size=sizes[1],
        activation=activation,
        kernel_initializer=kernel_initializer,
        name=f"{name}_single_eb_00",
    )

    model = Concatenate(name=f"{name}_single_skip2_concat")([model, model_skip2])

    for idx in range(inception_blocks):
        model = inception_block(
            model,
            size=sizes[1],
            activation=activation,
            kernel_initializer=kernel_initializer,
            name=f"{name}_single_ib_05_{idx}",
        )

    model = expansion_block(
        model,
        size=sizes[1],
        activation=activation,
        kernel_initializer=kernel_initializer,
        name=f"{name}_single_eb_01",
    )

    model = Concatenate(name=f"{name}_single_skip1_concat")([model, model_skip1])

    for idx in range(inception_blocks):
        model = inception_block(
            model,
            size=sizes[0],
            activation=activation,
            kernel_initializer=kernel_initializer,
            name=f"{name}_tail_single_00_{idx}",
        )

    model = inception_block(
        model,
        size=sizes[0],
        activation=activation,
        kernel_initializer=kernel_initializer,
        name=f"{name}_tail_ib_01",
    )

    model = Conv2D(
        sizes[0],
        kernel_size=3,
        padding="same",
        activation=activation,
        kernel_initializer=kernel_initializer,
        name=f"{name}_tail_conv_00",
    )(model)

    output = Conv2D(
        1,
        kernel_size=3,
        padding="same",
        activation="relu",
        kernel_initializer=kernel_initializer,
        name=f"{name}_tail_output",
        dtype="float32",
    )(model)

    return Model(inputs=[model_input], outputs=output)
