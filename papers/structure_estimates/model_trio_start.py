from model_layers import inception_block, expansion_block, reduction_block
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (
    Conv2D,
    Concatenate,
)


def model_trio_down(
    shape_rgbn,
    shape_swir,
    shape_sar,
    activation="relu",
    kernel_initializer="glorot_normal",
    sizes=[32, 40, 48],
    inception_blocks=3,
    name="denmark",
):
    model_input_rgbn = Input(shape=shape_rgbn, name=f"{name}_trio_input_rgbn")
    rgbn = Conv2D(
        sizes[0],
        kernel_size=5,
        padding="same",
        activation=activation,
        kernel_initializer=kernel_initializer,
        name=f"{name}_initial_rgbn_trio",
    )(model_input_rgbn)

    for idx in range(inception_blocks):
        rgbn = inception_block(
            rgbn,
            size=sizes[0],
            activation=activation,
            kernel_initializer=kernel_initializer,
            name=f"{name}_trio_rgbn_ib_00_{idx}",
        )

    model_input_sar = Input(shape=shape_sar, name=f"{name}_trio_input_sar")
    sar = Conv2D(
        sizes[0],
        kernel_size=5,
        padding="same",
        activation=activation,
        kernel_initializer=kernel_initializer,
        name=f"{name}_initial_sar_trio",
    )(model_input_sar)

    for idx in range(inception_blocks):
        sar = inception_block(
            sar,
            size=sizes[0],
            activation=activation,
            kernel_initializer=kernel_initializer,
            name=f"{name}_trio_sar_ib_00_{idx}",
        )

    model_skip_outer = Concatenate(name=f"{name}_trio_skip_outer_concat")([rgbn, sar])

    model = inception_block(
        model_skip_outer,
        size=sizes[0],
        activation=activation,
        kernel_initializer=kernel_initializer,
        name=f"{name}_trio_ib_01",
    )

    model = reduction_block(
        model,
        size=sizes[0],
        activation=activation,
        kernel_initializer=kernel_initializer,
        name=f"{name}_trio_rb_00",
    )

    for idx in range(inception_blocks):
        model = inception_block(
            model,
            size=sizes[1],
            activation=activation,
            kernel_initializer=kernel_initializer,
            name=f"{name}_trio_ib_02_{idx}",
        )

    model_input_swir = Input(shape=shape_swir, name=f"{name}_trio_input_swir")

    swir = Conv2D(
        sizes[1],
        kernel_size=3,
        padding="same",
        activation=activation,
        kernel_initializer=kernel_initializer,
        name=f"{name}_initial_swir_trio",
    )(model_input_swir)

    for idx in range(inception_blocks):
        swir = inception_block(
            swir,
            size=sizes[1],
            activation=activation,
            kernel_initializer=kernel_initializer,
            name=f"{name}_trio_swir_ib_00_{idx}",
        )

    model_skip_inner = Concatenate(name=f"{name}_trio_skip2_merge_concat")(
        [model, swir]
    )

    for idx in range(inception_blocks):
        model = inception_block(
            model_skip_inner,
            size=sizes[1],
            activation=activation,
            kernel_initializer=kernel_initializer,
            name=f"{name}_trio_ib_03_{idx}",
        )

    model = reduction_block(
        model,
        size=sizes[1],
        activation=activation,
        kernel_initializer=kernel_initializer,
        name=f"{name}_trio_rb_01",
    )

    for idx in range(inception_blocks):
        model = inception_block(
            model,
            size=sizes[2],
            activation=activation,
            kernel_initializer=kernel_initializer,
            name=f"{name}_trio_ib_04_{idx}",
        )

    model = expansion_block(
        model,
        size=sizes[2],
        activation=activation,
        kernel_initializer=kernel_initializer,
        name=f"{name}_trio_eb_00",
    )

    model = Concatenate(name=f"{name}_trio_skip_inner_concat")(
        [model, model_skip_inner]
    )

    for idx in range(inception_blocks):
        model = inception_block(
            model,
            size=sizes[1],
            activation=activation,
            kernel_initializer=kernel_initializer,
            name=f"{name}_trio_ib_05_{idx}",
        )

    model = expansion_block(
        model,
        size=sizes[1],
        activation=activation,
        kernel_initializer=kernel_initializer,
        name=f"{name}_trio_eb_01",
    )

    model = Concatenate(name=f"{name}_trio_skip_outer_concat")(
        [model, model_skip_outer]
    )

    for idx in range(inception_blocks):
        model = inception_block(
            model,
            size=sizes[0],
            activation=activation,
            kernel_initializer=kernel_initializer,
            name=f"{name}_tail_trio_ib_00_{idx}",
        )

    model = inception_block(
        model,
        size=sizes[0],
        activation=activation,
        kernel_initializer=kernel_initializer,
        name=f"{name}_tail_trio_ib_01",
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
    )(model)

    return Model(inputs=[model_input_rgbn, model_input_swir], outputs=output)
