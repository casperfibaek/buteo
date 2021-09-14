from model_layers import inception_block, expansion_block, reduction_block
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (
    Conv2D,
    Concatenate,
)


def model_dual_down(
    shape_higher,
    shape_lower,
    activation="relu",
    kernel_initializer="glorot_normal",
    sizes=[32, 40, 48],
    inception_blocks=2,
    name="denmark",
):
    model_input_rgbn = Input(shape=shape_higher, name=f"{name}_dual_input_rgbn")
    rgbn = Conv2D(
        sizes[0],
        kernel_size=5,
        padding="same",
        activation=activation,
        kernel_initializer=kernel_initializer,
        name=f"{name}_initial_rgbn_dual",
    )(model_input_rgbn)

    for idx in range(inception_blocks):
        rgbn = inception_block(
            rgbn,
            size=sizes[0],
            activation=activation,
            kernel_initializer=kernel_initializer,
            name=f"{name}_dual_rgbn_ib_00_{idx}",
        )

    rgbn_skip_outer = inception_block(
        rgbn,
        size=sizes[0],
        activation=activation,
        kernel_initializer=kernel_initializer,
        name=f"{name}_dual_rgbn_ib_01",
    )

    rgbn = reduction_block(
        rgbn_skip_outer,
        size=sizes[0],
        activation=activation,
        kernel_initializer=kernel_initializer,
        name=f"{name}_dual_rgbn_rb_00",
    )

    for idx in range(inception_blocks):
        rgbn = inception_block(
            rgbn,
            size=sizes[1],
            activation=activation,
            kernel_initializer=kernel_initializer,
            name=f"{name}_dual_rgbn_ib_02_{idx}",
        )

    model_input_swir = Input(shape=shape_lower, name=f"{name}_dual_input_swir")

    swir = Conv2D(
        sizes[1],
        kernel_size=3,
        padding="same",
        activation=activation,
        kernel_initializer=kernel_initializer,
        name=f"{name}_initial_swir_dual",
    )(model_input_swir)

    for idx in range(inception_blocks):
        swir = inception_block(
            swir,
            size=sizes[1],
            activation=activation,
            kernel_initializer=kernel_initializer,
            name=f"{name}_dual_swir_ib_00_{idx}",
        )

    model_skip_inner = Concatenate(name=f"{name}_dual_skip2_merge_concat")([rgbn, swir])

    for idx in range(inception_blocks):
        model = inception_block(
            model_skip_inner,
            size=sizes[1],
            activation=activation,
            kernel_initializer=kernel_initializer,
            name=f"{name}_dual_ib_00_{idx}",
        )

    model = reduction_block(
        model,
        size=sizes[1],
        activation=activation,
        kernel_initializer=kernel_initializer,
        name=f"{name}_dual_rb_00",
    )

    for idx in range(inception_blocks):
        model = inception_block(
            model,
            size=sizes[2],
            activation=activation,
            kernel_initializer=kernel_initializer,
            name=f"{name}_dual_ib_01_{idx}",
        )

    model = expansion_block(
        model,
        size=sizes[2],
        activation=activation,
        kernel_initializer=kernel_initializer,
        name=f"{name}_dual_eb_00",
    )

    model = Concatenate(name=f"{name}_dual_skip_inner_concat")(
        [model, model_skip_inner]
    )

    for idx in range(inception_blocks):
        model = inception_block(
            model,
            size=sizes[1],
            activation=activation,
            kernel_initializer=kernel_initializer,
            name=f"{name}_dual_ib_02_{idx}",
        )

    model = expansion_block(
        model,
        size=sizes[1],
        activation=activation,
        kernel_initializer=kernel_initializer,
        name=f"{name}_dual_eb_01",
    )

    model = Concatenate(name=f"{name}_dual_skip_outer_concat")([model, rgbn_skip_outer])

    for idx in range(inception_blocks):
        model = inception_block(
            model,
            size=sizes[0],
            activation=activation,
            kernel_initializer=kernel_initializer,
            name=f"{name}_tail_dual_ib_00_{idx}",
        )

    model = inception_block(
        model,
        size=sizes[0],
        activation=activation,
        kernel_initializer=kernel_initializer,
        name=f"{name}_tail_dual_ib_01",
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

    return Model(inputs=[model_input_rgbn, model_input_swir], outputs=output)
