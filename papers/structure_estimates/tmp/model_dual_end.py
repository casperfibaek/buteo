from model_layers import inception_block, expansion_block, reduction_block
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (
    Conv2D,
    Concatenate,
)


def model_dual_end(
    shape_rgbn,
    shape_swir,
    activation="relu",
    kernel_initializer="glorot_normal",
    sizes=[32, 40, 48],
    inception_blocks=2,
    name="denmark",
):
    model_input_rgbn = Input(shape=shape_rgbn, name=f"{name}_dual_input_rgbn")
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

    rgbn_skip1 = inception_block(
        rgbn,
        size=sizes[0],
        activation=activation,
        kernel_initializer=kernel_initializer,
        name=f"{name}_dual_rgbn_ib_01_0",
    )

    rgbn = reduction_block(
        rgbn_skip1,
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

    rgbn_skip2 = inception_block(
        rgbn,
        size=sizes[1],
        activation=activation,
        kernel_initializer=kernel_initializer,
        name=f"{name}_dual_rgbn_ib_03",
    )

    rgbn = reduction_block(
        rgbn_skip2,
        size=sizes[1],
        activation=activation,
        kernel_initializer=kernel_initializer,
        name=f"{name}_dual_rgbn_rb_01",
    )

    for idx in range(inception_blocks):
        rgbn = inception_block(
            rgbn,
            size=sizes[2],
            activation=activation,
            kernel_initializer=kernel_initializer,
            name=f"{name}_dual_rgbn_ib_03_{idx}",
        )

    rgbn = expansion_block(
        rgbn,
        size=sizes[1],
        activation=activation,
        kernel_initializer=kernel_initializer,
        name=f"{name}_dual_rgbn_eb_01",
    )

    rgbn = Concatenate(name=f"{name}_dual_rgbn_skip2_concat")([rgbn, rgbn_skip2])

    for idx in range(inception_blocks):
        rgbn = inception_block(
            rgbn,
            size=sizes[1],
            activation=activation,
            kernel_initializer=kernel_initializer,
            name=f"{name}_dual_rgbn_ib_04_{idx}",
        )

    rgbn = expansion_block(
        rgbn,
        size=sizes[0],
        activation=activation,
        kernel_initializer=kernel_initializer,
        name=f"{name}_dual_merge_eb_00",
    )

    for idx in range(inception_blocks):
        rgbn = inception_block(
            rgbn,
            size=sizes[0],
            activation=activation,
            kernel_initializer=kernel_initializer,
            name=f"{name}_dual_merge_ib_01_{idx}",
        )

    # ------------------------ SWIR ------------------------
    model_input_swir = Input(shape=shape_swir, name=f"{name}_dual_input_swir")

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

    swir_skip1 = inception_block(
        swir,
        size=sizes[1],
        activation=activation,
        kernel_initializer=kernel_initializer,
        name=f"{name}_dual_swir_ib_01_0",
    )

    swir = reduction_block(
        swir_skip1,
        size=sizes[1],
        activation=activation,
        kernel_initializer=kernel_initializer,
        name=f"{name}_dual_swir_rb_00",
    )

    for idx in range(inception_blocks):
        swir = inception_block(
            swir,
            size=sizes[2],
            activation=activation,
            kernel_initializer=kernel_initializer,
            name=f"{name}_dual_swir_ib_02_{idx}",
        )

    swir = expansion_block(
        swir,
        size=sizes[1],
        activation=activation,
        kernel_initializer=kernel_initializer,
        name=f"{name}_dual_swir_rb_00",
    )

    swir = Concatenate(name=f"{name}_dual_swir_skip1_concat")([swir, swir_skip1])

    for idx in range(inception_blocks):
        swir = inception_block(
            swir,
            size=sizes[1],
            activation=activation,
            kernel_initializer=kernel_initializer,
            name=f"{name}_dual_swir_ib_03_{idx}",
        )

    swir = expansion_block(
        swir,
        size=sizes[1],
        activation=activation,
        kernel_initializer=kernel_initializer,
        name=f"{name}_dual_swir_rb_01",
    )

    for idx in range(inception_blocks):
        swir = inception_block(
            swir,
            size=sizes[0],
            activation=activation,
            kernel_initializer=kernel_initializer,
            name=f"{name}_dual_swir_ib_04_{idx}",
        )

    # ------------------------ MERGE ------------------------

    model = Concatenate(name=f"{name}_dual_merge_concat")([swir, rgbn])

    for idx in range(inception_blocks):
        model = inception_block(
            model,
            size=sizes[0],
            activation=activation,
            kernel_initializer=kernel_initializer,
            name=f"{name}_tail_ib_00_{idx}",
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
