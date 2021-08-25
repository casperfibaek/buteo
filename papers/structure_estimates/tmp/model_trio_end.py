from model_layers import inception_block, expansion_block, reduction_block
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (
    Conv2D,
    Concatenate,
)


def model_trio_end(
    shape_higher_01,
    shape_higher_02,
    shape_lower,
    activation="relu",
    kernel_initializer="glorot_normal",
    sizes=[24, 32, 40],
    inception_blocks=2,
    name="denmark",
):
    model_input_higher_01 = Input(
        shape=shape_higher_01, name=f"{name}_trio_input_higher_01"
    )
    model_higher_01 = Conv2D(
        sizes[0],
        kernel_size=5,
        padding="same",
        activation=activation,
        kernel_initializer=kernel_initializer,
        name=f"{name}_initial_model_higher_01_trio",
    )(model_input_higher_01)

    for idx in range(inception_blocks):
        model_higher_01 = inception_block(
            model_higher_01,
            size=sizes[0],
            activation=activation,
            kernel_initializer=kernel_initializer,
            name=f"{name}_trio_model_higher_01_ib_00_{idx}",
        )

    model_skip_outer = inception_block(
        model_higher_01,
        size=sizes[0],
        activation=activation,
        kernel_initializer=kernel_initializer,
        name=f"{name}_trio_model_higher_01_ib_01",
    )

    model_higher_01 = reduction_block(
        model_skip_outer,
        size=sizes[0],
        activation=activation,
        kernel_initializer=kernel_initializer,
        name=f"{name}_trio_model_higher_01_rb_00",
    )

    for idx in range(inception_blocks):
        model_higher_01 = inception_block(
            model_higher_01,
            size=sizes[1],
            activation=activation,
            kernel_initializer=kernel_initializer,
            name=f"{name}_trio_model_higher_01_ib_02_{idx}",
        )

    model_input_model_lower = Input(
        shape=shape_lower, name=f"{name}_trio_input_model_lower"
    )

    model_lower = Conv2D(
        sizes[1],
        kernel_size=3,
        padding="same",
        activation=activation,
        kernel_initializer=kernel_initializer,
        name=f"{name}_initial_model_lower_trio",
    )(model_input_model_lower)

    for idx in range(inception_blocks):
        model_lower = inception_block(
            model_lower,
            size=sizes[1],
            activation=activation,
            kernel_initializer=kernel_initializer,
            name=f"{name}_trio_model_lower_ib_00_{idx}",
        )

    model_skip_inner = Concatenate(name=f"{name}_trio_skip2_merge_concat")(
        [model_higher_01, model_lower]
    )

    for idx in range(inception_blocks):
        model = inception_block(
            model_skip_inner,
            size=sizes[1],
            activation=activation,
            kernel_initializer=kernel_initializer,
            name=f"{name}_trio_ib_00_{idx}",
        )

    model = reduction_block(
        model,
        size=sizes[1],
        activation=activation,
        kernel_initializer=kernel_initializer,
        name=f"{name}_trio_rb_00",
    )

    for idx in range(inception_blocks):
        model = inception_block(
            model,
            size=sizes[2],
            activation=activation,
            kernel_initializer=kernel_initializer,
            name=f"{name}_trio_ib_01_{idx}",
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
            name=f"{name}_trio_ib_02_{idx}",
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
            name=f"{name}_trio_ib_03_{idx}",
        )

    model_input_higher_02 = Input(
        shape=shape_higher_02, name=f"{name}_trio_input_higher_02"
    )
    model_higher_02 = Conv2D(
        sizes[0],
        kernel_size=5,
        padding="same",
        activation=activation,
        kernel_initializer=kernel_initializer,
        name=f"{name}_initial_model_higher_02_trio",
    )(model_input_higher_02)

    for idx in range(inception_blocks):
        model_higher_02 = inception_block(
            model_higher_02,
            size=sizes[0],
            activation=activation,
            kernel_initializer=kernel_initializer,
            name=f"{name}_trio_model_higher_02_ib_00_{idx}",
        )

    model_higher_02_skip_outer = inception_block(
        model_higher_02,
        size=sizes[0],
        activation=activation,
        kernel_initializer=kernel_initializer,
        name=f"{name}_trio_model_higher_02_ib_01",
    )

    model_higher_02 = reduction_block(
        model_higher_02,
        size=sizes[0],
        activation=activation,
        kernel_initializer=kernel_initializer,
        name=f"{name}_trio_model_higher_02_rb_00",
    )

    for idx in range(inception_blocks):
        model_higher_02 = inception_block(
            model_higher_02,
            size=sizes[1],
            activation=activation,
            kernel_initializer=kernel_initializer,
            name=f"{name}_trio_model_higher_02_ib_02_{idx}",
        )

    model_higher_02_skip_inner = inception_block(
        model_higher_02,
        size=sizes[0],
        activation=activation,
        kernel_initializer=kernel_initializer,
        name=f"{name}_trio_model_higher_02_ib_03",
    )

    model_higher_02 = reduction_block(
        model_higher_02_skip_inner,
        size=sizes[1],
        activation=activation,
        kernel_initializer=kernel_initializer,
        name=f"{name}_trio_higher_02_rb_01",
    )

    for idx in range(inception_blocks):
        model_higher_02 = inception_block(
            model_higher_02,
            size=sizes[2],
            activation=activation,
            kernel_initializer=kernel_initializer,
            name=f"{name}_trio_higher_02_ib_04_{idx}",
        )

    model_higher_02 = expansion_block(
        model_higher_02,
        size=sizes[2],
        activation=activation,
        kernel_initializer=kernel_initializer,
        name=f"{name}_trio_higher_02_eb_00",
    )

    model_higher_02 = Concatenate(name=f"{name}_trio_higher_02_skip_inner_concat")(
        [model_higher_02, model_higher_02_skip_inner]
    )

    for idx in range(inception_blocks):
        model_higher_02 = inception_block(
            model_higher_02,
            size=sizes[1],
            activation=activation,
            kernel_initializer=kernel_initializer,
            name=f"{name}_trio_higher_02_ib_05_{idx}",
        )

    model_higher_02 = expansion_block(
        model_higher_02,
        size=sizes[1],
        activation=activation,
        kernel_initializer=kernel_initializer,
        name=f"{name}_trio_higher_02_eb_01",
    )

    model_higher_02 = Concatenate(name=f"{name}_trio_skip_higher_02_outer_concat")(
        [model_higher_02, model_higher_02_skip_outer]
    )

    for idx in range(inception_blocks):
        model_higher_02 = inception_block(
            model_higher_02,
            size=sizes[0],
            activation=activation,
            kernel_initializer=kernel_initializer,
            name=f"{name}_trio_higher_02_ib_06_{idx}",
        )

    model = Concatenate(name=f"{name}_trio_higher_merge")([model, model_higher_02])

    model = inception_block(
        model,
        size=sizes[0],
        activation=activation,
        kernel_initializer=kernel_initializer,
        name=f"{name}_tail_trio_ib_00",
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

    return Model(
        inputs=[
            model_input_higher_01,
            model_input_higher_02,
            model_input_model_lower,
        ],
        outputs=output,
    )
