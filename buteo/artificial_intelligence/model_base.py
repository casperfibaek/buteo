"""
This is the basic model design used in fibaek et al., 2022.

TODO:
    - add documentation
"""

from tensorflow.keras import Model, Input
from tensorflow.keras.backend import clip
from tensorflow.keras.layers import (
    Conv2D,
    Conv2DTranspose,
    Concatenate,
    BatchNormalization,
    SpatialDropout2D,
    SeparableConv2D,
)
from tensorflow_addons.layers import (
    AdaptiveMaxPooling2D,
    AdaptiveAveragePooling2D,
)

def reduction_block(
    inputs,
    size=32,
    activation="relu",
    kernel_initializer="glorot_normal",
    kernel_regularizer=None,
    name=None,
):
    track1 = AdaptiveAveragePooling2D(
        output_size=[inputs.shape[1] // 2, inputs.shape[2] // 2],
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
        size,
        kernel_size=1,
        padding="same",
        strides=(1, 1),
        activation=activation,
        kernel_initializer=kernel_initializer,
        name=name + "_reduction_t3_0",
    )(inputs)
    track3 = Conv2D(
        size,
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
    track4 = AdaptiveMaxPooling2D(
        output_size=[inputs.shape[1] // 2, inputs.shape[2] // 2],
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


def inception_block_3(
    inputs,
    size=32,
    activation="relu",
    name=None,
    kernel_initializer="glorot_normal",
    kernel_regularizer=None,
    separable=False,
):
    track1 = AdaptiveMaxPooling2D(
        output_size=[inputs.shape[1], inputs.shape[2]],
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
        size,
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
    if separable:
        track4 = SeparableConv2D(
            size,
            kernel_size=1,
            padding="same",
            strides=(1, 1),
            activation=activation,
            kernel_initializer=kernel_initializer,
            name=name + "_inception_t4_0",
        )(inputs)
    else:
        track4 = Conv2D(
            size,
            kernel_size=1,
            padding="same",
            strides=(1, 1),
            activation=activation,
            kernel_initializer=kernel_initializer,
            name=name + "_inception_t4_0",
        )(inputs)
        track4 = Conv2D(
            size,
            kernel_size=1,
            padding="same",
            strides=(1, 1),
            activation=activation,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
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



def inception_block_5(
    inputs,
    size=32,
    activation="relu",
    name=None,
    kernel_initializer="glorot_normal",
    kernel_regularizer=None,
    separable=False,
):
    track1 = AdaptiveMaxPooling2D(
        output_size=[inputs.shape[1], inputs.shape[2]],
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
        size,
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
    if separable:
        track4 = SeparableConv2D(
            size,
            kernel_size=5,
            padding="same",
            strides=(1, 1),
            activation=activation,
            kernel_initializer=kernel_initializer,
            name=name + "_inception_t4_0",
        )(inputs)
    else:
        track4 = Conv2D(
            size,
            kernel_size=1,
            padding="same",
            strides=(1, 1),
            activation=activation,
            kernel_initializer=kernel_initializer,
            name=name + "_inception_t4_0",
        )(inputs)
        track4 = Conv2D(
            size,
            kernel_size=5,
            padding="same",
            strides=(1, 1),
            activation=activation,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            name=name + "_inception_t4_1",
        )(track4)

    return Concatenate(name=f"{name}_inception_concat")(
        [
            track1,
            track2,
            track3,
            track4,
        ]
    )


def create_model(
    shape_higher_01,
    shape_higher_02,
    shape_lower,
    activation="relu",
    output_activation="relu",
    kernel_initializer="glorot_normal",
    sizes=[40, 48, 56],
    inception_blocks=3,
    dropout_rate=0.1,
    name="model_test",
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
        model_higher_01 = inception_block_5(
            model_higher_01,
            size=sizes[0],
            activation=activation,
            kernel_initializer=kernel_initializer,
            name=f"{name}_trio_model_higher_01_ib_00_{idx}",
            separable=True,
        )
    
    model_higher_01 = SpatialDropout2D(dropout_rate)(model_higher_01)
    model_higher_01 = BatchNormalization()(model_higher_01)

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
        model_higher_02 = inception_block_5(
            model_higher_02,
            size=sizes[0],
            activation=activation,
            kernel_initializer=kernel_initializer,
            name=f"{name}_trio_model_higher_02_ib_00_{idx}",
            separable=True,
        )

    model_higher_02 = SpatialDropout2D(dropout_rate)(model_higher_02)
    model_higher_02 = BatchNormalization()(model_higher_02)

    model = Concatenate(name=f"{name}_trio_skip1_merge_concat")(
        [model_higher_01, model_higher_02]
    )

    model_skip_outer = inception_block_5(
        model,
        size=sizes[0],
        activation=activation,
        kernel_initializer=kernel_initializer,
        name=f"{name}_trio_model_higher_01_ib_01",
    )

    model = reduction_block(
        model,
        size=sizes[0],
        activation=activation,
        kernel_initializer=kernel_initializer,
        name=f"{name}_trio_model_higher_01_rb_00",
    )

    for idx in range(inception_blocks):
        model = inception_block_3(
            model,
            size=sizes[1],
            activation=activation,
            kernel_initializer=kernel_initializer,
            name=f"{name}_trio_model_higher_01_ib_02_{idx}",
        )

    model = SpatialDropout2D(dropout_rate)(model)
    model = BatchNormalization()(model)

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
        model_lower = inception_block_3(
            model_lower,
            size=sizes[1],
            activation=activation,
            kernel_initializer=kernel_initializer,
            name=f"{name}_trio_model_lower_ib_00_{idx}",
            separable=True,
        )

    model_lower = SpatialDropout2D(dropout_rate)(model_lower)
    model_lower = BatchNormalization()(model_lower)

    model_skip_inner = Concatenate(name=f"{name}_trio_skip2_merge_concat")(
        [model, model_lower]
    )

    for idx in range(inception_blocks):
        model = inception_block_3(
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
        model = inception_block_3(
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

    model = SpatialDropout2D(dropout_rate)(model)
    model = BatchNormalization()(model)

    model = Concatenate(name=f"{name}_trio_skip_inner_concat")(
        [model, model_skip_inner]
    )

    for idx in range(inception_blocks):
        model = inception_block_3(
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

    model = SpatialDropout2D(dropout_rate)(model)
    model = BatchNormalization()(model)

    model = Concatenate(name=f"{name}_trio_skip_outer_concat")(
        [model, model_skip_outer]
    )

    for idx in range(inception_blocks):
        model = inception_block_5(
            model,
            size=sizes[0],
            activation=activation,
            kernel_initializer=kernel_initializer,
            name=f"{name}_tail_trio_ib_00_{idx}",
        )

    model = inception_block_5(
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
        activation=output_activation,
        kernel_initializer=kernel_initializer,
        name=f"{name}_tail_output",
        dtype="float32",
    )(model)

    return Model(
        inputs=[
            model_input_higher_01,
            model_input_higher_02,
            model_input_model_lower,
        ],
        outputs=clip(output, min_value=0, max_value=100),
    )