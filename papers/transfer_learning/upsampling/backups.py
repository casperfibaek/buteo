# Mean Square Error:      3676.862
# Mean Absolute Error:    24.417


def define_model(
    shape_10m,
    shape_20m,
    name,
    activation="Mish",
    # filter_size=32,
    filter_size=[32, 32, 32],
    skip_connections=True,
    max_pooling=False,
    kernel_initializer="glorot_normal",
):
    input_10m = Input(shape=shape_10m, name=name + "_10m")
    input_20m = Input(shape=shape_20m, name=name + "_20m")

    # -------------- 10m -------------------
    model_10m = Conv2D(
        filter_size[0],
        kernel_size=3,
        padding="same",
        activation=activation,
        kernel_initializer=kernel_initializer,
    )(input_10m)

    model_10m = Conv2D(
        filter_size[0],
        kernel_size=3,
        padding="same",
        activation=activation,
        kernel_initializer=kernel_initializer,
    )(model_10m)

    if max_pooling:
        model_10m = MaxPooling2D(padding="same")(model_10m)
    else:
        model_10m = AveragePooling2D(padding="same")(model_10m)

    model_10m_skip_64 = Conv2D(
        filter_size[1],
        kernel_size=3,
        padding="same",
        activation=activation,
        kernel_initializer=kernel_initializer,
    )(model_10m)

    model_10m = Conv2D(
        filter_size[1],
        kernel_size=3,
        padding="same",
        activation=activation,
        kernel_initializer=kernel_initializer,
    )(model_10m)

    if max_pooling:
        model_10m = MaxPooling2D(padding="same")(model_10m)
    else:
        model_10m = AveragePooling2D(padding="same")(model_10m)

    model_10m = Conv2D(
        filter_size[2],
        kernel_size=3,
        padding="same",
        activation=activation,
        kernel_initializer=kernel_initializer,
    )(model_10m)

    model_10m = Conv2D(
        filter_size[2],
        kernel_size=3,
        padding="same",
        activation=activation,
        kernel_initializer=kernel_initializer,
    )(model_10m)

    model_10m = Conv2DTranspose(
        filter_size[2],
        kernel_size=3,
        strides=(2, 2),
        activation=activation,
        kernel_initializer=kernel_initializer,
        padding="same",
    )(model_10m)

    if skip_connections:
        model_10m = Concatenate()([model_10m, model_10m_skip_64])

    # -------------- 20m -------------------
    model_20m_skip_64 = Conv2D(
        filter_size[1],
        kernel_size=3,
        padding="same",
        activation=activation,
        kernel_initializer=kernel_initializer,
    )(input_20m)

    model_20m = Conv2D(
        filter_size[1],
        kernel_size=3,
        padding="same",
        activation=activation,
        kernel_initializer=kernel_initializer,
    )(model_20m_skip_64)

    if max_pooling:
        model_20m = MaxPooling2D(padding="same")(model_20m)
    else:
        model_20m = AveragePooling2D(padding="same")(model_20m)

    model_20m = Conv2D(
        filter_size[2],
        kernel_size=3,
        padding="same",
        activation=activation,
        kernel_initializer=kernel_initializer,
    )(model_20m)

    model_20m = Conv2D(
        filter_size[2],
        kernel_size=3,
        padding="same",
        activation=activation,
        kernel_initializer=kernel_initializer,
    )(model_20m)

    model_20m = Conv2DTranspose(
        filter_size[2],
        kernel_size=3,
        strides=(2, 2),
        activation=activation,
        kernel_initializer=kernel_initializer,
        padding="same",
    )(model_20m)

    if skip_connections:
        model_20m = Concatenate()([model_20m, model_20m_skip_64])

    # -------------- merged -------------------
    model = Concatenate()([model_10m, model_20m])

    model = Conv2D(
        filter_size[2],
        kernel_size=3,
        padding="same",
        activation=activation,
        kernel_initializer=kernel_initializer,
    )(model)

    model = Conv2D(
        filter_size[2],
        kernel_size=3,
        padding="same",
        activation=activation,
        kernel_initializer=kernel_initializer,
    )(model)

    model = Conv2DTranspose(
        filter_size[2],
        kernel_size=3,
        strides=(2, 2),
        activation=activation,
        kernel_initializer=kernel_initializer,
        padding="same",
    )(model)

    model = Conv2D(
        filter_size[2],
        kernel_size=3,
        padding="same",
        activation=activation,
        kernel_initializer=kernel_initializer,
    )(model)

    model = Conv2D(
        filter_size[2],
        kernel_size=3,
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

    return Model(inputs=[input_10m, input_20m], outputs=output)
