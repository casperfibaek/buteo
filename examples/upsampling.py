yellow_follow = 'C:/Users/caspe/Desktop/yellow/buteo/'
import sys; sys.path.append(yellow_follow) 

import os
import math
import numpy as np

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
import tensorflow_addons as tfa

np.set_printoptions(suppress=True)
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

folder = "C:/Users/caspe/Desktop/test/out/"

y = np.load(folder + "B04_10m_patches.npy")

x_s = np.load(folder + "B04_20m_patches.npy")
x_l = np.load(folder + "B03_10m_patches.npy")

# Shuffle the training dataset
shuffle_mask = np.random.permutation(len(y))
y = y[shuffle_mask]
x_s = x_s[shuffle_mask]
x_l = x_l[shuffle_mask]

split = int(y.shape[0] * 0.75)

y_train = y[0:split]
y_test = y[split:]

x_s_train = x_s[0:split]
x_s_test = x_s[split:]

x_l_train = x_l[0:split]
x_l_test = x_l[split:]

def define_model(shape_s, shape_l, name, activation=tfa.activations.mish):
    model_input_s = Input(shape=shape_s, name=name + "_s")
    model_input_l = Input(shape=shape_l, name=name + "_l")

    model_skip1_s = Conv2D(
        32,
        kernel_size=3,
        padding='same',
        activation=activation,
    )(model_input_s)
    model_skip1_l = Conv2D(
        32,
        kernel_size=3,
        padding='same',
        activation=activation,
    )(model_input_l)

    model_s = MaxPooling2D(padding="same")(model_skip1_s)
    model_l = MaxPooling2D(padding="same")(model_skip1_l)

    model_skip2_s = Conv2D(
        48,
        kernel_size=3,
        padding='same',
        activation=activation,
    )(model_s)
    model_skip2_l = Conv2D(
        48,
        kernel_size=3,
        padding='same',
        activation=activation,
    )(model_l)

    model_s = MaxPooling2D(padding="same")(model_skip2_s)
    model_l = MaxPooling2D(padding="same")(model_skip2_l)

    model_s = Conv2D(
        64,
        kernel_size=3,
        padding='same',
        activation=activation,
    )(model_s)
    model_l = Conv2D(
        64,
        kernel_size=3,
        padding='same',
        activation=activation,
    )(model_l)

    model_s = Conv2DTranspose(
        64,
        kernel_size=3,
        strides=(2, 2),
        activation=activation,
        padding="same",
    )(model_s)
    model_l = Conv2DTranspose(
        64,
        kernel_size=3,
        strides=(2, 2),
        activation=activation,
        padding="same",
    )(model_l)

    model_s = Concatenate()([model_skip2_s, model_s])
    model_l = Concatenate()([model_skip2_l, model_l])

    model_s = Conv2DTranspose(
        64,
        kernel_size=3,
        strides=(2, 2),
        activation=activation,
        padding="same",
    )(model_s)
    model_l = Conv2DTranspose(
        64,
        kernel_size=3,
        strides=(2, 2),
        activation=activation,
        padding="same",
    )(model_l)

    model_s = Concatenate()([model_skip1_s, model_s])
    model_l = Concatenate()([model_skip1_l, model_l])

    model_s = Conv2DTranspose(
        24,
        kernel_size=3,
        strides=(2, 2),
        activation=activation,
        padding="same",
    )(model_s)

    model = Concatenate()([model_s, model_l])

    model = Conv2D(
        16,
        kernel_size=5,
        padding='same',
        activation=activation,
    )(model)

    model = Conv2D(
        8,
        kernel_size=3,
        padding='same',
        activation=activation,
    )(model)

    output = Conv2D(
        1,
        kernel_size=3,
        padding='same',
        activation='relu',
    )(model)

    return Model(inputs=[model_input_s, model_input_l], outputs=output)

lr = 0.001
bs = 24
epochs = 1

def step_decay(epoch):
    initial_lrate = lr
    drop = 0.5
    epochs_drop = 3
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate

model = define_model(x_s_train.shape[1:], x_l_train.shape[1:], "Upsampling")

model.compile(
    optimizer=Adam(
        learning_rate=lr,
        name="Adam",
    ),
    loss='log_cosh',
    metrics=['mse', 'mae']
)

model.fit(
    x=[x_s_train, x_l_train],
    y=y_train,
    epochs=epochs,
    verbose=1,
    batch_size=bs,
    validation_split=0.2,
    callbacks=[
        LearningRateScheduler(step_decay),
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            min_delta=0.1,
            restore_best_weights=True,
        ),
    ],
    use_multiprocessing=True,
)

print(f"Batch_size: {str(bs)}")
loss, mse, mae = model.evaluate(x=[x_s_test, x_l_test], y=y_test, verbose=1, batch_size=bs, use_multiprocessing=True)
print(f"Mean Square Error:      {round(mse, 3)}")
print(f"Mean Absolute Error:    {round(mae, 3)}")
print(f"log_cosh:               {round(loss, 3)}")
print("")

