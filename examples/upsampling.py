yellow_follow = 'C:/Users/caspe/Desktop/yellow/buteo/'
import sys; sys.path.append(yellow_follow) 

import os
import math
import numpy as np

from sklearn.model_selection import train_test_split

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
import tensorflow_addons as tfa

np.set_printoptions(suppress=True)
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

folder = "C:/Users/caspe/Desktop/test/out/"

X = np.load(folder + "Fyn_B2_20m_patches.npy")
y = np.load(folder + "Fyn_B2_10m_patches.npy")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

def define_model(shape, name, activation=tfa.activations.mish):
    model_input = Input(shape=shape, name=name)
    model_skip1 = Conv2D(
        32,
        kernel_size=3,
        padding='same',
        activation=activation,
    )(model_input)

    model = MaxPooling2D(padding="same")(model_skip1)

    model_skip2 = Conv2D(
        48,
        kernel_size=3,
        padding='same',
        activation=activation,
    )(model)

    model = MaxPooling2D(padding="same")(model_skip2)

    model = Conv2D(
        64,
        kernel_size=3,
        padding='same',
        activation=activation,
    )(model)

    model = Conv2DTranspose(
        64,
        kernel_size=3,
        strides=(2, 2),
        activation=activation,
        padding="same",
    )(model)
    model = Concatenate()([model_skip2, model])

    model = Conv2DTranspose(
        64,
        kernel_size=3,
        strides=(2, 2),
        activation=activation,
        padding="same",
    )(model)
    model = Concatenate()([model_skip1, model])

    model = Conv2DTranspose(
        24,
        kernel_size=3,
        strides=(2, 2),
        activation=activation,
        padding="same",
    )(model)

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

    return Model(inputs=[model_input], outputs=output)

lr = 0.001
bs = 16
epochs = 10

def step_decay(epoch):
    initial_lrate = lr
    drop = 0.5
    epochs_drop = 3
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate

model = define_model(X_train.shape[1:], "Generative")

model.compile(
    optimizer=Adam(
        learning_rate=lr,
        name="Adam",
    ),
    loss='log_cosh',
    metrics=['mse', 'mae']
)

model.fit(
    x=X_train,
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
)

print(f"Batch_size: {str(bs)}")
loss, mse, mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Mean Square Error:      {round(mse, 3)}")
print(f"Mean Absolute Error:    {round(mae, 3)}")
print(f"log_cosh:               {round(loss, 3)}")
print("")

import pdb; pdb.set_trace()
