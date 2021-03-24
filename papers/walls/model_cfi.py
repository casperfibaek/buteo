import numpy as np
import os
import math

from sklearn.model_selection import train_test_split

# Tensorflow
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler

# Visualisation
from matplotlib import pyplot as plt

# Environment
np.set_printoptions(suppress=True)
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

folder = "C:/Users/caspe/Desktop/yellow/papers/walls/"
X = np.load(folder + "dtm.npy")
y = np.load(folder + "walls.npy")

present_mask = np.sum(y, axis=(1, 2)) > 0
X = X[present_mask]
y = y[present_mask]

y = np.multiply(y, 1000)

X = X[..., np.newaxis]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

def define_model(shape, name, activation='swish'):
    model_input = Input(shape=shape, name=name)
    model = Conv2D(
        64,
        kernel_size=3,
        padding='same',
        activation=activation,
    )(model_input)
    model_skip1 = Conv2D(
        64,
        kernel_size=3,
        padding='same',
        activation=activation,
    )(model)

    model = MaxPooling2D(padding="same")(model_skip1)

    model = Conv2D(
        96,
        kernel_size=3,
        padding='same',
        activation=activation,
    )(model)
    model_skip2 = Conv2D(
        96,
        kernel_size=3,
        padding='same',
        activation=activation,
    )(model)

    model = MaxPooling2D(padding="same")(model_skip2)

    model = Conv2D(
        128,
        kernel_size=3,
        padding='same',
        activation=activation,
    )(model)
    model = Conv2D(
        128,
        kernel_size=3,
        padding='same',
        activation=activation,
    )(model)

    model = Conv2DTranspose(
        96,
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

    output = Conv2D(
        1,
        kernel_size=3,
        padding='same',
        activation='relu',
    )(model)

    return Model(inputs=[model_input], outputs=output)

model = define_model(X_train.shape[1:], "bob")

lr = 0.001

model.compile(
    optimizer=Adam(learning_rate=lr),
    loss='mse',
    metrics=['log_cosh'],
)

def step_decay(epoch):
    initial_lrate = lr
    drop = 0.5
    epochs_drop = 5
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate


model.fit(
    X_train,
    y_train,
    epochs=25,
    validation_split=0.2,
    batch_size=32,
    callbacks=[
        LearningRateScheduler(step_decay),
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            min_delta=10,
            restore_best_weights=True,
        ),
    ]
)

pred = model.predict(X_test)

import pdb; pdb.set_trace()
# plt.imshow(pred[0,:,:,0]); plt.show()