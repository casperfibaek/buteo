yellow_follow = 'C:/Users/caspe/Desktop/yellow/lib/'
import sys; sys.path.append(yellow_follow) 

import os
import numpy as np
import tensorflow as tf

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import BatchNormalization, Dropout, Dropout, Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

np.set_printoptions(suppress=True)
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

folder = "C:/Users/caspe/Desktop/Paper_6_Generative/"

# Load data
X_train = np.load(folder + "X_train.npy").transpose(0, 2, 3, 1)
y_train = np.load(folder + "y_train.npy")[..., np.newaxis]
X_test = np.load(folder + "X_test.npy").transpose(0, 2, 3, 1)
y_test = np.load(folder + "y_test.npy")[..., np.newaxis]

def define_model(shape, name, activation='relu', kernel_initializer='normal', maxnorm=4, sizes=[64, 96, 128]):
    model_input = Input(shape=shape, name=name)
    model = Conv2D(sizes[0],
        kernel_size=3,
        padding='same',
        activation=activation,
        kernel_initializer=kernel_initializer,
        kernel_constraint=max_norm(maxnorm),
        bias_constraint=max_norm(maxnorm),
    )(model_input)
    model = BatchNormalization()(model)

    model = Conv2D(sizes[1],
        kernel_size=3,
        padding='same',
        activation=activation,
        kernel_initializer=kernel_initializer,
        kernel_constraint=max_norm(maxnorm),
        bias_constraint=max_norm(maxnorm),
    )(model)
    model = BatchNormalization()(model)

    model = Conv2D(sizes[2],
        kernel_size=3,
        padding='same',
        activation=activation,
        kernel_initializer=kernel_initializer,
        kernel_constraint=max_norm(maxnorm),
        bias_constraint=max_norm(maxnorm),
    )(model)
    model = BatchNormalization()(model)

    output = Conv2D(1, kernel_size=3, padding='same', activation='relu', kernel_initializer=kernel_initializer)(model)

    return Model(inputs=[model_input], outputs=output)

lr = 0.0001
bs = 512
epochs = 50

model = define_model(X_train.shape[1:], "Generative")

model.compile(
    optimizer=Adam(
        learning_rate=lr,
        name="Adam",
    ),
    loss='log_cosh',
    metrics=[
        'mse',
        'mae',
        'log_cosh',
    ]
)

model.fit(
    x=X_train,
    y=y_train,
    epochs=epochs,
    verbose=0,
    batch_size=bs,
    validation_split=0.2,
    callbacks=[
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_delta=0.1,
            min_lr=0.00001,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            min_delta=0.1,
            restore_best_weights=True,
        ),
    ],
)

print(f"Batch_size: {str(bs)}")
loss, mse, mae, log_cosh = model.evaluate(X_test, y_test, verbose=0)
print(f"Mean Square Error:      {round(mse, 3)}")
print(f"Mean Absolute Error:    {round(mae, 3)}")
print(f"log_cosh:               {round(log_cosh, 3)}")
print("")

import pdb; pdb.set_trace()
