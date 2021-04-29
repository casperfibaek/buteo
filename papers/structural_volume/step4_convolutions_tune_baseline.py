# Convolutions!
# Order of images
# b4, b4_tex, b8, b8_tex, bs_asc, bs_desc, coh_asc, coh_desc, nl
# Order of truth
# id, fid, muni_code, volume, area, people

# Local path, change this.
yellow_follow = 'C:/Users/caspe/Desktop/yellow/lib/'

import sys; sys.path.append(yellow_follow) 
import pandas as pd
import ml_utils
import numpy as np
import os

np.set_printoptions(suppress=True)
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

from sqlalchemy import create_engine

# Tensorflow
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.constraints import max_norm

folder = "C:/Users/caspe/Desktop/Paper_2_StructuralVolume/"

X_train = np.load(folder + "X_train_250000.npy")
y_train = np.load(folder + "y_train_250000.npy")

X_test = np.load(folder + "X_test_silkeborg.npy")
y_test = np.load(folder + "y_test_silkeborg.npy")

def define_model(shape, name, drop=0.2, activation=['relu', 'relu'], kernel_initializer='normal', maxnorm=3, sizes=[64, 96, 128, 256, 128]):
    model_input = Input(shape=shape, name=name)
    model = Conv2D(64, kernel_size=3, padding='same', activation=activation[0], kernel_initializer=kernel_initializer, kernel_constraint=max_norm(maxnorm), bias_constraint=max_norm(maxnorm))(model_input)
    model = BatchNormalization()(model)
    model = Conv2D(64, kernel_size=3, padding='same', activation=activation[0], kernel_initializer=kernel_initializer, kernel_constraint=max_norm(maxnorm), bias_constraint=max_norm(maxnorm))(model)
    model = BatchNormalization()(model)
    model = Dropout(drop)(model)

    model = MaxPooling2D(pool_size=(2, 2), padding='same', strides=(2, 2))(model)

    model = Conv2D(96, kernel_size=3, padding='same', activation=activation[0], kernel_initializer=kernel_initializer, kernel_constraint=max_norm(maxnorm), bias_constraint=max_norm(maxnorm))(model)
    model = BatchNormalization()(model)
    model = Conv2D(96, kernel_size=3, padding='same', activation=activation[0], kernel_initializer=kernel_initializer, kernel_constraint=max_norm(maxnorm), bias_constraint=max_norm(maxnorm))(model)
    model = BatchNormalization()(model)
    model = Dropout(drop)(model)

    model = Conv2D(128, kernel_size=2, padding='same', activation=activation[0], kernel_initializer=kernel_initializer, kernel_constraint=max_norm(maxnorm), bias_constraint=max_norm(maxnorm))(model)
    model = BatchNormalization()(model)
    model = Conv2D(128, kernel_size=2, padding='same', activation=activation[0], kernel_initializer=kernel_initializer, kernel_constraint=max_norm(maxnorm), bias_constraint=max_norm(maxnorm))(model)
    model = BatchNormalization()(model)
    model = Dropout(drop)(model)

    model = Flatten()(model)

    model = Dense(256, activation=activation[1], kernel_initializer=kernel_initializer)(model)
    model = BatchNormalization()(model)
    model = Dropout(drop)(model)
    model = Dense(128, activation=activation[1], kernel_initializer=kernel_initializer)(model)
    model = BatchNormalization()(model)
    model = Dropout(drop)(model)

    predictions = Dense(1, activation='relu')(model)

    return Model(inputs=[model_input], outputs=predictions)

model = define_model(ml_utils.get_shape(X_train), "conv2")

print(model.summary())

model.compile(
    optimizer=Adam(
        learning_rate=0.01,
        name="Adam",
    ),
    loss='log_cosh',
    metrics=[
        'mse',
        'mae',
        'log_cosh',
    ],
)

model.fit(
    x=X_train,
    y=y_train,
    epochs=10,
    verbose=1,
    batch_size=512,
    validation_split=0.2,
    callbacks=[
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_delta=5,
            min_lr=0.00001,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=9,
            min_delta=1,
            restore_best_weights=True,
        ),
    ]
)

zero_mask = y_test > 0

# Evaluate model
print("")
print(model.evaluate(X_test, y_test, verbose=2))
print(model.evaluate(X_test[zero_mask], y_test[zero_mask], verbose=2))
print("")
