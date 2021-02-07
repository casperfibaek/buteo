# Convolutions!
# Order of images
# b4, b4_tex, b8, b8_tex, bs_asc, bs_desc, coh_asc, coh_desc, nl
# Order of truth
# id, fid, muni_code, volume, area, people

# Local path, change this.
yellow_follow = 'C:/Users/caspe/Desktop/yellow/lib/'

import sys; sys.path.append(yellow_follow) 
import ml_utils
import numpy as np
import os
import math
import time

np.set_printoptions(suppress=True)
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

from sqlalchemy import create_engine

# Tensorflow
import tensorflow_addons as tfa
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.constraints import max_norm

folder = "C:/Users/caspe/Desktop/Paper_2_StructuralVolume/"

images = np.load(folder + "all_images.npy")
truth = np.load(folder + "images_ground_truth.npy")

target_muni = [
    665, # Lemvig
    740, # Silkeborg
    751, # Aarhus
]

target = [
    3, # Volume
    # 4, # Area
    # 5, # People
]

rotation = True
rotation_count = 4

for target in target_muni:

    # Select municipality
    test_muni_mask = (truth[:, 2] == target)
    train_muni_mask = (truth[:, 2] != target)

    X_test = images[test_muni_mask]
    y_test = truth[test_muni_mask]

    X_train = images[train_muni_mask]
    y_train = truth[train_muni_mask]

    all_layers = [
        # { "name": "s2", "layers": [0, 2] },
        # { "name": "bsa", "layers": [4] },
        # { "name": "bsd", "layers": [5] },
        # { "name": "bsa_bsd", "layers": [4, 5] },
        # { "name": "bsac", "layers": [4, 6] },
        # { "name": "bsdc", "layers": [5, 7] },
        # { "name": "bsac_bsdc", "layers": [4, 5, 6, 7] },
        # { "name": "bsac_s2", "layers": [0, 2, 4, 6] },
        # { "name": "bsa_bsd_s2", "layers": [0, 2, 4, 5 },
        { "name": "bsac_bsdc_s2", "layers": [0, 2, 4, 5, 6, 7] },
    ]

    layers = all_layers[0]["layers"]
    layer_name = all_layers[0]["name"]

    # Selected layers
    X_train = X_train[:, :, :, layers]
    y_train = y_train[:, target]

    X_test = X_test[:, :, :, layers]
    y_test = y_test[:, target]

    # Simple balance dataset (Equal amount 0 to rest)
    balance_target = y_train > 0
    frequency = ml_utils.count_freq(balance_target)
    minority = frequency.min(axis=0)[1]
    balance_mask = ml_utils.minority_class_mask(balance_target, minority)

    X_train = X_train[balance_mask]
    y_train = y_train[balance_mask]

    if rotation is True:
        X_train = ml_utils.add_rotations(X_train, axes=(1,2), k=rotation_count)
        y_train = np.concatenate([y_train] * rotation_count)

    # Shuffle the training dataset
    shuffle_mask = np.random.permutation(len(y_train))
    X_train = X_train[shuffle_mask]
    y_train = y_train[shuffle_mask]


    def define_model(shape, name, drop=0.4, activation=tfa.activations.mish, double=True, pool=True, kernel_initializer='normal', maxnorm=3, sizes=[64, 96, 128]):
        model_input = Input(shape=shape, name=name)
        model = Conv2D(sizes[0], kernel_size=3, padding='same', activation=activation, kernel_initializer=kernel_initializer, kernel_constraint=max_norm(maxnorm), bias_constraint=max_norm(maxnorm))(model_input)
        model = BatchNormalization()(model)

        if double:
            model = Conv2D(sizes[0], kernel_size=3, padding='same', activation=activation, kernel_initializer=kernel_initializer, kernel_constraint=max_norm(maxnorm), bias_constraint=max_norm(maxnorm))(model)
            model = BatchNormalization()(model)

        if pool:
            model = MaxPooling2D(pool_size=(2, 2), padding='same', strides=(2, 2))(model)

        model = Conv2D(sizes[1], kernel_size=3, padding='same', activation=activation, kernel_initializer=kernel_initializer, kernel_constraint=max_norm(maxnorm), bias_constraint=max_norm(maxnorm))(model)
        model = BatchNormalization()(model)

        if double:
            model = Conv2D(sizes[1], kernel_size=3, padding='same', activation=activation, kernel_initializer=kernel_initializer, kernel_constraint=max_norm(maxnorm), bias_constraint=max_norm(maxnorm))(model)
            model = BatchNormalization()(model)

        if pool:
            model = MaxPooling2D(pool_size=(2, 2), padding='same', strides=(2, 2))(model)

        model = Conv2D(sizes[2], kernel_size=3, padding='same', activation=activation, kernel_initializer=kernel_initializer, kernel_constraint=max_norm(maxnorm), bias_constraint=max_norm(maxnorm))(model)
        model = BatchNormalization()(model)

        if double:
            model = Conv2D(sizes[2], kernel_size=3, padding='same', activation=activation, kernel_initializer=kernel_initializer, kernel_constraint=max_norm(maxnorm), bias_constraint=max_norm(maxnorm))(model)
            model = BatchNormalization()(model)

        model = Flatten()(model)
        model = Dropout(drop)(model)

        model = Dense(256, activation='relu', kernel_initializer=kernel_initializer)(model)
        model = BatchNormalization()(model)
        model = Dropout(drop)(model)

        model = Dense(128, activation='relu', kernel_initializer=kernel_initializer)(model)
        model = BatchNormalization()(model)
        model = Dropout(drop)(model)

        predictions = Dense(1, activation='relu')(model)

        return Model(inputs=[model_input], outputs=predictions)

    LEARNING_RATE = [0.01]
    BATCH_SIZE = [256]
    DOUBLE = [True]
    MAX_NORM = [3]
    KERNEL = ['normal']
    LOSS = ['log_cosh']
    POOL = [True]


    for learning_rate in LEARNING_RATE:
        for batch_size in BATCH_SIZE:
            for maxnorm in MAX_NORM:
                for double in DOUBLE:
                    for kernel in KERNEL:
                        for loss in LOSS:
                            for pool in POOL:
                                before = time.perf_counter()
                                model = define_model(
                                    ml_utils.get_shape(X_train),
                                    "conv2",
                                    # activation=activation,
                                    kernel_initializer=kernel,
                                    maxnorm=maxnorm,
                                    double=double,
                                )

                                def step_decay(epoch):
                                    initial_lrate = 0.01
                                    drop = 0.5
                                    epochs_drop = 5
                                    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
                                    return lrate

                                model.compile(
                                    optimizer=Adam(
                                        learning_rate=learning_rate,
                                        name="Adam",
                                    ),
                                    loss=loss,
                                    metrics=[
                                        'mse',
                                        'mae',
                                        'log_cosh',
                                    ],
                                )

                                model.fit(
                                    x=X_train,
                                    y=y_train,
                                    epochs=100,
                                    verbose=1,
                                    batch_size=batch_size,
                                    validation_split=0.2,
                                    callbacks=[
                                        LearningRateScheduler(step_decay),
                                        EarlyStopping(
                                            monitor="val_loss",
                                            patience=15,
                                            min_delta=1,
                                            restore_best_weights=True,
                                        ),
                                    ]
                                )

                                after = time.perf_counter()

                                _loss, mse, mae, log_cosh = model.evaluate(X_test, y_test, verbose=0)
                                print(f"lr: {str(learning_rate)}, bs: {str(batch_size)}")
                                print(f"Mean Square Error:      {round(mse, 3)}")
                                print(f"Mean Absolute Error:    {round(mae, 3)}")
                                print(f"log_cosh:               {round(log_cosh, 3)}")
                                print(f"Processing took {after - before:0.4f} seconds")
                                print("")