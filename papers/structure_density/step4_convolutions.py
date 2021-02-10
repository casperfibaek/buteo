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
import math
import time
import numpy as np
import os

np.set_printoptions(suppress=True)
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

from sqlalchemy import create_engine

# Tensorflow
import tensorflow_addons as tfa
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.constraints import max_norm

folder = "C:/Users/caspe/Desktop/Paper_2_StructuralVolume/"

target_munis = [
    665, # Lemvig
    740, # Silkeborg
    751, # Aarhus
]

targets = [
    3, # Volume
    # 4, # Area
    # 5, # People
]

images = np.load(folder + "all_images.npy").transpose(0, 2, 3, 1) # channel-last format of tensorflow
truth = np.load(folder + "images_ground_truth.npy")

for target_muni in target_munis:
    for target in targets:

        before = time.perf_counter()
        rotation = True
        rotation_count = 4

        # Select municipality
        test_muni_mask = (truth[:, 2] == target_muni)
        train_muni_mask = (truth[:, 2] != target_muni)

        X_test = images[test_muni_mask]
        y_test = truth[test_muni_mask]

        X_train = images[train_muni_mask]
        y_train = truth[train_muni_mask]

        all_layers = [
            # { "name": "s2", "layers": [0, 2] },
            # { "name": "bsa", "layers": [4] },
            # { "name": "bsd", "layers": [5] },
            { "name": "bsa_bsd", "layers": [4, 5] },
            # { "name": "bsac", "layers": [4, 6] },
            # { "name": "bsdc", "layers": [5, 7] },
            # { "name": "bsac_bsdc", "layers": [4, 5, 6, 7] },
            # { "name": "bsac_s2", "layers": [0, 2, 4, 6] },
            # { "name": "bsa_bsd_s2", "layers": [0, 2, 4, 5 },
            # { "name": "bsac_bsdc_s2", "layers": [0, 2, 4, 5, 6, 7] },
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

        # train_mask_sub = ml_utils.create_submask(y_train, 100000)
        # X_train = X_train[train_mask_sub]
        # y_train = y_train[train_mask_sub]

        def define_model(shape, name, drop=0.2, activation=tfa.activations.mish, kernel_initializer='normal', maxnorm=4, sizes=[64, 96, 128]):
            model_input = Input(shape=shape, name=name)
            model = Conv2D(sizes[0], kernel_size=3, padding='same', activation=activation, kernel_initializer=kernel_initializer, kernel_constraint=max_norm(maxnorm), bias_constraint=max_norm(maxnorm))(model_input)
            model = BatchNormalization()(model)

            model = Conv2D(sizes[0], kernel_size=3, padding='same', activation=activation, kernel_initializer=kernel_initializer, kernel_constraint=max_norm(maxnorm), bias_constraint=max_norm(maxnorm))(model)
            model = BatchNormalization()(model)

            model = MaxPooling2D(pool_size=(2, 2), padding='same', strides=(2, 2))(model)

            model = Conv2D(sizes[1], kernel_size=3, padding='same', activation=activation, kernel_initializer=kernel_initializer, kernel_constraint=max_norm(maxnorm), bias_constraint=max_norm(maxnorm))(model)
            model = BatchNormalization()(model)

            model = Conv2D(sizes[1], kernel_size=3, padding='same', activation=activation, kernel_initializer=kernel_initializer, kernel_constraint=max_norm(maxnorm), bias_constraint=max_norm(maxnorm))(model)
            model = BatchNormalization()(model)

            model = MaxPooling2D(pool_size=(2, 2), padding='same', strides=(2, 2))(model)

            model = Conv2D(sizes[2], kernel_size=3, padding='same', activation=activation, kernel_initializer=kernel_initializer, kernel_constraint=max_norm(maxnorm), bias_constraint=max_norm(maxnorm))(model)
            model = BatchNormalization()(model)

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

        def step_decay(epoch):
            initial_lrate = 0.01
            drop = 0.5
            epochs_drop = 5
            lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
            return lrate


        model = define_model(ml_utils.get_shape(X_train), "conv2")

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
            epochs=50,
            verbose=1,
            batch_size=256,
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

        model.evaluate(X_test, y_test, verbose=2)

        # Evaluate model
        after = time.perf_counter()

        _loss, mse, mae, log_cosh = model.evaluate(X_test, y_test, verbose=0)
        print(f"Municipality: {str(target_muni)}")
        print(f"Mean Square Error:      {round(mse, 3)}")
        print(f"Mean Absolute Error:    {round(mae, 3)}")
        print(f"log_cosh:               {round(log_cosh, 3)}")
        print(f"Processing took {after - before:0.4f} seconds")
        print("")

        y_test = truth[test_muni_mask]
        pred = model.predict(X_test)

        y_test = pd.DataFrame(truth[test_muni_mask], columns=[["id", "fid", "muni_code", "volume", "area", "people"]])

        y_test[f"cnn_pred_{str(target)}_{str(target_muni)}"] = pred

        engine = create_engine(f"sqlite:///./predictions/bsa_bsd/cnn_pred_{layer_name}_{str(target)}_{str(target_muni)}.sqlite", echo=True)
        sqlite_connection = engine.connect()

        y_test.to_sql(f"cnn_pred_{layer_name}_{str(target)}_{str(target_muni)}", sqlite_connection, if_exists='fail')
        sqlite_connection.close()

import pdb; pdb.set_trace()
