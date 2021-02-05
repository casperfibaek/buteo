# Convolutions!
# Order of images
# b4, b4_tex, b8, b8_tex, bs_asc, bs_desc, coh_asc, coh_desc, nl
# Order of truth
# id, fid, muni_code, volume, area, people

# Local path, change this.
yellow_follow = 'C:/Users/caspe/Desktop/yellow/lib/'

import sys; sys.path.append(yellow_follow) 
import sqlite3
import pandas as pd
import ml_utils
import numpy as np
import os

np.set_printoptions(suppress=True)
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

from sqlalchemy import create_engine

# Tensorflow
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Dropout, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Flatten, Conv3D, MaxPooling3D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler

folder = "C:/Users/caspe/Desktop/Paper_2_StructuralVolume/"

epochs = 50
initial_learning_rate = 0.001
end_learning_rate = 0.00001

target_munis = [
    665, # Lemvig
    # 740, # Silkeborg
    # 751, # Aarhus
]

targets = [
    3, # Volume
    # 4, # Area
    # 5, # People
]

for target_muni in target_munis:
  for target in targets:

    rotation = True
    rotation_count = 4

    images = np.load(folder + "all_images.npy").transpose(0, 2, 3, 1) # channel-last format of tensorflow
    truth = np.load(folder + "images_ground_truth.npy")

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
        # { "name": "bsa_bsd", "layers": [4, 5] },
        # { "name": "bsac", "layers": [4, 6] },
        # { "name": "bsdc", "layers": [5, 7] },
        { "name": "bsac_bsdc", "layers": [4, 5, 6, 7] },
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

    # train_mask_sub = ml_utils.create_submask(y_train, 100000)
    # X_train = X_train[train_mask_sub]
    # y_train = y_train[train_mask_sub]

    if rotation is True:
        X_train = ml_utils.add_rotations(X_train, axes=(2,3), k=rotation_count)
        y_train = np.concatenate([y_train] * rotation_count)

    # Shuffle the training dataset
    shuffle_mask = np.random.permutation(len(y_train))
    X_train = X_train[shuffle_mask]
    y_train = y_train[shuffle_mask]

    def lr_time_based_decay(epoch):
        return initial_learning_rate - (epoch * ((initial_learning_rate - end_learning_rate) / epochs))

    def define_optimizer():
        return tfa.optimizers.Lookahead(
            Adam(
                learning_rate=initial_learning_rate,
                name="Adam",
            )
        )

    def define_model(shape, name):
        drop = 0.25
        model_input = Input(shape=shape, name=name)
        model = Conv2D(64, kernel_size=3, padding='same', activation=tfa.activations.mish, kernel_initializer='he_uniform')(model_input)
        model = BatchNormalization()(model)
        model = Conv2D(64, kernel_size=3, padding='same', activation=tfa.activations.mish, kernel_initializer='he_uniform')(model)
        model = BatchNormalization()(model)
        model = Dropout(drop)(model)

        model = MaxPooling2D(pool_size=(2, 2), padding='same', strides=(2, 2))(model)

        model = Conv2D(96, kernel_size=3, padding='same', activation=tfa.activations.mish, kernel_initializer='he_uniform')(model)
        model = BatchNormalization()(model)
        model = Conv2D(96, kernel_size=3, padding='same', activation=tfa.activations.mish, kernel_initializer='he_uniform')(model)
        model = BatchNormalization()(model)  
        model = Dropout(drop)(model)

        model = MaxPooling2D(pool_size=(2, 2), padding='same', strides=(2, 2))(model)

        model = Conv2D(128, kernel_size=2, padding='same', activation=tfa.activations.mish, kernel_initializer='he_uniform')(model)
        model = BatchNormalization()(model)
        model = Conv2D(128, kernel_size=2, padding='same', activation=tfa.activations.mish, kernel_initializer='he_uniform')(model)
        model = BatchNormalization()(model)
        model = Dropout(drop)(model)

        model = Flatten()(model)

        model = Dense(256, activation=tfa.activations.mish, kernel_initializer='he_uniform')(model)
        model = BatchNormalization()(model)
        model = Dense(128, activation=tfa.activations.mish, kernel_initializer='he_uniform')(model)
        model = BatchNormalization()(model)
        model = Dropout(drop)(model)

        predictions = Dense(1, activation='relu')(model)

        return Model(inputs=[model_input], outputs=predictions)

    model = define_model(ml_utils.get_shape(X_train), "conv2")

    model.compile(
        optimizer=define_optimizer(),
        loss="mean_absolute_error",
        metrics=[
            "mean_absolute_error",
            # "mean_absolute_percentage_error",
            # ml_utils.median_absolute_error,
            # ml_utils.median_absolute_percentage_error,
        ])

    model.fit(
        x=X_train,
        y=y_train, # area
        epochs=epochs,
        verbose=1,
        batch_size=384,
        validation_split=0.2,
        callbacks=[
            LearningRateScheduler(lr_time_based_decay, verbose=1),
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
    loss, mae = model.evaluate(X_test, y_test, verbose=2)
    print("")
    loss, z_mae = model.evaluate(X_test[zero_mask], y_test[zero_mask], verbose=2)
    print("")

    model.save(f'./models/cnn_{layer_name}_{str(target_muni)}_{str(target)}.h5', include_optimizer=False)

    y_test = truth[test_muni_mask]
    pred = model.predict(X_test)

    y_test = pd.DataFrame(truth[test_muni_mask], columns=[["id", "fid", "muni_code", "volume", "area", "people"]])

    y_test[f"cnn_pred_{str(target)}_{str(target_muni)}"] = pred

    engine = create_engine(f"sqlite:///./predictions/bsac_bsdc_v2/cnn_pred_{layer_name}_{str(target)}_{str(target_muni)}.sqlite", echo=True)
    sqlite_connection = engine.connect()

    y_test.to_sql(f"cnn_pred_{layer_name}_{str(target)}_{str(target_muni)}", sqlite_connection, if_exists='fail')
    sqlite_connection.close()

import pdb; pdb.set_trace()
