# Local path, change this.
yellow_follow = 'C:/Users/caspe/Desktop/yellow/lib/'

import sys; sys.path.append(yellow_follow) 
import sqlite3
import pandas as pd
import ml_utils
import numpy as np
import math

from sqlalchemy import create_engine

# Tensorflow
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import Model, Input
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, Callback

# model_all.save('./models/model_all.h5')
# model_red.save('./models/model_red.h5')

# Load data
folder = "C:/Users/caspe/Desktop/Paper_2_StructuralVolume/"
in_path = folder + "grid.sqlite"

db_cnx = sqlite3.connect(in_path)
df = pd.read_sql_query("SELECT * FROM 'grid';", db_cnx)

# Easy reference to the different features in the datasets.
s2 = [
    'b04_mean', 'b04_stdev', 'b04_min', 'b04_max',
    'b08_mean', 'b08_stdev', 'b08_min', 'b08_max',
    'b04t_mean', 'b04t_stdev', 'b04t_min', 'b04t_max',
    'b08t_mean', 'b08t_stdev', 'b08t_min', 'b08t_max',
]

s2_nt = [
    'b04_mean', 'b04_stdev', 'b04_min', 'b04_max',
    'b08_mean', 'b08_stdev', 'b08_min', 'b08_max',
]

bs_asc = ['bs_asc_mean', 'bs_asc_stdev', 'bs_asc_min', 'bs_asc_max']
bs_desc = ['bs_desc_mean', 'bs_desc_stdev', 'bs_desc_min', 'bs_desc_max']
coh_asc = ['coh_asc_mean', 'coh_asc_stdev', 'coh_asc_min', 'coh_asc_max']
coh_desc = ['coh_desc_mean', 'coh_desc_stdev', 'coh_desc_min', 'coh_desc_max']

nl = ['nl_mean', 'nl_stdev', 'nl_min', 'nl_max']

# The municipalities used as test targets
test_municipalities = [
    # 'Lemvig',       # Rural
    # 'Silkeborg',    # Mixed
    'Aarhus',       # Urban
]

targets = [
    "area",
]

analysis_to_run = [
    { "name": "bsac_bsdc_s2", "layers": bs_asc + coh_asc + bs_desc + coh_desc + s2 },
]

all_analysis = []

for analysis in analysis_to_run:
    for target in targets:

        if target == "volume":
            min_delta = 4.0
        elif target == "area":
            min_delta = 1.0
        elif target == "people":
            min_delta = 0.5
        else:
            min_delta = 1.0

        analysis_name = analysis["name"]
        analysis_layers = analysis["layers"]

        scores = { "name": analysis_name }

        for muni in test_municipalities:
            test = df[df['muni_name'] == muni]
            train = df[df['muni_name'] != muni]

            muni_code = str(int(test['muni_code'].iloc[0]))
            
            X_train = train[analysis_layers].values
            X_test = test[analysis_layers].values

            y_train = train[target].values
            y_test = test[target].values

            model_all = load_model('./models/model_all.h5')
            model_red = load_model('./models/model_red.h5')

            for layer in model_all.layers: layer.trainable = False
            for layer in model_red.layers: layer.trainable = False

            model_all_top = tf.keras.Sequential(model_all.layers[:-1])
            model_red_top = tf.keras.Sequential(model_red.layers[:-1])

            model = Concatenate()([model_all_top.output, model_red_top.output])
            model = Dense(32, activation=tfa.activations.mish, kernel_initializer="he_normal")(model)
            model = Dense(16, activation=tfa.activations.mish, kernel_initializer="he_normal")(model)
            model = Dense(1, activation='relu')(model)

            n_model = Model([model_all_top.input, model_red_top.input], model)

            initial_learning_rate = 0.01
            end_learning_rate = 0.00001
            mix_epochs = 50

            def lr_time_based_decay(epoch):
                return initial_learning_rate - (epoch * ((initial_learning_rate - end_learning_rate) / mix_epochs))

            n_model.compile(
                Adam(
                    learning_rate=initial_learning_rate,
                    name="Adam",
                ),
                loss="mean_absolute_error",
                metrics=[
                    "mean_absolute_error",
                    "mean_absolute_percentage_error",
                    ml_utils.median_absolute_error,
                    # ml_utils.median_absolute_percentage_error,
                ])

            n_model.fit(
                x=[X_train, X_train],
                y=y_train,
                epochs=mix_epochs,
                verbose=2,
                batch_size=1024,
                validation_split=0.2,
                callbacks=[
                    LearningRateScheduler(lr_time_based_decay, verbose=1),
                    EarlyStopping(
                        monitor="val_loss",
                        patience=16,
                        min_delta=min_delta,
                        restore_best_weights=True,
                    ),
                ]
            )

            print(n_model.evaluate([X_test, X_test], y_test))

            import pdb; pdb.set_trace()

            # predictions = np.rot90(np.array([np.rot90(model_all.predict(X_test))[0], np.rot90(model_red.predict(X_test))[0], np.rot90(n_model.predict([X_test, X_test]))[0]]))
            