# Local path, change this.
yellow_follow = 'C:/Users/caspe/Desktop/yellow/lib/'

import sys; sys.path.append(yellow_follow) 
import sqlite3
import pandas as pd
import ml_utils
import numpy as np
import datetime

from sqlalchemy import create_engine

# Tensorflow
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler

start_time = datetime.datetime.now()

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
    'Lemvig',       # Rural
    'Silkeborg',    # Mixed
    'Aarhus',       # Urban
]

targets = [
    "volume",
    "area",
    "people",
]

analysis_to_run = [
    { "name": "nl", "layers": nl },
    { "name": "s2", "layers": s2 },
    { "name": "s2_notex", "layers": s2_nt },
    { "name": "bsa", "layers": bs_asc },
    { "name": "bsd", "layers": bs_desc },
    { "name": "bsa_bsd", "layers": bs_asc + bs_desc },
    { "name": "bsac", "layers": bs_asc + coh_asc },
    { "name": "bsdc", "layers": bs_desc + coh_desc },
    { "name": "bsac_bsdc", "layers": bs_asc + coh_asc + bs_desc + coh_desc },
    { "name": "bsac_s2", "layers": bs_asc + coh_asc + s2 },
    { "name": "bsa_bsd_s2", "layers": bs_asc + bs_desc + s2 },
    { "name": "bsac_bsdc_s2", "layers": bs_asc + coh_asc + bs_desc + coh_desc + s2 },
    { "name": "bsac_bsdc_s2_nl", "layers": bs_asc + coh_asc + bs_desc + coh_desc + s2 + nl },
]

epochs = 60
initial_learning_rate = 0.01
end_learning_rate = 0.00001
save_results = True
logfile_name = 'log.txt'

# Define model
def define_model(shape, name):
    model_input = Input(shape=shape, name=name)
    model = Dense(1024, activation="swish", kernel_initializer="he_normal", name=f"{name}_dense1024")(model_input)

    model = Dropout(0.2, name=f"{name}_dropout02")(model)
    model = BatchNormalization(name=f"{name}_batchnorm1")(model)

    model = Dense(256, activation="swish", kernel_initializer="he_normal", name=f"{name}_dense256")(model)

    model = BatchNormalization(name=f"{name}_batchnorm2")(model)

    model = Dense(64, activation="swish", kernel_initializer="he_normal", name=f"{name}_dense64")(model)
    model = Dense(16, activation="swish", kernel_initializer="he_normal", name=f"{name}_dense16")(model)

    model = BatchNormalization(name=f"{name}_batchnorm3")(model)

    predictions = Dense(1, activation='relu', name=f"{name}_pred")(model)
    
    return Model(inputs=[model_input], outputs=predictions)


def lr_time_based_decay(epoch):
    return initial_learning_rate - (epoch * ((initial_learning_rate - end_learning_rate) / epochs))

# Define Optimizer
def define_optimizer():
    return tfa.optimizers.Lookahead(
        Adam(
            learning_rate=initial_learning_rate,
            name="Adam",
        )
    )

def train_model(model, x, y, epochs):
    # Compile and test models
    model.compile(
        optimizer=define_optimizer(),
        loss="mean_absolute_error",
        metrics=[
            "mean_absolute_error",
            "mean_absolute_percentage_error",
            ml_utils.median_absolute_error,
            ml_utils.median_absolute_percentage_error,
        ])

    model.fit(
        x=x,
        y=y,
        epochs=epochs,
        verbose=2,
        batch_size=2048,
        validation_split=0.1,
        callbacks=[
            LearningRateScheduler(lr_time_based_decay, verbose=1),
            EarlyStopping(
                monitor="val_loss",
                patience=5,
                min_delta=min_delta,
                restore_best_weights=True,
            ),
        ]
    )

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

        scores = { "name": analysis_name, "target": target }

        for muni in test_municipalities:
            test = df[df['muni_name'] == muni]
            train = df[df['muni_name'] != muni]

            # Split dataset into two. One with all data, one with most zero tiles removed
            nonzero_tiles = len(train[train[target] != 0])
            ten_percent = train[train[target] == 0].sample(int(nonzero_tiles * 0.10))
            train_reduced = train[train[target] != 0].append(ten_percent)
            del nonzero_tiles, ten_percent

            muni_code = str(int(test['muni_code'].iloc[0]))
            
            X_train = train[analysis_layers].values
            X_train_red = train_reduced[analysis_layers].values
            X_test = test[analysis_layers].values

            y_train = train[target].values
            y_train_red = train_reduced[target].values
            y_test = test[target].values

            model_all = define_model(X_train.shape[1], "all")
            model_red = define_model(X_train_red.shape[1], "red")

            train_model(model_all, X_train, y_train, epochs)
            train_model(model_red, X_train_red, y_train_red, epochs)

            for layer in model_all.layers: layer.trainable = False
            for layer in model_red.layers: layer.trainable = False

            model_all_top = tf.keras.Sequential(model_all.layers[:-1])
            model_red_top = tf.keras.Sequential(model_red.layers[:-1])

            model = Concatenate()([model_all_top.output, model_red_top.output])
            model = Dense(32, activation=tfa.activations.mish, kernel_initializer="he_normal")(model)
            model = Dense(16, activation=tfa.activations.mish, kernel_initializer="he_normal")(model)
            model = Dense(1, activation="relu", kernel_initializer="he_normal")(model)

            model_mix = Model([model_all_top.input, model_red_top.input], model)

            model_mix.compile(
                optimizer=tfa.optimizers.Lookahead(
                    Adam(
                        learning_rate=initial_learning_rate,
                        name="Adam",
                    )
                ),
                loss="mean_absolute_error",
                metrics=[
                    "mean_absolute_error",
                    "mean_absolute_percentage_error",
                    ml_utils.median_absolute_error,
                    ml_utils.median_absolute_percentage_error,
                ])

            model_mix.fit(
                x=[X_train, X_train],
                y=y_train,
                epochs=epochs,
                verbose=2,
                batch_size=1024,
                validation_split=0.1,
                callbacks=[
                    LearningRateScheduler(lr_time_based_decay, verbose=1),
                    EarlyStopping(
                        monitor="val_loss",
                        patience=9,
                        min_delta=min_delta,
                        restore_best_weights=True,
                    ),
                ]
            )

            zero_mask = y_test > 0

            # Evaluate model
            loss, mae, mape, meae, meape = model_mix.evaluate([X_test, X_test], y_test, verbose=1)
            loss, z_mae, z_mape, z_meae, z_meape = model_mix.evaluate([X_test[zero_mask], X_test[zero_mask]], y_test[zero_mask], verbose=1)

            scores[muni] = {
                "mean_absolute_error": mae,
                "mean_absolute_percentage_error": mape,
                "median_absolute_error": meae,
                "median_absolute_percentage_error": meape,
                "z_mean_absolute_error": z_mae,
                "z_mean_absolute_percentage_error": z_mape,
                "z_median_absolute_error": z_meae,
                "z_median_absolute_percentage_error": z_meape,
            }

            if save_results:
                save_name = f"{analysis_name}_{target}_{muni_code}"
                # Save the predicted values
                pred = model_mix.predict([X_test, X_test])
                test[f"grid_{save_name}"] = pred
                save_file = test[["fid", f"grid_{save_name}", target]]

                engine = create_engine(f"sqlite:///{folder}results_merge/grid_{save_name}.sqlite", echo=True)
                sqlite_connection = engine.connect()

                save_file.to_sql(f"grid_{save_name}", sqlite_connection, if_exists='fail')
                sqlite_connection.close()

        all_analysis.append(scores)

print("Writing to log file now..")

import sys
sys.stdout = open(logfile_name, 'w')
end_time = datetime.datetime.now()

for i, analysis in enumerate(all_analysis):
    analysis_name = analysis["name"]
    analysis_target = analysis["target"]

    print(f"Analysis: {analysis_name}, target: {analysis_target}")
    print("")

    combined = {
        "MAE": [],
        "MAPE": [],
        "MeAE": [],
        "MeAPE": [],
        "Z_MAE": [],
        "Z_MAPE": [],
        "Z_MeAE": [],
        "Z_MeAPE": [],
    }

    for munipality in test_municipalities:
        test_area_name = munipality
        test_data = analysis[test_area_name]

        mean_absolute_error = test_data["mean_absolute_error"]
        mean_absolute_percentage_error = test_data["mean_absolute_percentage_error"]
        median_absolute_error = test_data["median_absolute_error"]
        median_absolute_percentage_error = test_data["median_absolute_percentage_error"]

        z_mean_absolute_error = test_data["z_mean_absolute_error"]
        z_mean_absolute_percentage_error = test_data["z_mean_absolute_percentage_error"]
        z_median_absolute_error = test_data["z_median_absolute_error"]
        z_median_absolute_percentage_error = test_data["z_median_absolute_percentage_error"]

        combined["MAE"].append(mean_absolute_error)
        combined["MAPE"].append(mean_absolute_percentage_error)
        combined["MeAE"].append(median_absolute_error)
        combined["MeAPE"].append(median_absolute_percentage_error)

        combined["Z_MAE"].append(z_mean_absolute_error)
        combined["Z_MAPE"].append(z_mean_absolute_percentage_error)
        combined["Z_MeAE"].append(z_median_absolute_error)
        combined["Z_MeAPE"].append(z_median_absolute_percentage_error)
    
        print(f"    Test area: {test_area_name}")
        print(f"    Mean Absolute Error (MAE):                {ml_utils.pad(str(round(mean_absolute_error, 3)), 5, 3)}")
        print(f"    Mean Absolute Percentage Error (MAPE):    {ml_utils.pad(str(round(mean_absolute_percentage_error, 3)), 5, 3)}")
        print(f"    Median Absolute Error (MeAE):             {ml_utils.pad(str(round(median_absolute_error, 3)), 5, 3)}")
        print(f"    Median Absolute Percentage Error (MeAPE): {ml_utils.pad(str(round(median_absolute_percentage_error, 3)), 5, 3)}")
        
        print(f"  Z Mean Absolute Error (MAE):                {ml_utils.pad(str(round(z_mean_absolute_error, 3)), 5, 3)}")
        print(f"  Z Mean Absolute Percentage Error (MAPE):    {ml_utils.pad(str(round(z_mean_absolute_percentage_error, 3)), 5, 3)}")
        print(f"  Z Median Absolute Error (MeAE):             {ml_utils.pad(str(round(z_median_absolute_error, 3)), 5, 3)}")
        print(f"  Z Median Absolute Percentage Error (MeAPE): {ml_utils.pad(str(round(z_median_absolute_percentage_error, 3)), 5, 3)}")
        print("")
    
    mae_mean = np.array(combined['MAE']).mean()
    mae_std = np.array(combined['MAE']).std()

    mape_mean = np.array(combined['MAPE']).mean()
    mape_std = np.array(combined['MAPE']).std()

    meae_mean = np.array(combined['MeAE']).mean()
    meae_std = np.array(combined['MeAE']).std()

    meape_mean = np.array(combined['MeAPE']).mean()
    meapee_std = np.array(combined['MeAPE']).std()

    z_mae_mean = np.array(combined['Z_MAE']).mean()
    z_mae_std = np.array(combined['Z_MAE']).std()

    z_mape_mean = np.array(combined['Z_MAPE']).mean()
    z_mape_std = np.array(combined['Z_MAPE']).std()

    z_meae_mean = np.array(combined['Z_MeAE']).mean()
    z_meae_std = np.array(combined['Z_MeAE']).std()

    z_meape_mean = np.array(combined['Z_MeAPE']).mean()
    z_meapee_std = np.array(combined['Z_MeAPE']).std()

    print(f"    Combined Score:")
    print(f"    Mean Absolute Error (MAE):               {ml_utils.pad(str(round(mae_mean, 3)), 5, 3)} ({ml_utils.pad(str(round(mae_std, 3)), 5, 3)} stdev)")
    print(f"    Mean Absolute Percentage Error (MAPE):   {ml_utils.pad(str(round(mape_mean, 3)), 5, 3)} ({ml_utils.pad(str(round(mape_std, 3)), 5, 3)} stdev)")
    print(f"    Median Absolute Error (MeAE):            {ml_utils.pad(str(round(meae_mean, 3)), 5, 3)} ({ml_utils.pad(str(round(meae_std, 3)), 5, 3)} stdev)")
    print(f"    Median Absolute Percentage Error (MAPE): {ml_utils.pad(str(round(meape_mean, 3)), 5, 3)} ({ml_utils.pad(str(round(meapee_std, 3)), 5, 3)} stdev)")

    print(f"  Z Mean Absolute Error (MAE):               {ml_utils.pad(str(round(z_mae_mean, 3)), 5, 3)} ({ml_utils.pad(str(round(z_mae_std, 3)), 5, 3)} stdev)")
    print(f"  Z Mean Absolute Percentage Error (MAPE):   {ml_utils.pad(str(round(z_mape_mean, 3)), 5, 3)} ({ml_utils.pad(str(round(z_mape_std, 3)), 5, 3)} stdev)")
    print(f"  Z Median Absolute Error (MeAE):            {ml_utils.pad(str(round(z_meae_mean, 3)), 5, 3)} ({ml_utils.pad(str(round(z_meae_std, 3)), 5, 3)} stdev)")
    print(f"  Z Median Absolute Percentage Error (MAPE): {ml_utils.pad(str(round(z_meape_mean, 3)), 5, 3)} ({ml_utils.pad(str(round(z_meapee_std, 3)), 5, 3)} stdev)")
    print("")

print(f"Start: {start_time}")
print(f"End:   {end_time}")

import pdb; pdb.set_trace()
