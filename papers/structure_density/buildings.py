import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_probability as tfp
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

folder = "/home/cfi/Desktop/building_analysis/"

db_path = folder + "buildings.sqlite" 
db_cnx = sqlite3.connect(db_path)

df = pd.read_sql_query("SELECT fid, area, perimeter, ipq, vol_sum, hot_mean, sync FROM 'buildings';", db_cnx)
df["area_norm"] = (df["area"] - df["area"].min()) / (df["area"].max() - df["area"].min())
df["peri_norm"] = (df["perimeter"] - df["perimeter"].min()) / (df["perimeter"].max() - df["perimeter"].min())
df["ipq_norm"] = (df["ipq"] - df["ipq"].min()) / (df["ipq"].max() - df["ipq"].min())

training = df[df["sync"] == 1]
target_base = df[df["sync"] == 0]
# target = target_base.drop(["fid", "vol_sum", "hot_mean", "hot", "area", "perimeter"], axis=1)

# Regression model to find hot_mean
y = training["vol_sum"]
y = y.astype('float64')

X = training.drop(["fid", "vol_sum", "hot_mean", "area", "perimeter", "sync", "ipq"], axis=1)
X = X.astype('float64')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

n_features = X.shape[1]

# ***********************************************************************
#                   ANALYSIS
# ***********************************************************************

model_input = Input(shape=X.shape[1], name="input")
model = Dense(128, activation=tfa.activations.mish, kernel_initializer="he_normal")(model_input)
model = Dropout(0.25)(model)
model = Dense(64, activation=tfa.activations.mish, kernel_initializer="he_normal")(model)
model = Dense(32, activation=tfa.activations.mish, kernel_initializer="he_normal")(model)
model = Dense(8, activation=tfa.activations.mish, kernel_initializer="he_normal")(model)

predictions = Dense(1, activation="relu", dtype="float32")(model)

model = Model(inputs=[model_input], outputs=predictions)

optimizer = tfa.optimizers.Lookahead(
    Adam(
        learning_rate=tfa.optimizers.TriangularCyclicalLearningRate(
            initial_learning_rate=1e-4,
            maximal_learning_rate=1e-2,
            step_size=6,
            scale_mode='cycle',
            name='TriangularCyclicalLearningRate',
        ),
        name="Adam",
    )
)

def median_error(y_actual, y_pred):
    return tfp.stats.percentile(tf.math.abs(y_actual - y_pred), 50.0)

def abs_percentage(y_actual, y_pred):
    return tfp.stats.percentile(
        tf.divide(
            tf.abs(tf.subtract(y_actual, y_pred)), (y_actual + 1e-10)
        )
    , 50.0)

model.compile(
    optimizer=optimizer,
    # loss="mean_absolute_error",
    # loss="mean_squared_error",
    # loss=tfa.losses.PinballLoss(),
    loss=tf.keras.losses.Huber(),
    metrics=[
        "mean_absolute_error",
        median_error,
        abs_percentage,
    ])

model.fit(
    x=X_train,
    y=y_train,
    epochs=100,
    verbose=1,
    batch_size=256,
    validation_split=0.3,
    callbacks=[
        EarlyStopping(
            monitor="val_loss",
            patience=15,
            min_delta=1.0,
            restore_best_weights=True,
        ),
    ]
)

# compile the model
# model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

# model.fit(X, y, epochs=500, verbose=1, batch_size=2048)

# y_pred = model.predict(target)

# target_base["vol_sum_pred"] = y_pred
# target_base.to_csv(folder + "vol_sum_pred.csv")

import pdb; pdb.set_trace()