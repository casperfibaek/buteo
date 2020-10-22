import sys; sys.path.append('../../lib/')
import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import ml_utils


import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_probability as tfp
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

import numpy as np
import matplotlib.pyplot as plt

folder = "C:/Users/caspe/Desktop/Paper_2_StruturalDensity/building_analysis/"

db_path = folder + "buildings.sqlite" 
db_cnx = sqlite3.connect(db_path)

df = pd.read_sql_query("SELECT fid, area, perimeter, ipq, vol_sum, hot_mean, sync FROM 'buildings' WHERE area > 10 and vol_sum > 25;", db_cnx)
df["area_norm"] = (df["area"] - df["area"].min()) / (df["area"].max() - df["area"].min())
df["peri_norm"] = (df["perimeter"] - df["perimeter"].min()) / (df["perimeter"].max() - df["perimeter"].min())
df["ipq_norm"] = (df["ipq"] - df["ipq"].min()) / (df["ipq"].max() - df["ipq"].min())

training = df[df["sync"] == 1]
target_base = df[df["sync"] == 0]

# Regression model to find hot_mean
y = training["vol_sum"]
y = y.values

X = training.drop(["fid", "vol_sum", "hot_mean", "area", "perimeter", "sync", "ipq"], axis=1)
X = X.values

min_volume = 10
average_volume = 500
max_volume = average_volume * 4

labels = [*range(average_volume, max_volume, average_volume)]
truth_labels = np.digitize(y, labels)
minority = ml_utils.count_freq(truth_labels).min(axis=0)[1]
balance_mask = ml_utils.minority_class_mask(truth_labels, minority)

X = X[balance_mask]
y = y[balance_mask]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

n_features = X.shape[1]

# ***********************************************************************
#                   ANALYSIS
# ***********************************************************************

model_input = Input(shape=X.shape[1], name="input")
model = Dense(512, activation=tfa.activations.mish, kernel_initializer="he_normal")(model_input)
model = Dense(128, activation=tfa.activations.mish, kernel_initializer="he_normal")(model)
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
    batch_size=1024,
    validation_split=0.3,
    callbacks=[
        EarlyStopping(
            monitor="val_loss",
            patience=12,
            min_delta=1.0,
            restore_best_weights=True,
        ),
    ]
)

loss, mean_absolute_error, median_absolute_error, absolute_percentage_error = model.evaluate(X_test, y_test, verbose=1)

print("Test accuracy:")
print(f"Mean Absolute Error (MAE): {str(round(mean_absolute_error, 5))}")
print(f"Median Absolute Error (MAE): {str(round(median_absolute_error, 5))}")
print(f"Absolute Percentage Error (MAPE): {str(round(absolute_percentage_error, 5))}")

truth = y_test
predicted = model.predict(X_test).squeeze()

labels = [*range(average_volume, max_volume, average_volume)]

truth_labels = np.digitize(truth, labels)
predicted_labels = np.digitize(predicted, labels)
labels_unique = np.unique(truth_labels) * average_volume
truth_labels_volume = truth_labels * average_volume

residuals = truth - predicted

fig1, ax = plt.subplots()
ax.set_title("violin area")

per_class = []
for cl in labels_unique: per_class.append(residuals[truth_labels_volume == cl])

import pdb; pdb.set_trace()

ax.violinplot(per_class, showextrema=False, showmeans=True, showmedians=True, vert=True, points=500, widths=1)
plt.show()
