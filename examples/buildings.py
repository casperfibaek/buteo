import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import tensorflow
import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout

folder = "C:/Users/caspe/Desktop/Paper_2_StruturalDensity/Data/buildings/"

db_path = folder + "buildings.sqlite" 
db_cnx = sqlite3.connect(db_path)

df = pd.read_sql_query("SELECT fid, vol_sum, hot_mean, area, perimeter, ipq FROM 'buildings';", db_cnx)
df["area_norm"] = (df["area"] - df["area"].min()) / (df["area"].max() - df["area"].min())
df["peri_norm"] = (df["perimeter"] - df["perimeter"].min()) / (df["perimeter"].max() - df["perimeter"].min())
threshold = 1.0

training = df[df["hot_mean"] >= threshold]
target_base = df[df["hot_mean"] < threshold]
target = target_base.drop(["fid", "vol_sum", "hot_mean", "area", "perimeter"], axis=1)

# Regression model to find hot_mean
y = training["hot_mean"]
y = y.astype('float64')

X = training.drop(["fid", "vol_sum", "hot_mean", "area", "perimeter"], axis=1)
X = X.astype('float64')

n_features = X.shape[1]

# hot_mean
model = Sequential()
model.add(Dense(16, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)))
model.add(Dense(12, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(8, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(1, activation='relu'))

# compile the model
model.compile(
    loss='mse',
    optimizer='adam',
    metrics=['mae', 'mse'],
)

model.fit(X, y, epochs=500, verbose=1, batch_size=1024)

y_pred = model.predict(target)

target_base["hot_mean_pred"] = y_pred
target_base.to_csv(folder + "hot_mean_pred.csv")

import pdb; pdb.set_trace()