import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import tensorflow
import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

folder = "C:/Users/caspe/Desktop/Paper_2_StruturalDensity/Data/buildings/"

db_path = folder + "buildings.sqlite" 
db_cnx = sqlite3.connect(db_path)

df = pd.read_sql_query("SELECT fid, vol_sum, hot_mean, hot, area, perimeter, ipq FROM 'buildings';", db_cnx)
df["area_norm"] = (df["area"] - df["area"].min()) / (df["area"].max() - df["area"].min())
df["peri_norm"] = (df["perimeter"] - df["perimeter"].min()) / (df["perimeter"].max() - df["perimeter"].min())
df["hot_norm"] = (df["hot"] - df["hot"].min()) / (df["hot"].max() - df["hot"].min())
threshold = 1.0

training = df[df["hot_mean"] >= threshold]
target_base = df[df["hot_mean"] < threshold]
target = target_base.drop(["fid", "vol_sum", "hot_mean", "hot", "area", "perimeter"], axis=1)

# Regression model to find hot_mean
y = training["vol_sum"]
y = y.astype('float64')

X = training.drop(["fid", "vol_sum", "hot_mean", "hot", "area", "perimeter"], axis=1)
X = X.astype('float64')

import pdb; pdb.set_trace()

n_features = X.shape[1]

# define model
model = Sequential([
    Dense(64, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)),
    Dense(32, activation='relu', kernel_initializer='he_normal'),
    Dense(16, activation='relu', kernel_initializer='he_normal'),
    Dense(8, activation='relu', kernel_initializer='he_normal'),
    Dense(1),
])

# compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

model.fit(X, y, epochs=500, verbose=1, batch_size=2048)

y_pred = model.predict(target)

# target_base["vol_sum_pred"] = y_pred
# target_base.to_csv(folder + "vol_sum_pred.csv")

import pdb; pdb.set_trace()