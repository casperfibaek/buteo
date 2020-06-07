import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import tensorflow
import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout

folder = "/mnt/c/users/caspe/Desktop/Paper_2_StruturalDensity/Data/buildings/"

db_path = folder + "buildings.sqlite" 
db_cnx = sqlite3.connect(db_path)

df = pd.read_sql_query("SELECT fid, vol_sum, hot_mean, area, perimeter, ipq FROM 'buildings' LIMIT 10000;", db_cnx)
df["area_norm"] = (df["area"] - df["area"].min()) / (df["area"].max() - df["area"].min())
df["peri_norm"] = (df["perimeter"] - df["perimeter"].min()) / (df["perimeter"].max() - df["perimeter"].min())
threshold = 1.0

training = df[df["hot_mean"] >= threshold]
target_base = df[df["hot_mean"] < threshold]
target = target_base.drop(["fid", "vol_sum", "hot_mean", "area", "perimeter"], axis=1)

# Regression model to find hot_mean
y = training["vol_sum"]
y = y.astype('float64')

X = training.drop(["fid", "vol_sum", "hot_mean", "area", "perimeter"], axis=1)
X = X.astype('float64')

n_features = X.shape[1]

# vol_sum
model = Sequential()
model.add(Dense(32, activation='relu', kernel_initializer='normal', input_shape=(n_features,)))
model.add(Dense(16, activation='relu', kernel_initializer='normal'))
model.add(Dense(1))

# compile the model
model.compile(
    loss='mse',
    # optimizer='adam',
    metrics=['mae', 'mse'],
)

model.fit(X, y, epochs=500, verbose=1, batch_size=64)

y_pred = model.predict(target)
target_base["vol_sum_pred"] = y_pred

sample_base = df.sample(25, random_state=42)
sample = sample_base.drop(["fid", "vol_sum", "hot_mean", "area", "perimeter"], axis=1)

sample_pred = model.predict(sample)
sample_base["vol_sum_pred"] = sample_pred


import pdb; pdb.set_trace()