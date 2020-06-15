import sys; sys.path.append('..'); sys.path.append('../lib/')
from lib.raster_io import raster_to_array, array_to_raster
from lib.stats_filters import truncate_filter
import sqlite3
import pandas as pd
import numpy as np
import tensorflow
import keras
import matplotlib.pyplot as plt
import datetime
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_absolute_error, median_absolute_error
from imblearn.over_sampling import SMOTE
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from sklearn import preprocessing
from sklearn.feature_selection import RFECV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from lib.ml_utils import best_param_random

base = "C:\\Users\\caspe\\Desktop\\Paper_2_StruturalDensity\\analysis\\"

# db_engine = create_engine(f'sqlite:///{base}silkeborg_regression.sqlite') 
db_path = base + "silkeborg_zonal.sqlite"

cnx = sqlite3.connect(db_path)
seed = 42

df = pd.read_sql_query(f"SELECT * FROM 'silkeborg_zonal';", cnx)
df = df.dropna()

df_above = df[df["area_sum"] > 0]
df_zero = df[df["area_sum"] == 0].sample(3000)
df = pd.concat([df_above, df_zero])

cols = [
    # 'bs_mean', 'bs_median', 'bs_stdev', 'bs_min', 'bs_max',
    'bst_mean', 'bst_median', 'bst_stdev', 'bst_min', 'bst_max',
    'bsc_mean', 'bsc_median', 'bsc_stdev', 'bsc_min', 'bsc_max',
    # 'coh_mean', 'coh_median', 'coh_stdev', 'coh_min', 'coh_max',
    'b2_mean', 'b2_median', 'b2_stdev', 'b2_min', 'b2_max',
    'b3_mean', 'b3_median', 'b3_stdev', 'b3_min', 'b3_max',
    'b4_mean', 'b4_median', 'b4_stdev', 'b4_min', 'b4_max',
    'b8_mean', 'b8_median', 'b8_stdev', 'b8_min', 'b8_max',
    'msavi_mean', 'msavi_median', 'msavi_stdev', 'msavi_min', 'msavi_max',
    'bs_asc_mean', 'bs_asc_median', 'bs_asc_stdev', 'bs_asc_min', 'bs_asc_max',
    'bs_desc_mean', 'bs_desc_median', 'bs_desc_stdev', 'bs_desc_min', 'bs_desc_max',
]

# import pdb; pdb.set_trace()

scaler_X = preprocessing.MinMaxScaler()
scaler_y = preprocessing.MinMaxScaler()

X = df.drop(["ogc_fid", "fid", "area_sum", "perimeter_sum", "volume_sum"], axis=1)
X = X.drop(cols, axis=1)

X = scaler_X.fit_transform(X)
X = X.astype('float32')

y = pd.DataFrame(df["area_sum"])
y = scaler_y.fit_transform(y)
y = y.astype('float32')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=seed)

# determine the number of input features
n_features = X.shape[1]

# define model
model = Sequential([
    Dense(64, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)),
    Dense(32, activation='relu', kernel_initializer='he_normal'),
    Dense(16, activation='relu', kernel_initializer='he_normal'),
    Dense(8, activation='relu', kernel_initializer='he_normal'),
    Dense(1),
])

# model.add(Dropout(0.25))

# compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

model.fit(X_train, y_train, epochs=300, verbose=1, batch_size=64)

pred = model.predict(X_test)
pred = scaler_y.inverse_transform(model.predict(X_test))
test = scaler_y.inverse_transform(y_test)
res = test - pred

med_ae = median_absolute_error(pred, test)
mean_ae = mean_absolute_error(pred, test)

merged = np.concatenate((test, pred, res), axis=1)
merged_df = pd.DataFrame(merged, columns=["test", "pred", "res"])

print(med_ae, mean_ae)
merged_df.to_csv(base + "merged_area.csv")

import pdb; pdb.set_trace()
