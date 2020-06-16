from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.constraints import max_norm

import sqlite3
import pandas as pd

folder = "C:\\Users\\caspe\\Desktop\\Paper_2_StruturalDensity\\analysis\\"

# path_grid_320m = folder + "grid_320m.sqlite"
# cnx_grid_320m = sqlite3.connect(path_grid_320m)
# df_320m = pd.read_sql_query(f"SELECT * FROM 'grid_320m';", cnx_grid_320m)

# path_grid_160m = folder + "grid_160m.sqlite"
# cnx_grid_160m = sqlite3.connect(path_grid_160m)
# df_160m = pd.read_sql_query(f"SELECT * FROM 'grid_160m';", cnx_grid_160m)

path_grid_80m = folder + "grid_80m.sqlite"
cnx_grid_80m = sqlite3.connect(path_grid_80m)
df_80m = pd.read_sql_query(f"SELECT * FROM 'grid_80m';", cnx_grid_80m)

cols = [
    # 'ogc_fid', 'fid',
    # 'b_area', 'b_volume', 'ppl_ha',
    'b02_mean', 'b02_median', 'b02_stdev', 'b02_min', 'b02_max',
    'b03_mean', 'b03_median', 'b03_stdev', 'b03_min', 'b03_max',
    'b04_mean', 'b04_median', 'b04_stdev', 'b04_min', 'b04_max',
    'b08_mean', 'b08_median', 'b08_stdev', 'b08_min', 'b08_max',
    'bs_mean', 'bs_median', 'bs_stdev', 'bs_min', 'bs_max',
    # 'bs_asc_mean', 'bs_asc_median', 'bs_asc_stdev', 'bs_asc_min', 'bs_asc_max',
    # 'bs_desc_mean', 'bs_desc_median', 'bs_desc_stdev', 'bs_desc_min', 'bs_desc_max',
    # 'bs_coh_mean', 'bs_coh_median', 'bs_coh_stdev', 'bs_coh_min', 'bs_coh_max',
    'coh_mean', 'coh_median', 'coh_stdev', 'coh_min', 'coh_max',
    # 'coh_asc_mean', 'coh_asc_median', 'coh_asc_stdev', 'coh_asc_min', 'coh_asc_max',
    # 'coh_desc_mean', 'coh_desc_median', 'coh_desc_stdev', 'coh_desc_min', 'coh_desc_max',
    # 'msavi2_mean', 'msavi2_median', 'msavi2_stdev', 'msavi2_min', 'msavi2_max'
]

df = df_80m

seed = 42 # Ensure replicability

# df["bin_test"] = df["b_volume"] > 0
df["bin_test"] = df["b_volume"] * (100 * 100) >= (700 * 3)

# Ensure that the binary sample sizes are equal
above = df[df["bin_test"] == True]
below = df[df["bin_test"] == False]
sample_size = min([len(above), len(below)])

df = pd.concat([above.sample(sample_size, random_state=seed), below.sample(sample_size, random_state=seed)])

scaler = MinMaxScaler()

y = (df["bin_test"]).to_numpy("float32")
X = scaler.fit_transform(df[cols].values.astype("float32"))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)

X_val = X_train[-10000:]
y_val = y_train[-10000:]
X_train = X_train[:-10000]
y_train = y_train[:-10000]

n_features = X_train.shape[1]

model = Sequential([
    Dense(512, activation='swish', kernel_initializer='he_normal', input_shape=(n_features,)),
    Dropout(0.25),
    Dense(256, activation='swish', kernel_initializer='he_normal', kernel_constraint=max_norm(3)),
    Dropout(0.25),
    Dense(128, activation='swish', kernel_initializer='he_normal', kernel_constraint=max_norm(3)),
    Dropout(0.25),
    Dense(1, activation='sigmoid'),
])

optimizer = Adam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    name='Adam',
)

model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=500, verbose=1, batch_size=64, validation_data=(X_val, y_val), callbacks=[
    EarlyStopping(monitor='loss', patience=5)
])

loss, acc = model.evaluate(X_test, y_test, verbose=1)
print('Test Accuracy: %.3f' % acc)

# import pdb; pdb.set_trace()