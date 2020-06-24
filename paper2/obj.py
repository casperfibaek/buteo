from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.metrics import BinaryAccuracy

import os
import ml_utils
import numpy as np
import pandas as pd

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

folder = "C:\\Users\\caspe\\Desktop\\Paper_2_StruturalDensity\\analysis\\"

seed = 42 # Ensure replicability
grid = 160
batches = 16
validation_split = 0.3
kfolds = 5
msg = f"{str(grid)} rgb nir coh"

# learning_rate = 0.001
def learning_rate_decay(epoch):
  if epoch < 5:
    return 1e-3
  elif epoch >= 5 and epoch < 10:
    return 1e-4
  else:
    return 1e-5


# ***********************************************************************
#                   LOADING DATA
# ***********************************************************************

bs_asc = ['bs_asc_mean', 'bs_asc_median', 'bs_asc_stdev', 'bs_asc_min', 'bs_asc_max']
bs_desc = ['bs_desc_mean', 'bs_desc_median', 'bs_desc_stdev', 'bs_desc_min', 'bs_desc_max']
coh_asc = ['coh_asc_mean', 'coh_asc_median', 'coh_asc_stdev', 'coh_asc_min', 'coh_asc_max']
coh_desc = ['coh_desc_mean', 'coh_desc_median', 'coh_desc_stdev', 'coh_desc_min', 'coh_desc_max']

df_y = pd.read_csv(folder + f"zonal\\{grid}_zonal_y.csv", index_col=0)

# Load backscatter
df_bs = pd.read_csv(folder + f"zonal\\{grid}_zonal_bs.csv", index_col=0)

df_X = pd.concat([
    df_bs.clip(df_bs.quantile(0.01), df_bs.quantile(0.99), axis=1)[bs_asc],
    df_bs.clip(df_bs.quantile(0.01), df_bs.quantile(0.99), axis=1)[bs_desc],
    pd.read_csv(folder + f"zonal\\{grid}_zonal_coh.csv", index_col=0)[coh_asc],
    pd.read_csv(folder + f"zonal\\{grid}_zonal_coh.csv", index_col=0)[coh_desc],
    pd.read_csv(folder + f"zonal\\{grid}_zonal_rgb.csv", index_col=0),
    pd.read_csv(folder + f"zonal\\{grid}_zonal_nir.csv", index_col=0),
], axis=1)

scaler = MinMaxScaler()

X = scaler.fit_transform(df_X.values).astype("float32")
y = (((df_y["b_volume"] * (100 * 100)) / 400) > 1).to_numpy(dtype="int64")

# ***********************************************************************
#                   PREPARING DATA
# ***********************************************************************

# Find minority class
frequency = ml_utils.count_freq(y)
minority = frequency.min(axis=0)[1]

# Undersample
mask = ml_utils.minority_class_mask(y, minority)
y = y[mask]
X = X[mask]

# Shuffle
shuffle = np.random.permutation(len(y))
y = y[shuffle]
X = X[shuffle]

mask = None
shuffle = None
df_y = None
df_X = None

# ***********************************************************************
#                   ANALYSIS
# ***********************************************************************

skf = StratifiedKFold(n_splits=kfolds)

scores = []
for train_index, test_index in skf.split(np.zeros(len(y)), y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model = Sequential([
        Dense(512, activation='swish', kernel_initializer='he_normal', input_shape=(X_train.shape[1],)),
        BatchNormalization(),
        Dense(256, activation='swish', kernel_initializer='he_normal'),
        BatchNormalization(),
        Dense(128, activation='swish', kernel_initializer='he_normal'),
        BatchNormalization(),
        Dense(1, activation='sigmoid'),
    ])

    model.compile(optimizer=Adam(name='Adam'), loss='binary_crossentropy', metrics=[BinaryAccuracy()])

    model.fit(
        x=X_train,
        y=y_train,
        epochs=500,
        verbose=1,
        batch_size=batches,
        validation_split=validation_split,
        callbacks=[
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                min_delta=0.01,
                restore_best_weights=True,
            ),
            LearningRateScheduler(learning_rate_decay),
        ]
    )

    loss, acc = model.evaluate(X_test, y_test, verbose=1)
    print('Test Accuracy: %.3f' % acc)

    scores.append(acc)


mean = np.array(scores).mean()
std = np.array(scores).std()
print(mean, std)
print(msg)

import pdb; pdb.set_trace()
