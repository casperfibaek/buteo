from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.metrics import BinaryAccuracy

import ml_utils
import numpy as np
import pandas as pd

folder = "C:\\Users\\caspe\\Desktop\\Paper_2_StruturalDensity\\analysis\\"

seed = 42 # Ensure replicability
grid = 320
batches = 32
validation_split = 0.3
learning_rate = 0.001
kfolds = 5

# ***********************************************************************
#                   LOADING DATA
# ***********************************************************************

df_y = pd.read_csv(folder + f"zonal\\{grid}_zonal_y.csv", index_col=0)

df_X = pd.concat([
    # pd.read_csv(folder + f"zonal\\{grid}_zonal_rgb.csv", index_col=0),
    # pd.read_csv(folder + f"zonal\\{grid}_zonal_nir.csv", index_col=0),
    # pd.read_csv(folder + f"zonal\\{grid}_zonal_bs.csv", index_col=0)[['bs_asc_mean', 'bs_asc_median', 'bs_asc_stdev', 'bs_asc_min', 'bs_asc_max']],
    pd.read_csv(folder + f"zonal\\{grid}_zonal_bs.csv", index_col=0)[['bs_desc_mean', 'bs_desc_median', 'bs_desc_stdev', 'bs_desc_min', 'bs_desc_max']],
    # pd.read_csv(folder + f"zonal\\{grid}_zonal_coh.csv", index_col=0)[['coh_asc_mean', 'coh_asc_median', 'coh_asc_stdev', 'coh_asc_min', 'coh_asc_max']],
    # pd.read_csv(folder + f"zonal\\{grid}_zonal_coh.csv", index_col=0)[['coh_desc_mean', 'coh_desc_median', 'coh_desc_stdev', 'coh_desc_min', 'coh_desc_max']],
], axis=1)


scaler = MinMaxScaler()

X = scaler.fit_transform(df_X.values.astype("float32"))
y = (((df_y["b_volume"] * (100 * 100)) / 400) > 1).to_numpy()

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

    optimizer = Adam(learning_rate=learning_rate, name='Adam')

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[BinaryAccuracy()])

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
            ReduceLROnPlateau(
                monitor='val_loss',
                patience=4,
                factor=0.1,
                min_lr=0.00001,
                cooldown=2,
                min_delta=0.01,
            ),
        ]
    )

    loss, acc = model.evaluate(X_test, y_test, verbose=1)
    print('Test Accuracy: %.3f' % acc)

    scores.append(acc)


mean = np.array(scores).mean()
std = np.array(scores).std()
print(mean, std)

import pdb; pdb.set_trace()
