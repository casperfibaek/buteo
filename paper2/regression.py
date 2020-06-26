from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras import Sequential, Model
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Flatten, BatchNormalization, Concatenate, Input
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.models import load_model

import os
import ml_utils
import numpy as np

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

folder = "C:\\Users\\caspe\\Desktop\\Paper_2_StruturalDensity\\analysis\\"
size = 160
seed = 42
kfolds = 5
batches = 96
validation_split = 0.3
rotation = False
noise = False
test = False
fold_split = True
test_size = 10000
noise_amount = 0.01
learning_rate = 0.01
msg = f"{str(size)} - rgbn + backscatter + coherence"

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

blue = 0
green = 1
red = 2
nir = 0

# Load and scale RGB channels
X_rgb = np.load(folder + f"{str(int(size))}_rgb.npy").astype('float32')
X_rgb[:, :, :, blue] = ml_utils.scale_to_01(np.clip(X_rgb[:, :, :, blue], 0, 4000))
X_rgb[:, :, :, green] = ml_utils.scale_to_01(np.clip(X_rgb[:, :, :, green], 0, 5000))
X_rgb[:, :, :, red] = ml_utils.scale_to_01(np.clip(X_rgb[:, :, :, red], 0, 6000))

# Load and scale NIR channel (Add additional axis to match RGB)
X_nir = np.load(folder + f"{str(int(size))}_nir.npy").astype('float32')
X_nir = X_nir[:, :, :, np.newaxis]
X_nir[:, :, :, nir] = ml_utils.scale_to_01(np.clip(X_nir[:, :, :, nir], 0, 11000))

# Merge RGB and NIR
X = np.concatenate([X_rgb, X_nir], axis=3)

# Load Backscatter (asc + desc), clip the largest outliers (> 99%)
bs = np.load(folder + f"{str(int(size))}_bs.npy")[:, :, :, [ml_utils.sar_class("asc"), ml_utils.sar_class("desc")]]
bs = ml_utils.scale_to_01(np.clip(bs, 0, np.quantile(bs, 0.99)))
bs = np.concatenate([
    bs.mean(axis=(1,2)),
    bs.std(axis=(1,2)),
    bs.min(axis=(1,2)),
    bs.max(axis=(1,2)),
    np.median(bs, axis=(1,2)),
], axis=1)

# Load coherence
coh = np.load(folder + f"{str(int(size))}_coh.npy")[:, :, :, [ml_utils.sar_class("asc"), ml_utils.sar_class("desc")]]
coh = np.concatenate([
    coh.mean(axis=(1,2)),
    coh.std(axis=(1,2)),
    coh.min(axis=(1,2)),
    coh.max(axis=(1,2)),
    np.median(coh, axis=(1,2)),
], axis=1)

sar = np.concatenate([bs, coh], axis=1)

labels = [*range(140, 5740, 140)]
y = np.load(folder + f"{str(int(size))}_y.npy")[:, ml_utils.y_class("area")]
y = (size * size) * y # Small house (100m2 * 4m avg. height)

og_y = y
og_X = X
og_sar = sar

X_rgb = None
X_nir = None
bs = None
coh = None

# ***********************************************************************
#                   PREPARING DATA
# ***********************************************************************

# Subsample the areas with zero houses (vast majority)
zero_house = y == 0
zero_house_y = y[zero_house]
zero_house_X = X[zero_house]
zero_house_sar = sar[zero_house]

shuffle = np.random.permutation(len(zero_house_y))
zero_house_y = y[shuffle]
zero_house_X = X[shuffle]
zero_house_sar = sar[shuffle]

# Remove the top 5 percentile
outlier_mask = np.logical_and(y > 0, y < np.quantile(y[y > 0], 0.95))

y = y[outlier_mask]
X = X[outlier_mask]
sar = sar[outlier_mask]

# Find minority class
class_mask = np.digitize(y, np.percentile(y, [10, 20, 30, 40, 50, 60, 70, 80, 90]))
frequency = ml_utils.count_freq(class_mask)
minority = frequency.min(axis=0)[1]

# Undersample
undersample_mask = ml_utils.minority_class_mask(class_mask, minority)
y = y[undersample_mask]
X = X[undersample_mask]
sar = sar[undersample_mask]
class_mask = class_mask[undersample_mask]

# Add the subsampled houses back in
y = np.concatenate([y, zero_house_y[0:minority * 2]])
X = np.concatenate([X, zero_house_X[0:minority * 2]])
sar = np.concatenate([sar, zero_house_sar[0:minority * 2]])
class_mask = np.concatenate([
    class_mask,
    np.full(zero_house_y[0:minority * 2].shape, 99, dtype="int64"), # class 99 for the added "zero house" tiles
])

if test is True:
    y = y[:test_size]
    X = X[:test_size]
    sar = sar[:test_size]
    class_mask = class_mask[:test_size]

# Rotate and add all images, add random noise to images to reduce overfit.
if rotation is True:
    y = np.concatenate([y, y, y, y])
    X = ml_utils.add_rotations(X)
    sar = ml_utils.add_rotations(sar)
    class_mask = np.concatenate([class_mask, class_mask, class_mask, class_mask])

if noise is True:
    X = ml_utils.add_noise(X, noise_amount)

# Shuffle
shuffle = np.random.permutation(len(y))
y = y[shuffle]
X = X[shuffle]
sar = sar[shuffle]
class_mask = class_mask[shuffle]

undersample_mask = None
shuffle = None

# ***********************************************************************
#                   ANALYSIS
# ***********************************************************************

if size == 80:
    kernel_start = (3, 3)
    kernel_mid = (2, 2)
    kernel_end = (2, 2)
elif size == 160:
    kernel_start = (5, 5)
    kernel_mid = (5, 5)
    kernel_end = (3, 3)
else:
    kernel_start = (7, 7)
    kernel_mid = (5, 5)
    kernel_end = (3, 3)


def create_mlp_model(shape, name):
    model_input = Input(shape=shape, name=name)
    model = Dense(512, activation='swish', kernel_initializer='he_normal', kernel_constraint=max_norm(3))(model_input)
    model = BatchNormalization()(model)
    model = Dense(256, activation='swish', kernel_initializer='he_normal', kernel_constraint=max_norm(3))(model)
    model = BatchNormalization()(model)
    model = Dense(128, activation='swish', kernel_initializer='he_normal', kernel_constraint=max_norm(3))(model)
    model = BatchNormalization()(model)

    return (model, model_input)


def create_cnn_model(shape, name):
    model_input = Input(shape=shape, name=name)
    model = Conv2D(64, kernel_size=kernel_start, padding='same', activation='swish', kernel_initializer='he_uniform', kernel_constraint=max_norm(3))(model_input)
    model = Conv2D(64, kernel_size=kernel_start, padding='same', activation='swish', kernel_initializer='he_uniform', kernel_constraint=max_norm(3))(model)
    model = MaxPooling2D(pool_size=(2, 2))(model)
    model = BatchNormalization()(model)

    model = Conv2D(128, kernel_size=kernel_mid, padding='same', activation='swish', kernel_initializer='he_uniform', kernel_constraint=max_norm(3))(model)
    model = Conv2D(128, kernel_size=kernel_mid, padding='same', activation='swish', kernel_initializer='he_uniform', kernel_constraint=max_norm(3))(model)
    model = MaxPooling2D(pool_size=(2, 2))(model)
    model = BatchNormalization()(model)

    model = Conv2D(256, kernel_size=kernel_end, padding='same', activation='swish', kernel_initializer='he_uniform', kernel_constraint=max_norm(3))(model)
    model = Conv2D(256, kernel_size=kernel_end, padding='same', activation='swish', kernel_initializer='he_uniform', kernel_constraint=max_norm(3))(model)
    model = GlobalAveragePooling2D()(model)
    model = BatchNormalization()(model)

    model = Flatten()(model)

    model = Dense(512, activation='swish', kernel_initializer='he_uniform', kernel_constraint=max_norm(3))(model)
    model = BatchNormalization()(model)

    return (model, model_input)

# Are we going to ensure balanced regression sets?
kf = KFold(n_splits=kfolds)
skf = StratifiedKFold(n_splits=kfolds)

if fold_split is True:
    splits = skf.split(np.zeros(len(class_mask)), class_mask)
else:
    splits = kf.split(np.zeros(len(y)), y)

split = 0
scores_mae = []
scores_mse = []

for train_index, test_index in splits:
    X_train_1, X_test_1 = X[train_index], X[test_index]
    X_train_2, X_test_2 = sar[train_index], sar[test_index]

    y_train, y_test = y[train_index], y[test_index]

    model_graph_1, input_graph_1 = create_cnn_model(ml_utils.get_shape(X_train_1), "sentinel_2")
    model_graph_2, input_graph_2 = create_mlp_model((X_train_2.shape[1],), "sentinel_1")

    model = Concatenate()([
        model_graph_1,
        model_graph_2,
    ])

    model = Dense(512, activation='swish', kernel_initializer='he_uniform')(model)
    model = BatchNormalization()(model)
    model = Dropout(0.25)(model)

    predictions = Dense(1)(model)

    model = Model(inputs=[
        input_graph_1,
        input_graph_2,
    ], outputs=predictions)

    model.compile(optimizer=Adam(learning_rate=learning_rate, name='Adam'), loss='mean_absolute_error', metrics=['mean_absolute_error', 'mean_squared_error'])

    model.fit(
        x=[
            X_train_1,
            X_train_2,
        ],
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
                factor=0.1,
                patience=5,
                min_lr=0.00001,
            ),
            ModelCheckpoint(
                f'./models/regression_{str(split)}.hdf5',
                save_best_only=True,
                monitor='val_loss',
                mode='min',
            )
        ]
    )

    loss, mean_absolute_error, mean_squared_error = model.evaluate([X_test_1, X_test_2], y_test, verbose=1)

    print("Test accuracy:")
    print(f"Mean Absolute Error (MAE): {str(round(mean_absolute_error, 5))}")
    print(f"Mean Squared Error (MSE): {str(round(mean_squared_error, 5))}")

    scores_mae.append(mean_absolute_error)
    scores_mse.append(mean_squared_error)

    split += 1

mean_mae = np.array(scores_mae).mean()
std_mae = np.array(scores_mae).std()
mean_mse = np.array(scores_mse).mean()
std_mse = np.array(scores_mse).std()

print("Final regression:")
print(f"Mean Absolute Error: {str(round(mean_mae, 5))}, stdev: {str(round(std_mae, 5))}")
print(f"Mean Squared Error: {str(round(mean_mse, 5))}, stdev: {str(round(std_mse, 5))}")

best_model = load_model(f"./models/regression_{str(np.argmin(np.array(scores_mae)))}.hdf5")

y_labels = np.digitize(og_y, labels)
y_pred = np.digitize(best_model.predict([og_X, og_sar]), labels)

print(confusion_matrix(y_labels, y_pred))
print(classification_report(y_labels, y_pred))
print(accuracy_score(y_labels, y_pred))

np.savetxt("./models/confusion_matrix_160_area_140_interval.csv", confusion_matrix(y_labels, y_pred), delimiter=",") 

from playsound import playsound; playsound(folder + "alarm.wav")