from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import Sequential, Model
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Flatten, BatchNormalization, Concatenate, Input
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.mixed_precision import experimental as mixed_precision
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
batches = 32
validation_split = 0.3
rotation = False
noise = False
test = False
fold_split = True
test_size = 50000
noise_amount = 0.01
learning_rate = 0.01
inserted_zeros = 1 # * minority class
msg = f"{str(size)} - rgbn + backscatter + coherence"

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

# Remove zero values from origial sample
y = y[~zero_house]
X = X[~zero_house]
sar = sar[~zero_house]

# Remove the top 5 percentile
outlier_mask = (y < np.quantile(y, 0.95))

y = y[outlier_mask]
X = X[outlier_mask]
sar = sar[outlier_mask]


# Shuffle before we can add back in
shuffle = np.random.permutation(len(zero_house_y))
zero_house_y = zero_house_y[shuffle]
zero_house_X = zero_house_X[shuffle]
zero_house_sar = zero_house_sar[shuffle]

# Find minority class
class_mask = np.digitize(y, np.percentile(y, [10, 20, 30, 40, 50, 60, 70, 80, 90]))
frequency = ml_utils.count_freq(class_mask) # Not necessary here (makes no difference)
minority = frequency.min(axis=0)[1]

# Undersample
undersample_mask = ml_utils.minority_class_mask(class_mask, minority)
y = y[undersample_mask]
X = X[undersample_mask]
sar = sar[undersample_mask]
class_mask = class_mask[undersample_mask]

# Add the subsampled houses back in
y = np.concatenate([y, zero_house_y[0:int(minority * inserted_zeros)]])
X = np.concatenate([X, zero_house_X[0:int(minority * inserted_zeros)]])
sar = np.concatenate([sar, zero_house_sar[0:int(minority * inserted_zeros)]])
class_mask = np.concatenate([
    class_mask,
    np.full(zero_house_y[0:int(minority * inserted_zeros)].shape, 99, dtype="int64"), # class 99 for the added "zero house" tiles
])

# Shuffle
shuffle = np.random.permutation(len(y))
y = y[shuffle]
X = X[shuffle]
sar = sar[shuffle]
class_mask = class_mask[shuffle]

# Rotate and add all images, add random noise to images to reduce overfit.
if rotation is True:
    y = np.concatenate([y, y])
    X = ml_utils.add_rotations(X, 2)
    sar = np.concatenate([sar, sar]) # Change if going for seperate convolutions instead of summary statistics..
    class_mask = np.concatenate([class_mask, class_mask])

if noise is True:
    X = ml_utils.add_noise(X, noise_amount)

if test is True:
    y = y[:test_size]
    X = X[:test_size]
    sar = sar[:test_size]
    class_mask = class_mask[:test_size]

zero_house = None
zero_house_y = None
zero_house_X = None
zero_house_sar = None
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
    kernel_mid = (3, 3)
    kernel_end = (2, 2)
else:
    kernel_start = (5, 5)
    kernel_mid = (3, 3)
    kernel_end = (3, 3)


def create_mlp_model(shape, name, size):
    model_input = Input(shape=shape, name=name)
    model = Dense(512, activation='swish', kernel_initializer='he_normal')(model_input)
    model = BatchNormalization()(model)
    model = Dense(256, activation='swish', kernel_initializer='he_normal')(model)
    model = BatchNormalization()(model)
    model = Dense(128, activation='swish', kernel_initializer='he_normal')(model)
    model = BatchNormalization()(model)

    return (model, model_input)


def create_cnn_model(shape, name, size):
    model_input = Input(shape=shape, name=name)
    model = Conv2D(64, kernel_size=kernel_start, padding='valid', activation='swish', kernel_initializer='he_normal')(model_input)
    model = Conv2D(64, kernel_size=kernel_mid, padding='same', activation='swish', kernel_initializer='he_normal')(model)
    model = MaxPooling2D(pool_size=(2, 2))(model)
    model = BatchNormalization()(model)

    model = Conv2D(128, kernel_size=kernel_mid, padding='valid', activation='swish', kernel_initializer='he_normal')(model)
    model = Conv2D(128, kernel_size=kernel_mid, padding='same', activation='swish', kernel_initializer='he_normal')(model)

    # if size > 160:
    model = MaxPooling2D(pool_size=(2, 2))(model)
    model = BatchNormalization()(model)

    # model = Conv2D(128, kernel_size=kernel_end, padding='valid', activation='swish', kernel_initializer='he_normal')(model)
    # model = Conv2D(128, kernel_size=kernel_end, padding='same', activation='swish', kernel_initializer='he_normal')(model)

    # model = GlobalAveragePooling2D()(model)
    # model = BatchNormalization()(model)

    model = Flatten()(model)

    model = Dense(256, activation='swish', kernel_initializer='he_normal')(model)
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
scores_meae = []

for train_index, test_index in splits:
    X_train_1, X_test_1 = X[train_index], X[test_index]
    X_train_2, X_test_2 = sar[train_index], sar[test_index]

    y_train, y_test = y[train_index], y[test_index]

    model_graph_1, input_graph_1 = create_cnn_model(ml_utils.get_shape(X_train_1), "sentinel_2", size)
    model_graph_2, input_graph_2 = create_mlp_model((X_train_2.shape[1],), "sentinel_1", size)

    model = Concatenate()([
        model_graph_1,
        model_graph_2,
    ])

    model = Dense(512, activation='swish', kernel_initializer='he_normal')(model)
    model = BatchNormalization()(model)

    predictions = Dense(1, activation='relu', dtype='float32')(model)

    model = Model(inputs=[
        input_graph_1,
        input_graph_2,
    ], outputs=predictions)
   
    def median_error(y_actual, y_pred):
        return tfp.stats.percentile(tf.math.abs(y_actual - y_pred), 50.0)

    model.compile(
        optimizer=Adam(
            learning_rate=learning_rate,
            name='Adam',
        ),
        loss='mean_squared_error',
        metrics=[
            'mean_absolute_error',
            median_error
        ])

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
                min_delta=1.0,
                restore_best_weights=True,
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                cooldown=1,
                min_delta=1.0,
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

    loss, mean_absolute_error, median_absolute_error = model.evaluate([X_test_1, X_test_2], y_test, verbose=1)

    print("Test accuracy:")
    print(f"Mean Absolute Error (MAE): {str(round(mean_absolute_error, 5))}")
    print(f"Median Absolute Error (MSE): {str(round(median_absolute_error, 5))}")

    ml_utils.viz([X_test_1, X_test_2], y_test, model, target="area")
    
    scores_mae.append(mean_absolute_error)
    scores_meae.append(median_absolute_error)

    split += 1

mean_mae = np.array(scores_mae).mean()
std_mae = np.array(scores_mae).std()
mean_meae = np.array(scores_meae).mean()
std_meae = np.array(scores_meae).std()

print("Final regression:")
print(f"Mean Absolute Error: {str(round(mean_mae, 5))}, stdev: {str(round(std_mae, 5))}")
print(f"Median Absolute Error: {str(round(mean_meae, 5))}, stdev: {str(round(std_meae, 5))}")

model_mean = f"./models/regression_{str(np.argmin(np.array(scores_mae)))}.hdf5"
model_median = f"./models/regression_{str(np.argmin(np.array(scores_meae)))}.hdf5"

print(f"Best model is (mae): {model_mean}")
print(f"Best model is (meae): {model_median}")

from playsound import playsound; playsound(folder + "alarm.wav")

import matplotlib.pyplot as plt

best_model = load_model(model_median)

truth = y.astype("float32")
predicted = best_model.predict([X, sar]).squeeze().astype("float32")

truth_labels = np.digitize(y, labels, right=True)
predicted_labels = np.digitize(predicted, labels, right=True)
labels_unique = np.unique(truth_labels)

residuals = (truth - predicted).astype('float32')
median_absolute = np.median(np.abs(residuals))

fig1, ax = plt.subplots()
ax.set_title('Boxplot area')

per_class = []
for cl in labels_unique:
    per_class.append(residuals[truth_labels == cl])

ax.boxplot(per_class, showfliers=False)
ax.violinplot(per_class, showextrema=False, showmedians=True)

plt.show()

import pdb; pdb.set_trace()