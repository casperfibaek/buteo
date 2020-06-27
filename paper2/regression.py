from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import Sequential, Model
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Flatten, BatchNormalization, Concatenate, Input, SeparableConv2D
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import ml_utils
import numpy as np

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

folder = "C:\\Users\\caspe\\Desktop\\Paper_2_StruturalDensity\\analysis\\"
size = 160
start_lr = 0.01
epochs = 100
batches = 64
seed = 42
validation_split = 0.3
rotation = False
noise = False
test = False
test_size = 50000
inserted_zeros = 1 # * minority class
target = "area"
msg = f"{str(size)} - rgbn + backscatter + coherence"

# ***********************************************************************
#                   LOADING DATA
# ***********************************************************************

blue = 0
green = 1
red = 2
infra = 0

# Load and scale RGB channels
rgb = np.load(folder + f"{str(int(size))}_rgb.npy").astype("float32")
rgb[:, :, :, blue] = ml_utils.scale_to_01(np.clip(rgb[:, :, :, blue], 0, 4000))
rgb[:, :, :, green] = ml_utils.scale_to_01(np.clip(rgb[:, :, :, green], 0, 5000))
rgb[:, :, :, red] = ml_utils.scale_to_01(np.clip(rgb[:, :, :, red], 0, 6000))

# Load and scale NIR channel (Add additional axis to match RGB)
nir = np.load(folder + f"{str(int(size))}_nir.npy").astype("float32")
nir = nir[:, :, :, np.newaxis]
nir[:, :, :, infra] = ml_utils.scale_to_01(np.clip(nir[:, :, :, infra], 0, 11000))

# # Load Backscatter (asc + desc), clip the largest outliers (> 92%)
bs_asc = np.load(folder + f"{str(int(size))}_bs.npy")[:, :, :, [ml_utils.sar_class("asc")]]
bs_asc = ml_utils.scale_to_01(np.clip(bs_asc, 0, np.quantile(bs_asc, 0.98)))
bs_desc = np.load(folder + f"{str(int(size))}_bs.npy")[:, :, :, [ml_utils.sar_class("desc")]]
bs_desc = ml_utils.scale_to_01(np.clip(bs_desc, 0, np.quantile(bs_desc, 0.98)))

# # Load coherence
# coh_asc = np.load(folder + f"{str(int(size))}_coh.npy")[:, :, :, [ml_utils.sar_class("asc")]]
# coh_asc = ml_utils.scale_to_01(coh_asc)
# coh_desc = np.load(folder + f"{str(int(size))}_coh.npy")[:, :, :, [ml_utils.sar_class("desc")]]
# coh_desc = ml_utils.scale_to_01(coh_desc)

X = np.concatenate([
    rgb,
    nir,
    bs_asc,
    bs_desc,
    # coh_asc,
    # coh_desc,
], axis=3)

if target == "area":
    labels = [*range(140, 5740, 140)]
else:
    labels = [*range(500, 20500, 500)]

y = np.load(folder + f"{str(int(size))}_y.npy")[:, ml_utils.y_class(target)]
y = (size * size) * y # Small house (100m2 * 4m avg. height)

# ***********************************************************************
#                   PREPARING DATA
# ***********************************************************************

# # Subsample the areas with zero houses (vast majority)
# zero_house = y == 0
# zero_house_y = y[zero_house]
# zero_house_X = X[zero_house]

# # Remove zero values from origial sample
# y = y[~zero_house]
# X = X[~zero_house]

# Balance dataset
histogram_mask = ml_utils.histogram_selection(y)
y = y[histogram_mask]
X = X[histogram_mask]

# # Shuffle zero_house sample before we can add back in
# shuffle = np.random.permutation(len(zero_house_y))
# zero_house_y = zero_house_y[shuffle]
# zero_house_X = zero_house_X[shuffle]

# # Add the subsampled houses back in
# house_samples = int(round((len(y) * 0.2) * inserted_zeros))
# y = np.concatenate([y, zero_house_y[0:house_samples]])
# X = np.concatenate([X, zero_house_X[0:house_samples]])

# Shuffle
shuffle = np.random.permutation(len(y))
y = y[shuffle]
X = X[shuffle]

if test is True:
    y = y[:test_size]
    X = X[:test_size]

zero_house = None
zero_house_y = None
zero_house_X = None
undersample_mask = None
shuffle = None

# Randomly rotate images
# X = ml_utils.add_randomness(X)

# ***********************************************************************
#                   ANALYSIS
# ***********************************************************************

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model_input = Input(shape=ml_utils.get_shape(X_train), name="input")
model = SeparableConv2D(256, kernel_size=(5, 5), padding="valid", activation="swish", kernel_initializer="he_normal")(model_input)
model = SeparableConv2D(256, kernel_size=(3, 3), padding="same", activation="swish", kernel_initializer="he_normal")(model)
model = MaxPooling2D(pool_size=(2, 2))(model)
model = BatchNormalization()(model)

# model = SeparableConv2D(512, kernel_size=(3, 3), padding="valid", activation="swish", kernel_initializer="he_normal", kernel_constraint=max_norm(3))(model)
# model = SeparableConv2D(512, kernel_size=(2, 2), padding="same", activation="swish", kernel_initializer="he_normal", kernel_constraint=max_norm(3))(model)
# model = MaxPooling2D(pool_size=(2, 2))(model)
# model = BatchNormalization()(model)

model = SeparableConv2D(512, kernel_size=(3, 3), padding="valid", activation="swish", kernel_initializer="he_normal")(model)
model = SeparableConv2D(512, kernel_size=(2, 2), padding="same", activation="swish", kernel_initializer="he_normal")(model)
model = GlobalAveragePooling2D()(model)
model = BatchNormalization()(model)

model = Flatten()(model)

model = Dense(512, activation="swish", kernel_initializer="he_normal")(model)
model = BatchNormalization()(model)

predictions = Dense(1, activation="relu", dtype="float32")(model)

model = Model(inputs=[model_input], outputs=predictions)

def median_error(y_actual, y_pred):
    return tfp.stats.percentile(tf.math.abs(y_actual - y_pred), 50.0)

def median_loss(y_actual, y_pred):
    return tf.math.square(tfp.stats.percentile(tf.math.abs(y_actual - y_pred), 50.0))

model.compile(
    optimizer=Adam(
        learning_rate=start_lr,
        name="Adam",
    ),
    # loss="mean_squared_error",
    loss="mean_absolute_error",
    metrics=[
        "mean_absolute_error",
        median_error,
    ])

model.fit(
    x=X_train,
    y=y_train,
    epochs=epochs,
    verbose=1,
    batch_size=batches,
    validation_split=validation_split,
    callbacks=[
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            min_delta=1.0,
            restore_best_weights=True,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            cooldown=1,
            min_delta=1.0,
            min_lr=0.00001,
        ),
    ]
)

loss, mean_absolute_error, median_absolute_error = model.evaluate(X_test, y_test, verbose=1)

print("Test accuracy:")
print(f"Mean Absolute Error (MAE): {str(round(mean_absolute_error, 5))}")
print(f"Median Absolute Error (MSE): {str(round(median_absolute_error, 5))}")

from playsound import playsound; playsound(folder + "alarm.wav", block=False)

import matplotlib.pyplot as plt

truth = y.astype("float32")

predicted = model.predict(X).squeeze().astype("float32")

truth_labels = np.digitize(truth, labels, right=True)
predicted_labels = np.digitize(predicted, labels, right=True)
labels_unique = np.unique(truth_labels)

residuals = (truth - predicted).astype("float32")
residuals = residuals / 140 if target == "area" else residuals / 700

fig1, ax = plt.subplots()
ax.set_title("violin area")

per_class = []
for cl in labels_unique: per_class.append(residuals[truth_labels == cl])

ax.violinplot(per_class, showextrema=False, showmedians=True)

plt.show()

import pdb; pdb.set_trace()