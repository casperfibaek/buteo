from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_probability as tfp
from tensorflow.keras import Sequential, Model, regularizers
from tensorflow.keras.constraints import max_norm, min_max_norm
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Flatten, BatchNormalization, Concatenate, Input, SeparableConv2D, AveragePooling2D, GlobalMaxPool2D
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import ml_utils
import numpy as np
import matplotlib.pyplot as plt

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

folder = "C:\\Users\\caspe\\Desktop\\Paper_2_StruturalDensity\\analysis\\"
size = 160
start_lr = 0.01
epochs = 100
batches = 96
seed = 42
validation_split = 0.3
rotation = False
noise = False
test = True
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

# σ0 (dB) = 10*log10 (abs (σ0))
# # Load Backscatter (asc + desc), clip the largest outliers (> 92%)
bs_asc = 10 * np.log10(np.abs(np.load(folder + f"{str(int(size))}_bs.npy")[:, :, :, [ml_utils.sar_class("asc")]]))
bs_asc = ml_utils.scale_to_01(np.clip(bs_asc, np.quantile(bs_asc, 0.02), np.quantile(bs_asc, 0.98)))

bs_desc = 10 * np.log10(np.abs(np.load(folder + f"{str(int(size))}_bs.npy")[:, :, :, [ml_utils.sar_class("desc")]]))
bs_desc = ml_utils.scale_to_01(np.clip(bs_desc, np.quantile(bs_desc, 0.02), np.quantile(bs_desc, 0.98)))

# # Load coherence
coh_asc = np.load(folder + f"{str(int(size))}_coh.npy")[:, :, :, [ml_utils.sar_class("asc")]]
coh_asc = ml_utils.scale_to_01(coh_asc)
coh_desc = np.load(folder + f"{str(int(size))}_coh.npy")[:, :, :, [ml_utils.sar_class("desc")]]
coh_desc = ml_utils.scale_to_01(coh_desc)

X = np.concatenate([
    # rgb,
    # nir,
    bs_asc,
    bs_desc,
    # coh_asc,
    # coh_desc,
], axis=3)

y = np.load(folder + f"{str(int(size))}_y.npy")[:, ml_utils.y_class(target)]
y = (size * size) * y # Small house (100m2 * 4m avg. height)

if target == "area":
    labels = [*range(0, int(round((y.max()))), 140)]
else:
    labels = [*range(0, int(round((y.max()))), 700)]

# ***********************************************************************
#                   PREPARING DATA
# ***********************************************************************

# Shuffle
shuffle = np.random.permutation(len(y))
y = y[shuffle]
X = X[shuffle]

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Balance training dataset
# histogram_mask = ml_utils.histogram_selection(y, resolution=9, zero_class=True, outliers=True)
histogram_mask = y > 0
house_count = np.sum(np.logical_and(y > 0, y <= 140))
yn = y[~histogram_mask][0:house_count]
Xn = X[~histogram_mask][0:house_count]

y = np.concatenate([y[histogram_mask], yn])
X = np.concatenate([X[histogram_mask], Xn])

# Shuffle
shuffle = np.random.permutation(len(y))
y = y[shuffle]
X = X[shuffle]

if test is True:
    y = y[:test_size]
    X = X[:test_size]

# ***********************************************************************
#                   ANALYSIS
# ***********************************************************************

model_input = Input(shape=ml_utils.get_shape(X), name="input")
model = Conv2D(256, kernel_size=(5, 5), padding="valid", activation=tfa.activations.mish, kernel_initializer="he_normal")(model_input)
model = Conv2D(256, kernel_size=(3, 3), padding="same", activation=tfa.activations.mish, kernel_initializer="he_normal")(model)
model = MaxPooling2D(pool_size=(2,2))(model)
model = BatchNormalization()(model)

model = Conv2D(512, kernel_size=(3, 3), padding="valid", activation=tfa.activations.mish, kernel_initializer="he_normal")(model_input)
model = Conv2D(512, kernel_size=(3, 3), padding="same", activation=tfa.activations.mish, kernel_initializer="he_normal")(model)
model = MaxPooling2D(pool_size=(2,2))(model)
model = BatchNormalization()(model)

model = Conv2D(256, kernel_size=(3, 3), padding="valid", activation=tfa.activations.mish, kernel_initializer="he_normal")(model)
model = Conv2D(256, kernel_size=(2, 2), padding="same", activation=tfa.activations.mish, kernel_initializer="he_normal")(model)
model = GlobalAveragePooling2D()(model)
model = BatchNormalization()(model)
model = Flatten()(model)

model = Dense(2048, activation=tfa.activations.mish, kernel_initializer="he_normal")(model)
model = BatchNormalization()(model)
model = Dropout(0.25)(model)

predictions = Dense(1, activation="relu", dtype="float32")(model)

model = Model(inputs=[model_input], outputs=predictions)

def median_error(y_actual, y_pred):
    return tfp.stats.percentile(tf.math.abs(y_actual - y_pred), 50.0)

optimizer = tfa.optimizers.Lookahead(
    Adam(
        learning_rate=tfa.optimizers.TriangularCyclicalLearningRate(
            initial_learning_rate=1e-4,
            maximal_learning_rate=1e-2,
            step_size=6,
            scale_mode='cycle',
            name='TriangularCyclicalLearningRate',
        ),
        name="Adam",
    )
)

model.compile(
    optimizer=optimizer,
    # loss="mean_absolute_error",
    # loss="mean_squared_error",
    # loss=tfa.losses.PinballLoss(),
    loss=tf.keras.losses.Huber(),
    metrics=[
        "mean_absolute_error",
        median_error,
    ])

model.fit(
    x=X,
    y=y,
    epochs=epochs,
    verbose=1,
    batch_size=batches,
    validation_split=validation_split,
    callbacks=[
        EarlyStopping(
            monitor="val_loss",
            patience=15,
            min_delta=1.0,
            restore_best_weights=True,
        ),
    ]
)

from playsound import playsound; playsound(folder + "alarm.wav", block=False)


loss, mean_absolute_error, median_absolute_error = model.evaluate(X, y, verbose=1)

print("Test accuracy:")
print(f"Mean Absolute Error (MAE): {str(round(mean_absolute_error, 5))}")
print(f"Median Absolute Error (MAE): {str(round(median_absolute_error, 5))}")

truth = y.astype("float32")
predicted = model.predict(X).squeeze().astype("float32")

truth_labels = np.digitize(truth, labels, right=True)
predicted_labels = np.digitize(predicted, labels, right=True)
labels_unique = np.unique(truth_labels)

residuals = ((truth - predicted) / 140).astype("float32")

fig1, ax = plt.subplots()
ax.set_title("violin area")

per_class = []
for cl in labels_unique: per_class.append(residuals[truth_labels == cl])

ax.violinplot(per_class, showextrema=False, showmedians=True, vert=False, widths=1)

plt.show()

import pdb; pdb.set_trace()