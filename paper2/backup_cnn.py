import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Flatten, BatchNormalization, Concatenate, Input, SpatialDropout2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.constraints import max_norm, unit_norm

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

folder = "C:\\Users\\caspe\\Desktop\\Paper_2_StruturalDensity\\analysis\\"

size = "160m"
seed = 42
val_split = 0.1
test_split = 0.33
batches = 128
kernel_size = (3, 3)

X = np.load(folder + f"{size}_images.npy")

# 0. area, 1. volume, 2. people
y = np.load(folder + f"{size}_labels.npy")[1] # volume
y = (y * (100 * 100)) / 700 # Average houses per hectare

y = np.max([
    (y < 0.5) * 0,
    np.logical_and((y >= 0.5), (y < 2)) * 1,
    (y >= 2) * 2,
], axis=0).astype('int64')

# Rotate and add all images
y = np.concatenate([
    y,
    y,
    y,
    y,
])

X = np.concatenate([
    X,
    np.rot90(X, k=1, axes=(1, 2)),
    np.rot90(X, k=2, axes=(1, 2)),
    np.rot90(X, k=3, axes=(1, 2)),
])

def count_freq(arr):
    yy = np.bincount(arr)
    ii = np.nonzero(yy)[0]
    return np.vstack((ii, yy[ii])).T

# Find minority class
frequency = count_freq(y)
minority = frequency.min(axis=0)[1]

print(frequency)

# https://stackoverflow.com/a/44233061/8564588
def minority_class_mask(arr, minority):
    return np.hstack([
        np.random.choice(np.where(y == l)[0], minority, replace=False) for l in np.unique(y)
    ])

def get_shape(numpy_arr):
    return (numpy_arr.shape[1], numpy_arr.shape[2], numpy_arr.shape[3])

mask = minority_class_mask(y, minority)

y = y[mask]
X = X[mask]

# B02, B03, B04, B08, BS, COH
# X = X[:, :, :, 4:5] # BS - 72.5%
# X = X[:, :, :, 4:6] # BS COH - 75.1%
# X = X[:, :, :, 0:4] # B02, B03, B04, B08 - 87.5%
# X = X[:, :, :, 2:6] # B04, B08, BS, COH - 87.3%

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split, random_state=seed)
split_amount = int(minority * val_split)

y_val = y_train[-split_amount:]
y_train = y_train[:-split_amount]

X_test_rgb = X_test[:, :, :, [0, 1, 2]]
X_test_nir = X_test[:, :, :, [3]]
X_test_coh = X_test[:, :, :, [4]]
X_test_bs = X_test[:, :, :, [5]]

X_val_rgb = X_train[-split_amount:, :, :, [0, 1, 2]]
X_val_nir = X_train[-split_amount:, :, :, [3]]
X_val_coh = X_train[-split_amount:, :, :, [4]]
X_val_bs = X_train[-split_amount:, :, :, [5]]

X_train_rgb = X_train[:-split_amount, :, :, [0, 1, 2]]
X_train_nir = X_train[:-split_amount, :, :, [3]]
X_train_coh = X_train[:-split_amount, :, :, [4]]
X_train_bs = X_train[:-split_amount, :, :, [5]]


classes = len(np.unique(y_train))

def create_cnn_model(shape, name):
    model_input = Input(shape=shape, name=name)
    model = Conv2D(64, kernel_size=(5, 5), padding='same', activation='swish', kernel_initializer='he_uniform', kernel_constraint=max_norm(3))(model_input)
    model = Conv2D(64, kernel_size=kernel_size, padding='same', activation='swish', kernel_initializer='he_uniform', kernel_constraint=max_norm(3))(model)
    model = MaxPooling2D(pool_size=(2, 2))(model)
    model = SpatialDropout2D(0.2)(model)
    model = BatchNormalization()(model)

    model = Conv2D(128, kernel_size=kernel_size, padding='same', activation='swish', kernel_initializer='he_uniform', kernel_constraint=max_norm(3))(model)
    model = Conv2D(128, kernel_size=kernel_size, padding='same', activation='swish', kernel_initializer='he_uniform', kernel_constraint=max_norm(3))(model)
    model = MaxPooling2D(pool_size=(2, 2))(model)
    model = SpatialDropout2D(0.2)(model)
    model = BatchNormalization()(model)

    model = Conv2D(256, kernel_size=kernel_size, padding='same', activation='swish', kernel_initializer='he_uniform', kernel_constraint=max_norm(3))(model)
    model = Conv2D(256, kernel_size=kernel_size, padding='same', activation='swish', kernel_initializer='he_uniform', kernel_constraint=max_norm(3))(model)
    model = GlobalAveragePooling2D()(model)
    model = BatchNormalization()(model)
    model = Dropout(0.25)(model)

    model = Flatten()(model)

    return (model, model_input)


model_rgb, input_rgb = create_cnn_model(get_shape(X_train_rgb), "input_rgb")
model_nir, input_nir = create_cnn_model(get_shape(X_train_nir), "input_nir")
model_coh, input_coh = create_cnn_model(get_shape(X_train_coh), "input_coh")
model_bs, input_bs = create_cnn_model(get_shape(X_train_bs), "input_bs")

model = Concatenate()([model_rgb, model_nir, model_coh, model_bs])
model = Dense(512, activation='swish', kernel_initializer='he_uniform', kernel_constraint=max_norm(3))(model)
model = BatchNormalization()(model)
model = Dropout(0.5)(model)
model = Dense(256, activation='swish', kernel_initializer='he_uniform', kernel_constraint=max_norm(3))(model)
model = BatchNormalization()(model)
model = Dropout(0.5)(model)

predictions = Dense(classes, activation='softmax')(model)

model = Model(inputs=[input_rgb, input_nir, input_coh, input_bs], outputs=predictions)

optimizer = Adam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    name='Adam',
)

model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(
    x=[X_train_rgb, X_train_nir, X_train_coh, X_train_bs],
    y=y_train,
    epochs=500,
    verbose=1,
    batch_size=batches,
    validation_data=([X_val_rgb, X_val_nir, X_val_coh, X_val_bs], y_val),
    callbacks=[
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
        )
    ]
)

loss, acc = model.evaluate([X_test_rgb, X_test_nir, X_test_coh, X_test_bs], y_test, verbose=1)
print('Test Accuracy: %.3f' % acc)

y_pred = model.predict([X_test_rgb, X_test_nir, X_test_coh, X_test_bs])
y_prop = np.max(y_pred, axis=1)
y_pred = np.argmax(y_pred, axis=1)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))