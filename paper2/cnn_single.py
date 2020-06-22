from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Flatten, BatchNormalization, Concatenate, Input
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import BinaryAccuracy, Accuracy

import os
import ml_utils
import numpy as np

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

folder = "C:\\Users\\caspe\\Desktop\\Paper_2_StruturalDensity\\analysis\\"
size = 320
seed = 42
validation_split = 0.3
kfolds = 5
batches = 16
learning_rate = 0.01
rotation = False
noise = False
noise_amount = 0.01

# ***********************************************************************
#                   LOADING DATA
# ***********************************************************************

blue = 0
green = 1
red = 2
nir = 0

# X = np.load(folder + f"{str(int(size))}_rgb.npy").astype('float32')
X = np.load(folder + f"{str(int(size))}_nir.npy").astype('float32')
X = X[:, :, :, np.newaxis]

# X[:, :, :, blue] = ml_utils.scale_to_01(np.clip(X[:, :, :, blue], 0, 4000))
# X[:, :, :, green] = ml_utils.scale_to_01(np.clip(X[:, :, :, green], 0, 5000))
# X[:, :, :, red] = ml_utils.scale_to_01(np.clip(X[:, :, :, red], 0, 6000))

X[:, :, :, nir] = ml_utils.scale_to_01(np.clip(X[:, :, :, nir], 0, 11000))

y = np.load(folder + f"{str(int(size))}_y.npy")[:, ml_utils.y_class("volume")]

y = (y * (100 * 100)) / 400 # Small house (100m2 * 4m avg. height)
y = (y >= 1.0).astype('int64')

# ***********************************************************************
#                   PREPARING DATA
# ***********************************************************************

# Rotate and add all images, add random noise to images to reduce overfit.
if rotation is True:
    X = ml_utils.add_rotations(X)
    y = np.concatenate([y, y, y, y])

if noise is True:
    X = ml_utils.add_noise(X, noise_amount)

# Find minority class
frequency = ml_utils.count_freq(y)
minority = frequency.min(axis=0)[1]

# Undersample data
mask = ml_utils.minority_class_mask(y, minority)
y = y[mask]
X = X[mask]
  
# Shuffle data
shuffle = np.random.permutation(len(y))
y = y[shuffle]
X = X[shuffle]

# ***********************************************************************
#                   ANALYSIS
# ***********************************************************************

def create_cnn_model(shape, name):
    model_input = Input(shape=shape, name=name)
    model = Conv2D(64, kernel_size=(5, 5), padding='same', activation='swish', kernel_initializer='he_uniform')(model_input)
    model = Conv2D(64, kernel_size=(3, 3), padding='same', activation='swish', kernel_initializer='he_uniform')(model)
    model = MaxPooling2D(pool_size=(2, 2))(model)
    model = BatchNormalization()(model)

    model = Conv2D(128, kernel_size=(3, 3), padding='same', activation='swish', kernel_initializer='he_uniform')(model)
    model = Conv2D(128, kernel_size=(3, 3), padding='same', activation='swish', kernel_initializer='he_uniform')(model)
    model = MaxPooling2D(pool_size=(2, 2))(model)
    model = BatchNormalization()(model)

    model = Conv2D(256, kernel_size=(3, 3), padding='same', activation='swish', kernel_initializer='he_uniform')(model)
    model = Conv2D(256, kernel_size=(3, 3), padding='same', activation='swish', kernel_initializer='he_uniform')(model)
    model = GlobalAveragePooling2D()(model)
    model = BatchNormalization()(model)

    model = Flatten()(model)

    return (model, model_input)


skf = StratifiedKFold(n_splits=kfolds)
scores = []

for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model_graph, input_graph = create_cnn_model(ml_utils.get_shape(X_train), "input")

    model = Dense(512, activation='swish', kernel_initializer='he_uniform')(model_graph)
    model = BatchNormalization()(model)
    model = Dropout(0.5)(model)

    predictions = Dense(1, activation='sigmoid')(model)

    model = Model(inputs=[input_graph], outputs=predictions)

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
            LearningRateScheduler(ml_utils.decay),
        ]
    )

    loss, acc = model.evaluate(X_test, y_test, verbose=1)
    print('Test Accuracy: %.3f' % acc)

    scores.append(acc)

mean = np.array(scores).mean()
std = np.array(scores).std()

print(mean, std)

from playsound import playsound; playsound(folder + "alarm.wav")
import pdb; pdb.set_trace()
