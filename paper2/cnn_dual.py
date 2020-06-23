from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from keras.constraints import max_norm
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
size = 160
seed = 42
kfolds = 5
batches = 16
validation_split = 0.3
rotation = False
noise = False
noise_amount = 0.01
msg = f"{str(size)} - rgbn + backscatter + coherence"

def learning_rate_decay(epoch):
  if epoch < 4:
    return 1e-3
  elif epoch >= 3 and epoch < 8:
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

X1 = np.load(folder + f"{str(int(size))}_rgb.npy").astype('float32')
X1[:, :, :, blue] = ml_utils.scale_to_01(np.clip(X1[:, :, :, blue], 0, 4000))
X1[:, :, :, green] = ml_utils.scale_to_01(np.clip(X1[:, :, :, green], 0, 5000))
X1[:, :, :, red] = ml_utils.scale_to_01(np.clip(X1[:, :, :, red], 0, 6000))

X2 = np.load(folder + f"{str(int(size))}_nir.npy").astype('float32')
X2 = X2[:, :, :, np.newaxis]
X2[:, :, :, nir] = ml_utils.scale_to_01(np.clip(X2[:, :, :, nir], 0, 11000))

X2 = np.concatenate([X1, X2], axis=3)

X3_raw = np.load(folder + f"{str(int(size))}_bs.npy")[:, :, :, [ml_utils.sar_class("asc"), ml_utils.sar_class("desc")]]
X3 = ml_utils.scale_to_01(np.clip(X3_raw, 0, np.quantile(X3_raw, 0.99)))

X4 = np.load(folder + f"{str(int(size))}_coh.npy")[:, :, :, [ml_utils.sar_class("asc"), ml_utils.sar_class("desc")]]

X1 = None
X3_raw = None


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

# Undersample
mask = ml_utils.minority_class_mask(y, minority)
y = y[mask]
# X1 = X1[mask]
X2 = X2[mask]
X3 = X3[mask]
X4 = X4[mask]

# Shuffle
shuffle = np.random.permutation(len(y))
y = y[shuffle]
# X1 = X1[shuffle]
X2 = X2[shuffle]
X3 = X3[shuffle]
X4 = X4[shuffle]

mask = None
shuffle = None

# ***********************************************************************
#                   ANALYSIS
# ***********************************************************************

if size == 80:
    kernel_start = (3, 3)
    kernel_mid = (3, 3)
    kernel_size = (3, 3)
elif size == 160:
    kernel_start = (5, 5)
    kernel_mid = (5, 5)
    kernel_end = (3, 3)
else:
    kernel_start = (7, 7)
    kernel_mid = (5, 5)
    kernel_end = (3, 3)


def create_cnn_model(shape, name):
    model_input = Input(shape=shape, name=name)
    model = Conv2D(64, kernel_size=kernel_start, padding='same', activation='swish', kernel_initializer='he_uniform')(model_input)
    model = Conv2D(64, kernel_size=kernel_start, padding='same', activation='swish', kernel_initializer='he_uniform')(model)
    model = MaxPooling2D(pool_size=(2, 2))(model)
    model = BatchNormalization()(model)

    model = Conv2D(128, kernel_size=kernel_mid, padding='same', activation='swish', kernel_initializer='he_uniform')(model)
    model = Conv2D(128, kernel_size=kernel_mid, padding='same', activation='swish', kernel_initializer='he_uniform')(model)
    model = MaxPooling2D(pool_size=(2, 2))(model)
    model = BatchNormalization()(model)

    model = Conv2D(256, kernel_size=kernel_end, padding='same', activation='swish', kernel_initializer='he_uniform')(model)
    model = Conv2D(256, kernel_size=kernel_end, padding='same', activation='swish', kernel_initializer='he_uniform')(model)
    model = GlobalAveragePooling2D()(model)
    model = BatchNormalization()(model)

    model = Flatten()(model)

    model = Dense(512, activation='swish', kernel_initializer='he_uniform', kernel_constraint=max_norm(3))(model)
    model = BatchNormalization()(model)

    return (model, model_input)


skf = StratifiedKFold(n_splits=kfolds)

scores = []

for train_index, test_index in skf.split(np.zeros(len(y)), y):
    # X_train_1, X_test_1 = X1[train_index], X1[test_index]
    X_train_2, X_test_2 = X2[train_index], X2[test_index]
    X_train_3, X_test_3 = X3[train_index], X3[test_index]
    X_train_4, X_test_4 = X4[train_index], X4[test_index]

    y_train, y_test = y[train_index], y[test_index]

    # model_graph_1, input_graph_1 = create_cnn_model(ml_utils.get_shape(X_train_1), "input_1")
    model_graph_2, input_graph_2 = create_cnn_model(ml_utils.get_shape(X_train_2), "input_2")
    model_graph_3, input_graph_3 = create_cnn_model(ml_utils.get_shape(X_train_3), "input_3")
    model_graph_4, input_graph_4 = create_cnn_model(ml_utils.get_shape(X_train_4), "input_4")

    model = Concatenate()([
        # model_graph_1,
        model_graph_2,
        model_graph_3,
        model_graph_4,
    ])

    model = Dense(512, activation='swish', kernel_initializer='he_uniform')(model)
    model = BatchNormalization()(model)
    model = Dropout(0.5)(model)

    predictions = Dense(1, activation='sigmoid')(model)

    model = Model(inputs=[
        # input_graph_1,
        input_graph_2,
        input_graph_3,
        input_graph_4,
    ], outputs=predictions)

    model.compile(optimizer=Adam(name='Adam'), loss='binary_crossentropy', metrics=[BinaryAccuracy()])

    model.fit(
        x=[
            # X_train_1,
            X_train_2,
            X_train_3,
            X_train_4,
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
            LearningRateScheduler(learning_rate_decay),
        ]
    )

    loss, acc = model.evaluate([
        # X_test_1,
        X_test_2,
        X_test_3,
        X_test_4,
    ], y_test, verbose=1)
    print('Test Accuracy: %.3f' % acc)

    scores.append(acc)

mean = np.array(scores).mean()
std = np.array(scores).std()

print(mean, std)

from playsound import playsound; playsound(folder + "alarm.wav")
import pdb; pdb.set_trace()
