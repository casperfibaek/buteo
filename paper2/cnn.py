import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPool2D, AveragePooling2D, GlobalMaxPooling2D, Flatten, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.constraints import max_norm, unit_norm

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

folder = "C:\\Users\\caspe\\Desktop\\Paper_2_StruturalDensity\\analysis\\"

X = np.load(folder + "320m_images.npy")
# y = np.load(folder + "320m_labels.npy")[0] * (100 * 100) >= (140 * 3) # area
# y = np.load(folder + "320m_labels.npy")[1] * (100 * 100) >= (700 * 3) # volume
y = np.load(folder + "320m_labels.npy")[0] > 0 # ppl

seed = 42

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)
X_val = X_train[-10000:]
y_val = y_train[-10000:]
X_train = X_train[:-10000]
y_train = y_train[:-10000]


model = Sequential([
    Conv2D(32, kernel_size=(3,3), activation='swish', kernel_initializer='he_uniform', kernel_constraint=unit_norm(), input_shape=(32, 32, 6)),
    BatchNormalization(),
    MaxPool2D(pool_size=(2, 2)),
    BatchNormalization(),
    Conv2D(64, kernel_size=(3,3), activation='swish', kernel_initializer='he_uniform', kernel_constraint=unit_norm()),
    BatchNormalization(),
    # MaxPool2D(pool_size=(2, 2)),
    GlobalMaxPooling2D(),
    BatchNormalization(),
    Flatten(),
    Dense(64, activation='swish', kernel_initializer='he_uniform', kernel_constraint=max_norm(3)),
    Dropout(0.50),
    Dense(32, activation='swish', kernel_initializer='he_uniform', kernel_constraint=max_norm(3)),
    Dropout(0.50),
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

model.fit(X_train, y_train, epochs=25, verbose=1, batch_size=512, validation_data=(X_val, y_val), callbacks=[
    EarlyStopping(monitor='loss', patience=5)
])

loss, acc = model.evaluate(X_test, y_test, verbose=1)
print('Test Accuracy: %.3f' % acc)

import pdb; pdb.set_trace()