
from sklearn.model_selection import train_test_split

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler

import numpy as np
import os, math

# Define globals
folder = "C:/Users/caspe/Desktop/wall_ezra/"

np.set_printoptions(suppress=True)
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

lr = 0.001
bs = 16
epochs = 20

# Load data. Ensure datatype and axes
X = np.load(folder + "training_by_peak_walls.npy")
X = X / X.max() # Here we are just normalising the max, maybe you want -7.0 to 255.0 or something similar
X = X[:, :, np.newaxis]

y = np.load(folder + "training_by_peak_walls_labels.npy")
wall_mask = y != -99

X = X[wall_mask]
y = y[wall_mask]


# Split the training set. Use stratified classes to ensure equal selection accoss the classes
X_train, X_test, y_train, y_test = train_test_split(X, y.astype("float32"), stratify=y, shuffle=True, test_size=0.33)

# The shape of the input (disregard rows)
shape = (X_train.shape[1], 1)

# Test out the different functions
activation = "relu"
kernel_initializer = "normal"

# Try out different model stuff here
model = keras.Sequential()
model.add(layers.Conv1D(filters=32, activation=activation, kernel_initializer=kernel_initializer, kernel_size=3, input_shape=shape))
model.add(layers.Conv1D(filters=64, activation=activation, kernel_initializer=kernel_initializer, kernel_size=3))
model.add(layers.Flatten())
model.add(layers.Dense(units=64, activation=activation, kernel_initializer=kernel_initializer))
model.add(layers.Dense(units=32, activation=activation, kernel_initializer=kernel_initializer))
model.add(layers.Dense(units=1, activation='relu', kernel_initializer=kernel_initializer))

def step_decay(epoch):
    initial_lrate = lr
    drop = 0.5
    epochs_drop = 3
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate

model.compile(
   loss='log_cosh',
   optimizer=Adam(
      learning_rate=lr,
      name="Adam",
    ),
    metrics=[
      'mae',
      'mse',
   ],
)

model.fit(
   x=X_train,
   y=y_train,
   epochs=epochs,
   batch_size=bs,
   verbose=1,
   validation_split=0.2,
   callbacks=[
      LearningRateScheduler(step_decay),
      EarlyStopping(
         monitor="val_loss",
         patience=5,
         min_delta=0.05,
         restore_best_weights=True,
      ),
   ],
)

print(f"Batch_size: {str(bs)}, learning_rate: {str(lr)}")
log_cosh, mae, mse = model.evaluate(X_test, y_test, verbose=0)
print(f"Mean Square Error:      {round(mse, 3)}")
print(f"Mean Absolute Error:    {round(mae, 3)}")
print(f"Log Cosh:               {round(log_cosh, 3)}")
print("")

# Batch_size: 32, learning_rate: 0.001
# Mean Square Error:      20.226
# Mean Absolute Error:    1.682

# Batch_size: 32, learning_rate: 0.001
# Mean Square Error:      21.373
# Mean Absolute Error:    2.39

# Investigate the results
pred = model.predict(X_test)

pred_int = np.rint(pred[:, 0]).astype(int) 
true_int = np.rint(y_test).astype(int)

comb = np.stack([pred_int, true_int])
print(comb)

# [[28 14 27 ... 24 26 19]
#  [27 18 27 ... 25 31 18]]

import pdb; pdb.set_trace()