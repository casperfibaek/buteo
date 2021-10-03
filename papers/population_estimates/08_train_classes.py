yellow_follow = "C:/Users/caspe/Desktop/buteo/"
import sys

sys.path.append(yellow_follow)

import os
import time
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

import tensorflow as tf
import keras.backend as K
from tensorflow.keras import mixed_precision
from tensorflow.keras.callbacks import (
    LearningRateScheduler,
    ModelCheckpoint,
    EarlyStopping,
)
from buteo.machine_learning.ml_utils import create_step_decay, SaveBestModel
from buteo.utils import timing
from model_trio_down_class import model_trio_down_class

np.set_printoptions(suppress=True)
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
mixed_precision.set_global_policy("mixed_float16")

model_name = "class_01"

folder = "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/data/"
outdir = folder + f"models/"
place = "dojo"

x_train = [
    np.load(folder + f"{place}/class_balanced_RGBN.npy"),
    np.load(folder + f"{place}/class_balanced_SAR.npy"),
    np.load(folder + f"{place}/class_balanced_RESWIR.npy"),
]

y_train = np.load(folder + f"{place}/class_balanced_label_area.npy")

shuffle_mask = np.random.permutation(y_train.shape[0])

for idx in range(len(x_train)):
    x_train[idx] = x_train[idx][shuffle_mask]

y_train = y_train[shuffle_mask].astype("float32")

x_test = []
for idx in range(len(x_train)):
    x_test.append(x_train[idx][-500:])
    x_train[idx] = x_train[idx][:-500]

y_test = y_train[-500:]
y_train = y_train[:-500]


# x_test = [
#     np.load(folder + f"{place}/test_RGBN.npy"),
#     np.load(folder + f"{place}/test_SAR.npy"),
#     np.load(folder + f"{place}/test_RESWIR.npy"),
# ]

# y_test = np.load(folder + f"{place}/test_label_area.npy")

cross_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
lr = 0.0001
min_delta = 0.005

with tf.device("/device:GPU:0"):
    # epochs = [20, 10, 5]
    # bs = [256, 128, 64]
    epochs = [5, 10, 15, 20]
    bs = [32, 64, 128, 256]
    inception_blocks = 3
    activation = "relu"
    output_activation = "softmax"
    initializer = "glorot_normal"

    donor_model_path = outdir + "student2"

    model = model_trio_down_class(
        x_train[0].shape[1:],
        x_train[1].shape[1:],
        x_train[2].shape[1:],
        donor_model_path,
        4,
        kernel_initializer=initializer,
        activation=activation,
        inception_blocks=inception_blocks,
        name=f"{model_name.lower()}",
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr,
        epsilon=1e-07,
        amsgrad=False,
        name="Adam",
    )

    model.compile(
        optimizer=optimizer,
        loss=cross_loss,
        metrics=["accuracy"],
    )

    start = time.time()

    save_best_model = SaveBestModel()

    for phase in range(len(bs)):
        use_epoch = np.cumsum(epochs)[phase]
        use_bs = bs[phase]
        initial_epoch = np.cumsum(epochs)[phase - 1] if phase != 0 else 0

        model.fit(
            x=x_train,
            y=y_train,
            validation_split=0.1,
            shuffle=True,
            epochs=use_epoch,
            initial_epoch=initial_epoch,
            verbose=1,
            batch_size=use_bs,
            use_multiprocessing=True,
            workers=0,
            callbacks=[
                EarlyStopping(
                    monitor="val_loss",
                    patience=3,
                    min_delta=min_delta,
                    restore_best_weights=True,
                ),
                save_best_model,
            ],
        )

    model.set_weights(save_best_model.best_weights)
    model.save(f"{outdir}{model_name.lower()}")

    timing(start)


def evaluate(x_test, y_test):
    y_pred = model.predict(x_test, verbose=1)

    def reshape(arr):
        return arr.reshape(arr.shape[0] * arr.shape[1] * arr.shape[2], arr.shape[3])

    def pred_to_eval(arr):
        return np.argmax(reshape(arr), axis=1)

    matrix = confusion_matrix(reshape(y_test), pred_to_eval(y_pred), normalize="pred")

    disp = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Uninhabited", "Residential", "Industrial", "Slum"],
    )
    disp.plot()
    plt.show()


evaluate(x_test, y_test)

import pdb

pdb.set_trace()


# teacher: mse: 104.7100 - mae: 3.4287 - tpe: 1.0075
# student: mse: 97.1954 - mae: 3.3744 - tpe: -0.0915
