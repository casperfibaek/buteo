import os
import pickle
import numpy as np
import tensorflow as tf

from tensorflow_utils import tpe, SaveBestModel, TimingCallback, OverfitProtection
from model_base import create_model

from tensorflow.keras import mixed_precision
from tensorflow.keras.callbacks import EarlyStopping

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
mixed_precision.set_global_policy("mixed_float16")

folder = "C:/Users/nmapservice/Desktop/CFI/"
x_train = pickle.load(open(folder + "x_train.pkl", 'rb'))
x_test = pickle.load(open(folder + "x_test.pkl", 'rb'))
y_train = pickle.load(open(folder + "y_train.pkl", 'rb'))
y_test = pickle.load(open(folder + "y_test.pkl", "rb"))

lr = 0.0001
min_delta = 0.005
epochs = [5, 10, 10]
bs = [32, 64, 128]
monitor = "val_loss"
outdir = folder + "models/"
model_name = "kili_base"

with tf.device("/device:GPU:0"):
    model = create_model(
        x_train[0].shape[1:],
        x_train[1].shape[1:],
        x_train[2].shape[1:],
        activation="relu",
        kernel_initializer=tf.keras.initializers.GlorotNormal(seed=42),
        inception_blocks=2,
        # sizes=[16, 16, 16],
    )

    model.summary()

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr,
        epsilon=1e-07,
        amsgrad=False,
        name="Adam",
    )

    model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=["mse", "mae", tpe], 
    )

    val_loss, _mse, _mae, _tpe = model.evaluate(x=x_test, y=y_test, batch_size=512)

    # This ensures that the weights of the best performing model is saved at the end
    save_best_model = SaveBestModel(save_best_metric=monitor)

    # Reduces the amount of total epochs by early stopping a new fit if it is not better than the previous fit.
    best_val_loss = val_loss
   
    for phase in range(len(bs)):
        use_epoch = np.cumsum(epochs)[phase]
        use_bs = bs[phase]
        initial_epoch = np.cumsum(epochs)[phase - 1] if phase != 0 else 0

        model.fit(
            x=x_train,
            y=y_train,
            validation_split=0.15,
            shuffle=True,
            epochs=use_epoch,
            initial_epoch=initial_epoch,
            verbose=1,
            batch_size=use_bs,
            use_multiprocessing=True,
            workers=0,
            callbacks=[
                save_best_model,
                EarlyStopping(          # it gives the model 3 epochs to improve results based on val_loss value, if it doesnt improve-drops too much, the model running
                    monitor=monitor,    # is stopped. If this continues, it would be overfitting (refer to notes)
                    patience=5,
                    min_delta=min_delta,
                    mode="min",         # loss is suppose to minimize
                    baseline=best_val_loss,     # Fit has to improve upon baseline
                    restore_best_weights=True,  # If stopped early, restore best weights.
                ),
                TimingCallback(
                    monitor=[
                        # "loss", "val_loss",
                        "mse", "val_mse",
                        "mae", "val_mae",
                        "tpe", "val_tpe",
                    ],
                ),
                OverfitProtection(
                    patience=3,
                    difference=0.1,
                    offset_start=3,
                ),
            ],
        )

        # Saves the val loss to the best_val_loss for early stopping between fits.
        model.set_weights(save_best_model.best_weights)
        val_loss, _mse, _mae, _tpe = model.evaluate(x=x_test, y=y_test, batch_size=512) # it evaluates the accuracy of the model we just created here
        best_val_loss = val_loss
        # model.save(f"{outdir}{model_name.lower()}_{str(use_epoch)}")

    print("Saving...")

    val_loss, _mse, _mae, _tpe = model.evaluate(x=x_test, y=y_test, batch_size=512) # it evaluates the accuracy of the model we just created here
    # model.save(f"{outdir}{model_name.lower()}")