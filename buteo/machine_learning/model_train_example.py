
import os, gc
import numpy as np
import tensorflow as tf

from model_base import create_model
from tensorflow_utils import tpe, SaveBestModel, TimingCallback, OverfitProtection

from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.models import load_model

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
tf.keras.mixed_precision.set_global_policy('mixed_float16')

folder = "C:/Users/nmapservice/Desktop/CFI/data/"

# load
data_test = np.load(folder + "ghana_test.npz")
y_test = tf.convert_to_tensor(data_test["labels"], dtype=tf.float32, name="test_label")
x_test = [
    tf.convert_to_tensor(data_test["rgbn"], dtype=tf.float32, name="test_rgbn"),
    tf.convert_to_tensor(data_test["sar"], dtype=tf.float32, name="test_sar"),
    tf.convert_to_tensor(data_test["reswir"], dtype=tf.float32, name="test_reswir"),
]
data_test = None

data_train = np.load(folder + "ghana_train.npz")

print(f"Shape of raw train: {data_train['labels'].shape}")
limit_start = 0
limit_end = data_train["labels"].shape[0]

val_limit = int(data_train["labels"][limit_start:limit_end].shape[0] * 0.075)

# Split up due to memory consumption
labels = data_train["labels"][limit_start:limit_end]
y_val = tf.convert_to_tensor(labels[-val_limit:], dtype=tf.float32, name="val_label")
y_train = tf.convert_to_tensor(labels[:-val_limit], dtype=tf.float32, name="train_label")
labels = None
gc.collect() # Forcibly collect garbage

rgbn = data_train["rgbn"][limit_start:limit_end]
rgbn_val = tf.convert_to_tensor(rgbn[-val_limit:], dtype=tf.float32, name="val_rgbn")
rgbn_train = tf.convert_to_tensor(rgbn[:-val_limit], dtype=tf.float32, name="train_rgbn")
rgbn = None
gc.collect()

sar = data_train["sar"][limit_start:limit_end]
sar_val = tf.convert_to_tensor(sar[-val_limit:], dtype=tf.float32, name="val_sar")
sar_train = tf.convert_to_tensor(sar[:-val_limit], dtype=tf.float32, name="train_sar")
sar = None
gc.collect()

reswir = data_train["reswir"][limit_start:limit_end]
reswir_val = tf.convert_to_tensor(reswir[-val_limit:], dtype=tf.float32, name="val_reswir")
reswir_train = tf.convert_to_tensor(reswir[:-val_limit], dtype=tf.float32, name="train_reswir")
reswir = None

data_train = None
gc.collect()

x_val = [
    rgbn_val,
    sar_val,
    reswir_val,
]
x_train = [
    rgbn_train,
    sar_train,
    reswir_train,
]

shapes = (x_train[0].shape[1:], x_train[1].shape[1:], x_train[2].shape[1:])

min_delta = 0.005
fits = [
    { "epochs": 10, "bs": 16, "lr": 0.0001 },
    { "epochs": 10, "bs": 32, "lr": 0.0001 },
    { "epochs": 10, "bs": 48, "lr": 0.0001 },
    { "epochs": 10, "bs": 64, "lr": 0.0001 },
    { "epochs": 10, "bs": 80, "lr": 0.0001 },
    { "epochs": 10, "bs": 96, "lr": 0.0001 },
    { "epochs": 20, "bs": 112, "lr": 0.0001 },
    { "epochs": 20, "bs": 112, "lr": 0.00001 },
    { "epochs": 20, "bs": 112, "lr": 0.000001 },
]

cur_sum = 0
for idx, val in enumerate(fits):
    fits[idx]["ie"] = cur_sum
    cur_sum += fits[idx]["epochs"]

monitor = "val_loss"
outdir = "C:/Users/nmapservice/Desktop/CFI/models/"
model_name = "ghana_02"
donor_model = "ghana_01_10"

if donor_model is not None:
    model = load_model(outdir + donor_model, custom_objects={"tpe": tpe})
else:
    model = create_model(
        shapes[0],
        shapes[1],
        shapes[2],
        activation="relu",
        kernel_initializer=tf.keras.initializers.GlorotNormal(seed=42),
        inception_blocks=3,
        sizes=[48, 56, 64],
    )

    # model.summary()

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=fits[0]["lr"],
        epsilon=1e-07,
        amsgrad=False,
        name="Adam",
    )

    model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=["mse", "mae", tpe], 
    )

print("Validation test:")
val_loss, _mse, _mae, _tpe = model.evaluate(x=x_val, y=y_val, batch_size=256)

print("Test dataset:")
model.evaluate(x=x_test, y=y_test, batch_size=256)

# This ensures that the weights of the best performing model is saved at the end
save_best_model = SaveBestModel(save_best_metric=monitor, initial_weights=model.get_weights())

# Reduces the amount of total epochs by early stopping a new fit if it is not better than the previous fit.
best_val_loss = val_loss

for phase in range(len(fits)):
    use_epoch = fits[phase]["epochs"]
    use_bs = fits[phase]["bs"]
    use_lr = fits[phase]["lr"]
    use_ie = fits[phase]["ie"]

    model.optimizer.lr.assign(use_lr)
    model.fit(
        x=x_train,
        y=y_train,
        validation_data=(x_val, y_val),
        shuffle=True,
        epochs=use_epoch + use_ie,
        initial_epoch=use_ie,
        verbose=1,
        batch_size=use_bs,
        use_multiprocessing=True,
        workers=0,
        callbacks=[
            save_best_model,
            EarlyStopping(          # it gives the model 3 epochs to improve results based on val_loss value, if it doesnt improve-drops too much, the model running
                monitor=monitor,    # is stopped. If this continues, it would be overfitting (refer to notes)
                patience=5 if phase != 0 else 10,
                min_delta=min_delta,
                mode="min",         # loss is suppose to minimize
                baseline=best_val_loss,     # Fit has to improve upon baseline
                restore_best_weights=True,  # If stopped early, restore best weights.
            ),
            OverfitProtection(
                patience=3, # 
                difference=0.1, # 10% overfit allowed
                offset_start=1, # disregard overfit for the first epoch
            ),
            TimingCallback(
                monitor=[
                    # "loss", "val_loss",
                    "mse", "val_mse",
                    "mae", "val_mae",
                    "tpe", "val_tpe",
                ],
            ),
        ],
    )

    # Saves the val loss to the best_val_loss for early stopping between fits.
    model.set_weights(save_best_model.best_weights)
    
    print("Validation test:")
    val_loss, _mse, _mae, _tpe = model.evaluate(x=x_val, y=y_val, batch_size=256) # it evaluates the accuracy of the model we just created here
    best_val_loss = val_loss

    print("Test dataset:")
    model.evaluate(x=x_test, y=y_test, batch_size=256)
    model.save(f"{outdir}{model_name.lower()}_{str(use_epoch + use_ie)}")

print("Saving...")
model.save(f"{outdir}{model_name.lower()}")
