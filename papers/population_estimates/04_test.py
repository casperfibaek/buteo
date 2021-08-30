import os
import numpy as np

import tensorflow as tf
from tensorflow.keras import mixed_precision

np.set_printoptions(suppress=True)
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
mixed_precision.set_global_policy("mixed_float16")

folder = "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/data/denmark/"
test_folder = "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/data/bornholm/"


def test_model(model_path):
    x_test = [
        np.load(test_folder + "patches/RGBN.npy"),
        np.load(test_folder + "patches/SAR.npy"),
        np.load(test_folder + "patches/RESWIR.npy"),
    ]
    y_test = np.load(test_folder + "patches/label_area.npy")

    model = tf.keras.models.load_model(model_path)

    _loss, mse, mae = model.evaluate(
        x=x_test,
        y=y_test,
        verbose=1,
        batch_size=32,
        use_multiprocessing=True,
    )

    print(f"Path: {model_path}")
    print(f"Mean Square Error:      {round(mse, 5)}")
    print(f"Mean Absolute Error:    {round(mae, 5)}")
    # print(f"Mean Absolute P.Error:  {round(mape, 5)}")
    print("")


test_model(
    folder + "models/big_model_dk_01_36",
)
