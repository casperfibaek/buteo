yellow_follow = "C:/Users/caspe/Desktop/buteo/"
import sys

sys.path.append(yellow_follow)

import os
import numpy as np

import tensorflow as tf
from tensorflow.keras import mixed_precision

np.set_printoptions(suppress=True)
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
mixed_precision.set_global_policy("mixed_float16")

folder = "C:/Users/caspe/Desktop/paper_2_Structural_Volume/data/"


def test_model(model_path, label, inputs):
    for place in ["holsterbro", "aarhus", "samsoe"]:
        x_test = []
        for layer in inputs:
            x_test.append(np.load(folder + f"patches/{place}_{layer}.npy"))

        y_test = np.load(folder + f"patches/{place}_label_{label}.npy")

        if label == "people":
            y_test = y_test * 10.0
        elif label == "area":
            y_test = y_test * 1.0
        elif label == "volume":
            y_test = y_test * 0.01
        else:
            raise Exception("Wrong label used.")

        model = tf.keras.models.load_model(model_path)

        _loss, mse, mae = model.evaluate(
            x=x_test,
            y=y_test,
            verbose=1,
            batch_size=32,
            use_multiprocessing=True,
        )

        print(place)
        print(f"Path: {model_path}")
        print(f"Label: {label}")
        print(f"Mean Square Error:      {round(mse, 3)}")
        print(f"Mean Absolute Error:    {round(mae, 3)}")
        print("")


test_model(folder + "models/rgb_people_32", "people", ["RGB"])
