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


def get_layer(name, place=""):
    if name == "rgb":
        return np.load(folder + f"patches/{place}_RGB.npy")
    elif name == "rgbn":
        return np.load(folder + f"patches/{place}_RGBN.npy")
    elif name == "VVa":
        return np.load(folder + f"patches/{place}_vv_asc_v2.npy")
    elif name == "VVa_VHa":
        return np.concatenate(
            [
                np.load(folder + f"patches/{place}_vv_asc_v2.npy"),
                np.load(folder + f"patches/{place}_vh_asc_v2.npy"),
            ],
            axis=3,
        )
    elif name == "VVa_VVd":
        return np.concatenate(
            [
                np.load(folder + f"patches/{place}_vv_asc.npy"),
                np.load(folder + f"patches/{place}_vv_desc.npy"),
            ],
            axis=3,
        )
    elif name == "VVa_COHa":
        return np.concatenate(
            [
                np.load(folder + f"patches/{place}_vv_asc.npy"),
                np.load(folder + f"patches/{place}_coh_asc.npy"),
            ],
            axis=3,
        )
    elif name == "VVa_VVd_COHa_COHd":
        return np.concatenate(
            [
                np.load(folder + f"patches/{place}_vv_asc.npy"),
                np.load(folder + f"patches/{place}_vv_desc.npy"),
                np.load(folder + f"patches/{place}_coh_asc.npy"),
                np.load(folder + f"patches/{place}_coh_desc.npy"),
            ],
            axis=3,
        )
    elif name == "VVa_VHa_COHa":
        return np.concatenate(
            [
                np.load(folder + f"patches/{place}_vv_asc_v2.npy"),
                np.load(folder + f"patches/{place}_vh_asc_v2.npy"),
                np.load(folder + f"patches/{place}_coh_asc.npy"),
            ],
            axis=3,
        )
    else:
        raise Exception("Could not find layer.")


def test_model(model_path, label, inputs):
    for place in ["holsterbro", "aarhus", "samsoe"]:
        x_test = get_layer(inputs, place)
        y_test = np.load(folder + f"patches/{place}_label_{label}.npy")

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
        print(f"Mean Square Error:      {round(mse, 5)}")
        print(f"Mean Absolute Error:    {round(mae, 5)}")
        print("")


test_model(folder + "models/vva_vvd_coha_cohd_people_34", "PEOPLE", "VVa_VVd_COHa_COHd")
