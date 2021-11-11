import sys

yellow_follow = "C:/Users/caspe/Desktop/buteo/"
sys.path.append(yellow_follow)
import numpy as np
from buteo.filters.convolutions import interp_array
from buteo.machine_learning.ml_utils import scale_to_range


def image_augmentation(list_of_inputs, list_of_labels, shuffle=True, options=None):
    """
    Augment the input images with random rotations, flips, and noise.
    """
    if not isinstance(list_of_inputs, list):
        list_of_inputs = [list_of_inputs]

    if not isinstance(list_of_labels, list):
        list_of_labels = [list_of_labels]

    base_options = {
        "scale": 0.075,
        "band": 0.035,
        "contrast": 0.025,
        "pixel": 0.02,
        "drop_threshold": 0.00,
        "clamp": False,
        "clamp_max": 1,
        "clamp_min": 0,
    }

    if options is None:
        options = base_options
    else:
        for key in options:
            if key not in base_options:
                raise ValueError(f"Invalid option: {key}")
            base_options[key] = options[key]
        options = base_options

    if shuffle:
        shuffle_mask = np.random.permutation(list_of_labels[0].shape[0])

    x_outputs = []
    y_outputs = []

    for arr in list_of_labels:
        if shuffle:
            y_outputs.append(arr[shuffle_mask])
        else:
            y_outputs.append(arr)

    for arr in list_of_inputs:
        base = np.array(arr, copy=True)

        scale = np.random.normal(1.0, options["scale"], (len(base), 1, 1, 1))

        cmax = np.random.normal(1.0, options["contrast"], (base.shape[0], 1, 1, 1))
        cmin = np.random.normal(1.0, options["contrast"], (base.shape[0], 1, 1, 1))

        band = np.random.normal(
            1.0, options["band"], (base.shape[0], 1, 1, base.shape[3])
        )

        pixel = np.random.normal(1.0, options["pixel"], base.shape)

        base = base * scale * band * pixel

        min_vals = base.min(axis=(1, 2)).reshape((base.shape[0], 1, 1, base.shape[3]))
        max_vals = base.max(axis=(1, 2)).reshape((base.shape[0], 1, 1, base.shape[3]))
        min_vals_adj = min_vals * cmin
        max_vals_adj = max_vals * cmax
        min_vals_adj = np.where(min_vals_adj >= max_vals_adj, min_vals, min_vals_adj)
        max_vals_adj = np.where(max_vals_adj <= min_vals_adj, max_vals, max_vals_adj)

        base = interp_array(base, min_vals, max_vals, min_vals_adj, max_vals_adj)
        base = base * (
            np.random.rand(base.shape[0], 1, 1, base.shape[3])
            > options["drop_threshold"]
        )

        if options["clamp"]:
            base = np.interp(
                base,
                (base.min(), base.max()),
                (options["clamp_min"], options["clamp_max"]),
            )

        if shuffle:
            x_outputs.append(base[shuffle_mask])
        else:
            x_outputs.append(base)

    return x_outputs, y_outputs
