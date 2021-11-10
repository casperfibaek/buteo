import sys

yellow_follow = "C:/Users/caspe/Desktop/buteo/"
sys.path.append(yellow_follow)
import numpy as np
from buteo.filters.convolutions import interp_array


def image_augmentation(list_of_inputs, label, batch_size, shuffle=True, options=None):

    if shuffle:
        shuffle_mask = np.random.permutation(label.shape[0])
        label = label[shuffle_mask]

    batches = list_of_inputs[0].shape[0] // batch_size
    yielded = 0

    base_options = {
        "scale": 0.075,
        "contrast": 0.035,
        "band": 0.025,
        "pixel": 0.015,
        "drop_threshold": 0.01,
        "fliplr": 0.2,
        "flipud": 0.1,
    }

    if options is None:
        options = base_options
    else:
        for key in options:
            if key not in base_options:
                raise ValueError("Invalid option: {}".format(key))

    while yielded < batches:

        batch = []

        for arr in list_of_inputs:

            if shuffle:
                arr = arr[shuffle_mask]

            base = arr[yielded * batch_size : (yielded + 1) * batch_size]

            scale = np.random.normal(1.0, options["scale"], (len(base), 1, 1, 1))

            cmax = np.random.normal(1.0, options["contrast"], (base.shape[0], 1, 1, 1))
            cmin = np.random.normal(1.0, options["contrast"], (base.shape[0], 1, 1, 1))

            band = np.random.normal(
                1.0, options["band"], (base.shape[0], 1, 1, base.shape[3])
            )

            pixel = np.random.normal(1.0, options["pixel"], base.shape)

            base = base * scale * band * pixel

            min_vals = base.min(axis=(1, 2)).reshape(
                (base.shape[0], 1, 1, base.shape[3])
            )
            max_vals = base.max(axis=(1, 2)).reshape(
                (base.shape[0], 1, 1, base.shape[3])
            )
            min_vals_adj = min_vals * cmin
            max_vals_adj = max_vals * cmax
            min_vals_adj = np.where(
                min_vals_adj >= max_vals_adj, min_vals, min_vals_adj
            )
            max_vals_adj = np.where(
                max_vals_adj <= min_vals_adj, max_vals, max_vals_adj
            )

            base = interp_array(base, min_vals, max_vals, min_vals_adj, max_vals_adj)
            base = base * (
                np.random.rand(base.shape[0], 1, 1, base.shape[3])
                > options["drop_threshold"]
            )

            batch.append(base)

        batch_shuffle = np.random.permutation(batch_size)
        flip_lr_mask = np.random.rand(batch_size) > base_options["fliplr"]
        flip_ud_mask = np.random.rand(batch_size) > base_options["flipud"]

        label_base = label[yielded * batch_size : (yielded + 1) * batch_size]

        label_batch = [
            np.concatenate(
                [
                    label_base[flip_lr_mask],
                    np.fliplr(label_base[~flip_lr_mask]),
                ]
            )[batch_shuffle]
        ]

        for idx, inp in enumerate(batch):
            flipped_lr_base = inp[flip_lr_mask]
            flipped_lr = np.fliplr(inp[~flip_lr_mask])

            batch[idx] = np.concatenate(
                [
                    flipped_lr_base,
                    flipped_lr,
                ]
            )[batch_shuffle]

            flipped_ud_base = inp[flip_ud_mask]
            flipped_lr = np.flipud(inp[~flip_ud_mask])

            batch[idx] = np.concatenate(
                [
                    flipped_ud_base,
                    flipped_lr,
                ]
            )[batch_shuffle]

        yield batch, label_batch
        yielded += 1

    return


# folder = "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/data/bornholm/patches/"

# x_train = [
#     np.load(folder + "RGBN.npy"),
#     np.load(folder + "SAR.npy"),
#     np.load(folder + "RESWIR.npy"),
# ]

# y_train = np.load(folder + "label_area.npy")

# bob = image_augmentation(x_train, y_train, batch_size=32, shuffle=True)
# carl = next(bob)

# import pdb

# pdb.set_trace()
