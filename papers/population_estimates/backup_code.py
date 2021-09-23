yellow_follow = "C:/Users/caspe/Desktop/buteo/"
import sys

sys.path.append(yellow_follow)

import os
import numpy as np

# from model_trio_down import model_trio_down

np.set_printoptions(suppress=True)

folder = "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/data/"
outdir = folder + f"models/"
place = "dojo"

# y_train_dk = np.load(folder + f"{place}/dk_label_area.npy")
# shuffle_mask_dk = np.random.permutation(y_train_dk.shape[0])

# rgbn_dk = np.load(folder + f"{place}/dk_RGBN.npy")[shuffle_mask_dk]
# sar_dk = np.load(folder + f"{place}/dk_SAR.npy")[shuffle_mask_dk]
# reswir_dk = np.load(folder + f"{place}/dk_RESWIR.npy")[shuffle_mask_dk]
# y_train_dk = y_train_dk[shuffle_mask_dk]

# np.save(folder + f"{place}/dk_RGBN_reduced.npy", rgbn_dk[:400000].astype("float32"))
# np.save(folder + f"{place}/dk_SAR_reduced.npy", sar_dk[:400000].astype("float32"))
# np.save(folder + f"{place}/dk_RESWIR_reduced.npy", reswir_dk[:400000].astype("float32"))
# np.save(
#     folder + f"{place}/dk_label_area_reduced.npy", y_train_dk[:400000].astype("float32")
# )

# y_train_dk = None
# shuffle_mask_dk = None
# rgbn_dk = None
# sar_dk = None
# reswir_dk = None
# y_train_dk = None
# mask_dk = None

noise = {
    "scale": 0.05,
    "band": 0.02,
    "pixel": 0.005,
}

# use_noise = True
# prefix = "_noise"

use_noise = False
prefix = ""

y_label = np.concatenate(
    [
        np.load(folder + f"{place}/ghana_label_area.npy"),
        np.load(folder + f"{place}/aux_label_area.npy"),
        np.load(folder + f"{place}/dk_reduced_label_area.npy"),
    ]
).astype("float32")

# noise scale
if use_noise:
    scale_noise = np.random.normal(1.0, noise["scale"], (len(y_label), 1, 1, 1)).astype(
        "float32"
    )

np.save(folder + f"{place}/all{prefix}_label_area.npy", y_label)
y_label = None

rgbn = np.concatenate(
    [
        np.load(folder + f"{place}/ghana_RGBN.npy"),
        np.load(folder + f"{place}/aux_RGBN.npy"),
        np.load(folder + f"{place}/dk_reduced_RGBN.npy"),
    ]
).astype("float32")

if use_noise:
    band_noise = np.random.normal(
        1.0, noise["band"], (rgbn.shape[0], 1, 1, rgbn.shape[3])
    ).astype("float32")
    pixel_noise = np.random.normal(1.0, noise["pixel"], rgbn.shape).astype("float32")
    rgbn = rgbn * scale_noise * band_noise * pixel_noise

np.save(folder + f"{place}/all{prefix}_RGBN.npy", rgbn.astype("float32"))
rgbn = None

sar = np.concatenate(
    [
        np.load(folder + f"{place}/ghana_SAR.npy"),
        np.load(folder + f"{place}/aux_SAR.npy"),
        np.load(folder + f"{place}/dk_reduced_SAR.npy"),
    ]
).astype("float32")

if use_noise:
    band_noise = np.random.normal(
        1.0, noise["band"], (sar.shape[0], 1, 1, sar.shape[3])
    ).astype("float32")
    pixel_noise = np.random.normal(1.0, noise["pixel"], sar.shape).astype("float32")
    sar = sar * scale_noise * band_noise * pixel_noise

np.save(folder + f"{place}/all{prefix}_SAR.npy", sar.astype("float32"))
sar = None

reswir = np.concatenate(
    [
        np.load(folder + f"{place}/ghana_RESWIR.npy"),
        np.load(folder + f"{place}/aux_RESWIR.npy"),
        np.load(folder + f"{place}/dk_reduced_RESWIR.npy"),
    ]
).astype("float32")

if use_noise:
    band_noise = np.random.normal(
        1.0, noise["band"], (reswir.shape[0], 1, 1, reswir.shape[3])
    ).astype("float32")
    pixel_noise = np.random.normal(1.0, noise["pixel"], reswir.shape).astype("float32")
    reswir = reswir * scale_noise * band_noise * pixel_noise

np.save(folder + f"{place}/all{prefix}_RESWIR.npy", reswir.astype("float32"))
reswir = None


exit()

# np.save(
#     folder + f"{place}/all_RGBN_32.npy",
#     np.load(folder + f"{place}/all_RGBN.npy").astype("float32"),
# )
# np.save(
#     folder + f"{place}/all_SAR_32.npy",
#     np.load(folder + f"{place}/all_SAR.npy").astype("float32"),
# )
# np.save(
#     folder + f"{place}/all_RESWIR_32.npy",
#     np.load(folder + f"{place}/all_RESWIR.npy").astype("float32"),
# )
# np.save(
#     folder + f"{place}/all_label_area_32.npy",
#     np.load(folder + f"{place}/all_label_area.npy").astype("float32"),
# )

# exit()

# x_train = [
#     np.concatenate(
#         [
#             np.load(folder + f"{place}/RGBN.npy"),
#             np.load(folder + f"{place}/extra_RGBN.npy"),
#             np.load(folder + f"{place}/dk_RGBN_red.npy"),
#         ]
#     ),
#     np.concatenate(
#         [
#             np.load(folder + f"{place}/SAR.npy"),
#             np.load(folder + f"{place}/extra_SAR.npy"),
#             np.load(folder + f"{place}/dk_SAR_red.npy"),
#         ]
#     ),
#     np.concatenate(
#         [
#             np.load(folder + f"{place}/RESWIR.npy"),
#             np.load(folder + f"{place}/extra_RESWIR.npy"),
#             np.load(folder + f"{place}/dk_RESWIR_red.npy"),
#         ]
#     ),
# ]

# y_train = np.concatenate(
#     [
#         np.load(folder + f"{place}/label_area.npy"),
#         np.load(folder + f"{place}/extra_label_area.npy"),
#         np.load(folder + f"{place}/dk_label_area_red.npy"),
#     ]
# )

# y_train_dk = None
# shuffle_mask_dk = None
# rgbn_dk = None
# sar_dk = None
# reswir_dk = None
# y_train_dk = None
# mask_dk = None


# shuffle_mask = np.random.permutation(y_train.shape[0])

# for idx in range(len(x_train)):
#     x_train[idx] = x_train[idx][shuffle_mask]

# y_train = y_train[shuffle_mask]

# limit_mask = y_train.sum(axis=(1, 2))[:, 0] > 100

# for idx in range(len(x_train)):
#     x_train[idx] = x_train[idx][limit_mask]

# y_train = y_train[limit_mask]

# x_val = []
# for idx in range(len(x_train)):
#     x_val.append(x_train[idx][-10000:])

# y_val = y_train[-10000:]

# for idx in range(len(x_train)):
#     x_train[idx] = x_train[idx][:20000]

# y_train = y_train[:20000]

# # noise scale
# scale_noise = np.random.normal(1.0, 0.02, len(y_train))

# # noise
# for idx in range(len(x_train)):
#     x_train[idx] = (
#         x_train[idx]
#         * scale_noise[idx]
#         * np.random.normal(1.0, 0.001, x_train[idx].shape)
#     )
