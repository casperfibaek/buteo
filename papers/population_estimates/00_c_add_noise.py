yellow_follow = "C:/Users/caspe/Desktop/buteo/"
import sys

from numpy.random import shuffle

sys.path.append(yellow_follow)

import os
import numpy as np
from glob import glob
from buteo.filters.convolutions import interp_array

# from model_trio_down import model_trio_down

np.set_printoptions(suppress=True)

folder = "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/data/"
outdir = folder + f"models/"
place = "dojo"

# for val in range(3):
#     num = str(val)
#     y_train_dk = np.load(folder + f"{place}/dk_balanced_label_area.npy")
#     shuffle_mask_dk = np.random.permutation(y_train_dk.shape[0])

#     rgbn_dk = np.load(folder + f"{place}/dk_balanced_RGBN.npy")[shuffle_mask_dk]
#     sar_dk = np.load(folder + f"{place}/dk_balanced_SAR.npy")[shuffle_mask_dk]
#     reswir_dk = np.load(folder + f"{place}/dk_balanced_RESWIR.npy")[shuffle_mask_dk]
#     y_train_dk = y_train_dk[shuffle_mask_dk]

#     np.save(
#         folder + f"{place}/dk_reduced{num}_RGBN.npy", rgbn_dk[:300000].astype("float32")
#     )
#     np.save(
#         folder + f"{place}/dk_reduced{num}_SAR.npy", sar_dk[:300000].astype("float32")
#     )
#     np.save(
#         folder + f"{place}/dk_reduced{num}_RESWIR.npy",
#         reswir_dk[:300000].astype("float32"),
#     )
#     np.save(
#         folder + f"{place}/dk_reduced{num}_label_area.npy",
#         y_train_dk[:300000].astype("float32"),
#     )

#     y_train_dk = None
#     shuffle_mask_dk = None
#     rgbn_dk = None
#     sar_dk = None
#     reswir_dk = None
#     y_train_dk = None
#     mask_dk = None

# exit()

noise = {
    "scale": 0.05,
    "contrast": 0.025,
    "band": 0.01,
    "pixel": 0.005,
}

# dk_version = "2"

use_noise = True
prefix = f"ghana_noise"

# use_noise = False
# prefix = "_nonoise"


y_label = np.concatenate(
    [
        # np.load(folder + f"{place}/ghana_balanced_label_area.npy"),
        np.load(folder + f"{place}/patches/ghana_label_area.npy"),
        # np.load(folder + f"{place}/aux_balanced_label_area.npy"),
        # np.load(folder + f"{place}/dk_reduced{dk_version}_label_area.npy"),
    ]
).astype("float32")

shuffle_mask = np.random.permutation(y_label.shape[0])

# noise scale
if use_noise:
    scale_noise = np.random.normal(1.0, noise["scale"], (len(y_label), 1, 1, 1)).astype(
        "float32"
    )

np.save(folder + f"{place}/{prefix}_label_area.npy", y_label[shuffle_mask])
y_label = None

rgbn = np.concatenate(
    [
        np.load(folder + f"{place}/patches/ghana_RGBN.npy"),
        # np.load(folder + f"{place}/ghana_balanced_RGBN.npy"),
        # np.load(folder + f"{place}/aux_balanced_RGBN.npy"),
        # np.load(folder + f"{place}/dk_reduced{dk_version}_RGBN.npy"),
    ]
).astype("float32")

if use_noise:
    cmax_rgbn = np.random.normal(
        1.0, noise["contrast"], (rgbn.shape[0], 1, 1, 1)
    ).astype("float32")
    cmin_rgbn = np.random.normal(
        1.0, noise["contrast"], (rgbn.shape[0], 1, 1, 1)
    ).astype("float32")
    band_noise = np.random.normal(
        1.0, noise["band"], (rgbn.shape[0], 1, 1, rgbn.shape[3])
    ).astype("float32")
    pixel_noise = np.random.normal(1.0, noise["pixel"], rgbn.shape).astype("float32")

    rgbn = rgbn * scale_noise * band_noise * pixel_noise

    min_vals = rgbn.min(axis=(1, 2)).reshape((rgbn.shape[0], 1, 1, rgbn.shape[3]))
    max_vals = rgbn.max(axis=(1, 2)).reshape((rgbn.shape[0], 1, 1, rgbn.shape[3]))
    min_vals_adj = min_vals * cmin_rgbn
    max_vals_adj = max_vals * cmax_rgbn
    min_vals_adj = np.where(min_vals_adj >= max_vals_adj, min_vals, min_vals_adj)
    max_vals_adj = np.where(max_vals_adj <= min_vals_adj, max_vals, max_vals_adj)

    rgbn = interp_array(rgbn, min_vals, max_vals, min_vals_adj, max_vals_adj)


np.save(folder + f"{place}/{prefix}_RGBN.npy", rgbn[shuffle_mask].astype("float32"))
rgbn = None

sar = np.concatenate(
    [
        np.load(folder + f"{place}/patches/ghana_SAR.npy"),
        # np.load(folder + f"{place}/ghana_balanced_SAR.npy"),
        # np.load(folder + f"{place}/aux_balanced_SAR.npy"),
        # np.load(folder + f"{place}/dk_reduced{dk_version}_SAR.npy"),
    ]
).astype("float32")

if use_noise:
    cmax = np.random.normal(1.0, noise["contrast"], (sar.shape[0], 1, 1, 1)).astype(
        "float32"
    )
    cmin = np.random.normal(1.0, noise["contrast"], (sar.shape[0], 1, 1, 1)).astype(
        "float32"
    )
    band_noise = np.random.normal(
        1.0, noise["band"], (sar.shape[0], 1, 1, sar.shape[3])
    ).astype("float32")
    pixel_noise = np.random.normal(1.0, noise["pixel"], sar.shape).astype("float32")

    sar = sar * scale_noise * band_noise * pixel_noise

    min_vals = sar.min(axis=(1, 2)).reshape((sar.shape[0], 1, 1, sar.shape[3]))
    max_vals = sar.max(axis=(1, 2)).reshape((sar.shape[0], 1, 1, sar.shape[3]))
    min_vals_adj = min_vals * cmin
    max_vals_adj = max_vals * cmax
    min_vals_adj = np.where(min_vals_adj >= max_vals_adj, min_vals, min_vals_adj)
    max_vals_adj = np.where(max_vals_adj <= min_vals_adj, max_vals, max_vals_adj)

    sar = interp_array(sar, min_vals, max_vals, min_vals_adj, max_vals_adj)


np.save(folder + f"{place}/{prefix}_SAR.npy", sar[shuffle_mask].astype("float32"))
sar = None

reswir = np.concatenate(
    [
        np.load(folder + f"{place}/patches/ghana_RESWIR.npy"),
        # np.load(folder + f"{place}/ghana_balanced_RESWIR.npy"),
        # np.load(folder + f"{place}/aux_balanced_RESWIR.npy"),
        # np.load(folder + f"{place}/dk_reduced{dk_version}_RESWIR.npy"),
    ]
).astype("float32")

if use_noise:
    band_noise = np.random.normal(
        1.0, noise["band"], (reswir.shape[0], 1, 1, reswir.shape[3])
    ).astype("float32")
    pixel_noise = np.random.normal(1.0, noise["pixel"], reswir.shape).astype("float32")

    reswir = reswir * scale_noise * band_noise * pixel_noise

    min_vals = reswir.min(axis=(1, 2)).reshape((reswir.shape[0], 1, 1, reswir.shape[3]))
    max_vals = reswir.max(axis=(1, 2)).reshape((reswir.shape[0], 1, 1, reswir.shape[3]))
    min_vals_adj = min_vals * cmin_rgbn
    max_vals_adj = max_vals * cmax_rgbn
    min_vals_adj = np.where(min_vals_adj >= max_vals_adj, min_vals, min_vals_adj)
    max_vals_adj = np.where(max_vals_adj <= min_vals_adj, max_vals, max_vals_adj)

    reswir = interp_array(reswir, min_vals, max_vals, min_vals_adj, max_vals_adj)

np.save(folder + f"{place}/{prefix}_RESWIR.npy", reswir[shuffle_mask].astype("float32"))
reswir = None


exit()

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

# base_kept = np.load(folder + f"{place}/ghana_label_area.npy")
# base = np.load(folder + f"{place}/dk_balanced_label_area.npy")
# shuffle_mask = np.random.permutation(base.shape[0])

# base = base[shuffle_mask]

# zero_mask = (base.sum(axis=(1, 2)) == 0)[:, 0]
# value_mask = (base.sum(axis=(1, 2)) > 0)[:, 0]
# above_sum = ((base.sum(axis=(1, 2)) > 100)[:, 0]).sum()

# zeros = shuffle_mask[zero_mask]
# zeros = zeros[:above_sum]

# values = shuffle_mask[value_mask]

# base_new = np.concatenate([shuffle_mask[zeros], shuffle_mask[values]])
# final_shuffle_mask = np.random.permutation(base_new.shape[0])
# mask = shuffle_mask[final_shuffle_mask]

# area_masked = np.load(folder + f"{place}/dk_balanced_label_area.npy")[:300000]
# np.save(folder + f"{place}/dk_reduced_label_area.npy", area_masked)

# volume_masked = np.load(folder + f"{place}/dk_balanced_label_volume.npy")[:300000]
# np.save(folder + f"{place}/dk_reduced_label_volume.npy", volume_masked)

# rgbn_masked = np.load(folder + f"{place}/dk_balanced_RGBN.npy")[:300000]
# np.save(folder + f"{place}/dk_reduced_RGBN.npy", rgbn_masked)

# sar_masked = np.load(folder + f"{place}/dk_balanced_SAR.npy")[:300000]
# np.save(folder + f"{place}/dk_reduced_SAR.npy", sar_masked)

# reswir_masked = np.load(folder + f"{place}/dk_balanced_RESWIR.npy")[:300000]
# np.save(folder + f"{place}/dk_reduced_RESWIR.npy", reswir_masked)

# y_label = np.concatenate(
#     [
#         np.load(folder + f"{place}/patches/dar_label_area.npy"),
#         np.load(folder + f"{place}/patches/kampala_label_area.npy"),
#         np.load(folder + f"{place}/patches/kilimanjaro_label_area.npy"),
#         np.load(folder + f"{place}/patches/mwanza_label_area.npy"),
#     ]
# )

# shuffle_mask = np.random.permutation(y_label.shape[0])

# y_label = y_label[shuffle_mask]

# rgbn = np.concatenate(
#     [
#         np.load(folder + f"{place}/patches/dar_RGBN.npy"),
#         np.load(folder + f"{place}/patches/kampala_RGBN.npy"),
#         np.load(folder + f"{place}/patches/kilimanjaro_RGBN.npy"),
#         np.load(folder + f"{place}/patches/mwanza_RGBN.npy"),
#     ]
# )[shuffle_mask]

# sar = np.concatenate(
#     [
#         np.load(folder + f"{place}/patches/dar_SAR.npy"),
#         np.load(folder + f"{place}/patches/kampala_SAR.npy"),
#         np.load(folder + f"{place}/patches/kilimanjaro_SAR.npy"),
#         np.load(folder + f"{place}/patches/mwanza_SAR.npy"),
#     ]
# )[shuffle_mask]

# reswir = np.concatenate(
#     [
#         np.load(folder + f"{place}/patches/dar_RESWIR.npy"),
#         np.load(folder + f"{place}/patches/kampala_RESWIR.npy"),
#         np.load(folder + f"{place}/patches/kilimanjaro_RESWIR.npy"),
#         np.load(folder + f"{place}/patches/mwanza_RESWIR.npy"),
#     ]
# )[shuffle_mask]

# np.save(folder + f"{place}/patches/aux_label_area.npy", y_label)
# np.save(folder + f"{place}/patches/aux_RGBN.npy", rgbn)
# np.save(folder + f"{place}/patches/aux_SAR.npy", sar)
# np.save(folder + f"{place}/patches/aux_RESWIR.npy", reswir)
