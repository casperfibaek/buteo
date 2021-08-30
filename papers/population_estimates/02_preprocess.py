import sys, numpy as np

sys.path.append("../../")
from buteo.machine_learning.ml_utils import (
    preprocess_optical,
    preprocess_sar,
)


def preprocess(prefix, folder, outdir, low=0, high=1, optical_top=8000):
    b02 = folder + f"{prefix}_B02_10m.npy"
    b03 = folder + f"{prefix}_B03_10m.npy"
    b04 = folder + f"{prefix}_B04_10m.npy"
    b08 = folder + f"{prefix}_B08_10m.npy"

    b05 = folder + f"{prefix}_B05_20m.npy"
    b06 = folder + f"{prefix}_B06_20m.npy"
    b07 = folder + f"{prefix}_B07_20m.npy"
    b11 = folder + f"{prefix}_B11_20m.npy"
    b12 = folder + f"{prefix}_B12_20m.npy"

    vv = folder + f"{prefix}_VV.npy"
    vh = folder + f"{prefix}_VH.npy"

    label_area = folder + f"{prefix}_label_area.npy"
    label_volume = folder + f"{prefix}_label_volume.npy"
    label_people = folder + f"{prefix}_label_people.npy"

    rgbn = preprocess_optical(
        np.stack(
            [
                np.load(b02),
                np.load(b03),
                np.load(b04),
                np.load(b08),
            ],
            axis=3,
        )[:, :, :, :, 0],
        target_low=low,
        target_high=high,
        cutoff_high=optical_top,
    )

    shuffle_mask = np.random.permutation(rgbn.shape[0])

    np.save(outdir + f"{prefix}_RGBN.npy", rgbn[shuffle_mask])

    reswir = preprocess_optical(
        np.stack(
            [
                np.load(b05),
                np.load(b06),
                np.load(b07),
                np.load(b11),
                np.load(b12),
            ],
            axis=3,
        )[:, :, :, :, 0],
        target_low=low,
        target_high=high,
        cutoff_high=optical_top,
    )

    np.save(outdir + f"{prefix}_RESWIR.npy", reswir[shuffle_mask])

    sar = preprocess_sar(
        np.stack(
            [
                np.load(vv),
                np.load(vh),
            ],
            axis=3,
        )[:, :, :, :, 0],
        target_low=low,
        target_high=high,
    )

    np.save(outdir + f"{prefix}_SAR.npy", sar[shuffle_mask])

    np.save(outdir + f"{prefix}_label_area", np.load(label_area)[shuffle_mask])
    np.save(outdir + f"{prefix}_label_volume", np.load(label_volume)[shuffle_mask])
    np.save(outdir + f"{prefix}_label_people", np.load(label_people)[shuffle_mask])


base_folder = "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/data/bornholm/patches/"
folder = base_folder + "raw/"
outdir = base_folder

preprocess("2020", folder, outdir)
preprocess("2021", folder, outdir)
exit()
# Merge seasons
rgbn_merge = np.concatenate(
    [
        np.load(outdir + "2020_RGBN.npy"),
        np.load(outdir + "2021_RGBN.npy"),
    ],
)

shuffle_mask = np.random.permutation(rgbn_merge.shape[0])

np.save(outdir + "RGBN.npy", rgbn_merge[shuffle_mask])
np.save(
    outdir + "RESWIR.npy",
    np.concatenate(
        [
            np.load(outdir + "2020_RESWIR.npy"),
            np.load(outdir + "2021_RESWIR.npy"),
        ],
    )[shuffle_mask],
)
np.save(
    outdir + "SAR.npy",
    np.concatenate(
        [
            np.load(outdir + "2020_SAR.npy"),
            np.load(outdir + "2021_SAR.npy"),
        ],
    )[shuffle_mask],
)

np.save(
    outdir + "label_area.npy",
    np.concatenate(
        [
            np.load(outdir + "2020_label_area.npy"),
            np.load(outdir + "2021_label_area.npy"),
        ],
    )[shuffle_mask],
)

np.save(
    outdir + "label_volume.npy",
    np.concatenate(
        [
            np.load(outdir + "2020_label_volume.npy"),
            np.load(outdir + "2021_label_volume.npy"),
        ],
    )[shuffle_mask],
)

np.save(
    outdir + "label_people.npy",
    np.concatenate(
        [
            np.load(outdir + "2020_label_people.npy"),
            np.load(outdir + "2021_label_people.npy"),
        ],
    )[shuffle_mask],
)
