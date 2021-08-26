import sys, numpy as np

sys.path.append("../../")
from buteo.machine_learning.ml_utils import (
    preprocess_optical,
    preprocess_sar,
    preprocess_coh,
)

base_folder = "C:/Users/caspe/Desktop/paper_2_Structural_Volume/data/patches/128x128/"
folder = base_folder + "raw/"
outdir = base_folder
prefix = ""


def preprocess(prefix, low=0, high=1, optical_top=6000):
    b02 = folder + prefix + "B02.npy"
    b03 = folder + prefix + "B03.npy"
    b04 = folder + prefix + "B04.npy"
    b08 = folder + prefix + "B08.npy"

    b05 = folder + prefix + "B05.npy"
    b06 = folder + prefix + "B06.npy"
    b07 = folder + prefix + "B07.npy"
    # b8A = folder + prefix + "B8A.npy"
    b11 = folder + prefix + "B11.npy"
    b12 = folder + prefix + "B12.npy"

    coh_asc = folder + prefix + "COH_asc.npy"
    coh_desc = folder + prefix + "COH_desc.npy"

    vv_asc = folder + prefix + "VV_asc.npy"
    vv_desc = folder + prefix + "VV_desc.npy"

    vv_asc_v2 = folder + prefix + "VV_asc_v2.npy"
    vh_asc_v2 = folder + prefix + "VH_asc_v2.npy"

    label_area = folder + prefix + "label_area.npy"
    label_volume = folder + prefix + "label_volume.npy"
    label_people = folder + prefix + "label_people.npy"

    rgb = preprocess_optical(
        np.stack(
            [
                np.load(b02),
                np.load(b03),
                np.load(b04),
            ],
            axis=3,
        )[:, :, :, :, 0],
        target_low=low,
        target_high=high,
        cutoff_high=optical_top,
    )

    shuffle_mask = np.random.permutation(rgb.shape[0])

    np.save(outdir + prefix + f"RGB.npy", rgb[shuffle_mask])

    # Free memory
    rgb = None

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

    np.save(outdir + prefix + f"RGBN.npy", rgbn[shuffle_mask])

    rgbn = None

    rededge = preprocess_optical(
        np.stack(
            [
                np.load(b05),
                np.load(b06),
                np.load(b07),
            ],
            axis=3,
        )[:, :, :, :, 0],
        target_low=low,
        target_high=high,
        cutoff_high=optical_top,
    )

    np.save(outdir + prefix + f"REDEDGE.npy", rededge[shuffle_mask])

    # Free memory
    rededge = None

    swir = preprocess_optical(
        np.stack(
            [
                np.load(b11),
                np.load(b12),
            ],
            axis=3,
        )[:, :, :, :, 0],
        target_low=low,
        target_high=high,
        cutoff_high=optical_top,
    )

    np.save(outdir + prefix + f"SWIR.npy", swir[shuffle_mask])

    # Free memory
    swir = None

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

    np.save(outdir + prefix + f"RESWIR.npy", reswir[shuffle_mask])

    # Free memory
    reswir = None

    np.save(
        outdir + prefix + "coh_asc",
        preprocess_coh(np.load(coh_asc), target_low=low, target_high=high)[
            shuffle_mask
        ],
    )
    np.save(
        outdir + prefix + "coh_desc",
        preprocess_coh(np.load(coh_desc), target_low=low, target_high=high)[
            shuffle_mask
        ],
    )

    np.save(
        outdir + prefix + "vv_asc",
        preprocess_sar(np.load(vv_asc), target_low=low, target_high=high)[shuffle_mask],
    )
    np.save(
        outdir + prefix + "vv_desc",
        preprocess_sar(np.load(vv_desc), target_low=low, target_high=high)[
            shuffle_mask
        ],
    )

    np.save(
        outdir + prefix + "vv_asc_v2",
        preprocess_sar(np.load(vv_asc_v2), target_low=low, target_high=high)[
            shuffle_mask
        ],
    )
    np.save(
        outdir + prefix + "vh_asc_v2",
        preprocess_sar(np.load(vh_asc_v2), target_low=low, target_high=high)[
            shuffle_mask
        ],
    )

    np.save(outdir + prefix + "label_area", np.load(label_area)[shuffle_mask])
    np.save(outdir + prefix + "label_volume", np.load(label_volume)[shuffle_mask])
    np.save(outdir + prefix + "label_people", np.load(label_people)[shuffle_mask])


preprocess("")
preprocess("aarhus_")
preprocess("holsterbro_")
preprocess("samsoe_")
