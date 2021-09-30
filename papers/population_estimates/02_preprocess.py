import sys, numpy as np

sys.path.append("../../")
from buteo.machine_learning.ml_utils import (
    preprocess_optical,
    preprocess_sar,
)


def preprocess(
    folder,
    outdir,
    low=0,
    high=1,
    optical_top=8000,
    use_noise=False,
    scale_noise=0.05,
    band_noise=0.01,
    pixel_noise=0.005,
):
    b02 = folder + "B02_10m.npy"
    b03 = folder + "B03_10m.npy"
    b04 = folder + "B04_10m.npy"
    b08 = folder + "B08_10m.npy"

    b05 = folder + "B05_20m.npy"
    b06 = folder + "B06_20m.npy"
    b07 = folder + "B07_20m.npy"
    b11 = folder + "B11_20m.npy"
    b12 = folder + "B12_20m.npy"

    vv = folder + "VV_10m.npy"
    vh = folder + "VH_10m.npy"

    label_area = folder + "label_area_10m.npy"

    area = np.load(label_area)
    shuffle_mask = np.random.permutation(area.shape[0])

    np.save(outdir + "label_area", area[shuffle_mask])

    if use_noise:
        noise_scale = np.random.normal(1.0, scale_noise, (area.shape[0], 1, 1, 1))

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

    if use_noise:
        noise_band = np.random.normal(
            1.0, band_noise, (rgbn.shape[0], 1, 1, rgbn.shape[3])
        )
        noise_pixel = np.random.normal(1.0, pixel_noise, rgbn.shape)
        rgbn = rgbn * (noise_scale * noise_band * noise_pixel).astype("float32")

    np.save(outdir + "RGBN.npy", rgbn[shuffle_mask])

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

    if use_noise:
        noise_band = np.random.normal(
            1.0, band_noise, (reswir.shape[0], 1, 1, reswir.shape[3])
        )
        noise_pixel = np.random.normal(1.0, pixel_noise, reswir.shape)
        reswir = reswir * (noise_scale * noise_band * noise_pixel).astype("float32")

    np.save(outdir + "RESWIR.npy", reswir[shuffle_mask])

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

    if use_noise:
        noise_band = np.random.normal(
            1.0, band_noise, (sar.shape[0], 1, 1, sar.shape[3])
        )
        noise_pixel = np.random.normal(1.0, pixel_noise, sar.shape)
        sar = sar * (noise_scale * noise_band * noise_pixel).astype("float32")

    np.save(outdir + "SAR.npy", sar[shuffle_mask])


base_folder = (
    # "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/data/uganda_kampala/patches/"
    # "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/data/tanzania_dar/patches/"
    # "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/data/tanzania_kilimanjaro/patches/"
    # "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/data/ghana/patches/"
    "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/data/ghana/vector/grid_cells_student2/patches/merged/"
)
folder = base_folder + "raw/"
outdir = base_folder

preprocess(folder, outdir, use_noise=True)
