import numpy as np

from utils import (
    preprocess_optical,
    preprocess_sar,
)


def get_image_paths(folder, test_or_train="train"):
    return [
        [
            [
                "B02",
                folder + f"B02.npy",
            ],
            [
                "B03",
                folder + f"B03.npy",
            ],
            [
                "B04",
                folder + f"B04.npy",
            ],
            [
                "B08",
                folder + f"B08.npy",
            ],
        ],
        [
            [
                "B11",
                folder + f"B11.npy",
            ],
            [
                "B12",
                folder + f"B12.npy",
            ],
        ],
        [
            [
                "VH",
                folder + f"VH.npy",
            ],
            [
                "VV",
                folder + f"VV.npy",
            ],
        ],
        [
            ["AREA", folder + f"AREA.npy"],
        ],
    ]


def merge_and_preprocess(folder):
    images = get_image_paths(folder)

    images_10m = preprocess_optical(
        np.stack(
            [
                np.load(images[0][0][1]),
                np.load(images[0][1][1]),
                np.load(images[0][2][1]),
                np.load(images[0][3][1]),
            ],
            axis=3,
        )[:, :, :, :, 0]
    )

    sample_arr = [True, False]
    shuffle_mask = np.random.permutation(images_10m.shape[0])
    flip = np.random.choice(sample_arr, size=images_10m.shape[0])
    norm = ~flip

    np.save(
        folder + f"RGBN.npy",
        np.concatenate(
            [
                np.rot90(images_10m[flip], k=2, axes=(1, 2)),
                images_10m[norm],
            ]
        )[shuffle_mask],
    )

    # Free memory
    images_10m = None

    images_20m = preprocess_optical(
        np.stack(
            [
                np.load(images[1][0][1]),
                np.load(images[1][1][1]),
            ],
            axis=3,
        )[:, :, :, :, 0]
    )

    np.save(
        folder + f"SWIR.npy",
        np.concatenate(
            [
                np.rot90(images_20m[flip], k=2, axes=(1, 2)),
                images_20m[norm],
            ]
        )[shuffle_mask],
    )

    # Free memory
    images_20m = None

    images_sar = preprocess_sar(
        np.stack(
            [
                np.load(images[2][0][1]),
                np.load(images[2][1][1]),
            ],
            axis=3,
        )[:, :, :, :, 0]
    )

    np.save(
        folder + f"SAR.npy",
        np.concatenate(
            [
                np.rot90(images_sar[flip], k=2, axes=(1, 2)),
                images_sar[norm],
            ]
        )[shuffle_mask],
    )

    # Free memory
    images_sar = None

    images_area = np.stack(
        [
            np.load(images[3][0][1]),
        ],
        axis=3,
    )[:, :, :, :, 0]

    np.save(
        folder + f"LABEL_AREA.npy",
        np.concatenate(
            [
                np.rot90(images_area[flip], k=2, axes=(1, 2)),
                images_area[norm],
            ]
        )[shuffle_mask],
    )

    # Free memory
    images_area = None


folder = "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/analysis/ghana/vector/grid_cells/patches/merged/"

merge_and_preprocess(folder)
