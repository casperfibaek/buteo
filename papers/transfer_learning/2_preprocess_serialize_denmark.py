import numpy as np

from utils import (
    preprocess_optical,
    preprocess_sar,
    random_scale_noise,
)


def get_image_paths(folder, test_or_train="train"):
    return [
        [
            [
                "B02",
                folder + f"2020_B02_10m_{test_or_train}.npy",
                folder + f"2021_B02_10m_{test_or_train}.npy",
            ],
            [
                "B03",
                folder + f"2020_B03_10m_{test_or_train}.npy",
                folder + f"2021_B03_10m_{test_or_train}.npy",
            ],
            [
                "B04",
                folder + f"2020_B04_10m_{test_or_train}.npy",
                folder + f"2021_B04_10m_{test_or_train}.npy",
            ],
            [
                "B08",
                folder + f"2020_B08_10m_{test_or_train}.npy",
                folder + f"2021_B08_10m_{test_or_train}.npy",
            ],
        ],
        [
            [
                "B11",
                folder + f"2020_B11_20m_{test_or_train}.npy",
                folder + f"2021_B11_20m_{test_or_train}.npy",
            ],
            [
                "B12",
                folder + f"2020_B12_20m_{test_or_train}.npy",
                folder + f"2021_B12_20m_{test_or_train}.npy",
            ],
        ],
        [
            [
                "VH",
                folder + f"2020_VH_{test_or_train}.npy",
                folder + f"2021_VH_{test_or_train}.npy",
            ],
            [
                "VV",
                folder + f"2020_VV_{test_or_train}.npy",
                folder + f"2021_VV_{test_or_train}.npy",
            ],
        ],
        [
            ["AREA", folder + f"area_{test_or_train}.npy"],
            ["VOLUME", folder + f"volume_{test_or_train}.npy"],
            ["PEOPLE", folder + f"people_{test_or_train}.npy"],
        ],
    ]


def merge_and_preprocess(folder, test_or_train="train"):
    images = get_image_paths(folder, test_or_train)

    images_10m = preprocess_optical(
        np.stack(
            [
                np.concatenate(
                    [np.load(images[0][0][1]), np.load(images[0][0][2])]
                ),  # B02
                np.concatenate(
                    [np.load(images[0][1][1]), np.load(images[0][1][2])]
                ),  # B03
                np.concatenate(
                    [np.load(images[0][2][1]), np.load(images[0][2][2])]
                ),  # B04
                np.concatenate(
                    [np.load(images[0][3][1]), np.load(images[0][3][2])]
                ),  # B08
            ],
            axis=3,
        )[:, :, :, :, 0]
    )

    sample_arr = [True, False]
    shuffle_mask = np.random.permutation(images_10m.shape[0])
    flip = np.random.choice(sample_arr, size=images_10m.shape[0])
    norm = ~flip

    np.save(
        folder + f"RGBN_{test_or_train}.npy",
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
                np.concatenate(
                    [np.load(images[1][0][1]), np.load(images[1][0][2])]
                ),  # B11
                np.concatenate(
                    [np.load(images[1][1][1]), np.load(images[1][1][2])]
                ),  # B12
            ],
            axis=3,
        )[:, :, :, :, 0]
    )

    np.save(
        folder + f"SWIR_{test_or_train}.npy",
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
                np.concatenate(
                    [np.load(images[2][0][1]), np.load(images[2][0][2])]
                ),  # VH
                np.concatenate(
                    [np.load(images[2][1][1]), np.load(images[2][1][2])]
                ),  # VV
            ],
            axis=3,
        )[:, :, :, :, 0]
    )

    np.save(
        folder + f"SAR_{test_or_train}.npy",
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
            np.concatenate(
                [np.load(images[3][0][1]), np.load(images[3][0][1])]
            ),  # Area
        ],
        axis=3,
    )[:, :, :, :, 0]

    np.save(
        folder + f"LABEL_AREA_{test_or_train}.npy",
        np.concatenate(
            [
                np.rot90(images_area[flip], k=2, axes=(1, 2)),
                images_area[norm],
            ]
        )[shuffle_mask],
    )

    # Free memory
    images_area = None

    images_volume = np.stack(
        [
            np.concatenate(
                [np.load(images[3][1][1]), np.load(images[3][1][1])]
            ),  # Area
        ],
        axis=3,
    )[:, :, :, :, 0]

    np.save(
        folder + f"LABEL_VOLUME_{test_or_train}.npy",
        np.concatenate(
            [
                np.rot90(images_volume[flip], k=2, axes=(1, 2)),
                images_volume[norm],
            ]
        )[shuffle_mask],
    )

    # Free memory
    images_volume = None

    images_people = np.stack(
        [
            np.concatenate(
                [np.load(images[3][2][1]), np.load(images[3][2][1])]
            ),  # Area
        ],
        axis=3,
    )[:, :, :, :, 0]

    np.save(
        folder + f"LABEL_PEOPLE_{test_or_train}.npy",
        np.concatenate(
            [
                np.rot90(images_people[flip], k=2, axes=(1, 2)),
                images_people[norm],
            ]
        )[shuffle_mask],
    )

    # Free memory
    images_people = None


folder = "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/analysis/denmark/"
merge_and_preprocess(folder, "train")
merge_and_preprocess(folder, "test")
