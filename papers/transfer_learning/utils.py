import numpy as np


def random_scale_noise(arr, std=0.01):
    return (
        (arr * np.random.normal(1, std)) * np.random.normal(1, std, arr.shape)
    ).astype(arr.dtype)


def rotate_shuffle(arrs=[]):
    sample_arr = [True, False]
    shuffle_mask = np.random.permutation(arrs[0].shape[0])
    flip = np.random.choice(sample_arr, size=arrs[0].shape[0])
    norm = ~flip

    rotated = []

    for idx in range(len(arrs)):
        rotated.append(
            np.concatenate(
                [
                    np.rot90(arrs[idx][flip], k=2, axes=(1, 2)),
                    arrs[idx][norm],
                ]
            )[shuffle_mask]
        )

    return rotated


def preprocess_optical(arr, cutoff=10000):
    return np.true_divide(np.where(arr > cutoff, cutoff, arr), cutoff).astype("float32")


def preprocess_sar(arr, cutoffs=[-30, 10]):
    with np.errstate(divide="ignore", invalid="ignore"):
        arr_db = 10 * np.where(arr != 0, np.log10(arr), 0)
    diff = cutoffs[1] - cutoffs[0]
    thresholded = np.where(
        arr_db > cutoffs[1],
        cutoffs[1],
        np.where(arr_db < cutoffs[0], cutoffs[0], arr_db),
    )

    return np.true_divide(np.add(thresholded, diff), diff)
