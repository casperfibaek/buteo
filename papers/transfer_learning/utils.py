import numpy as np


# https://stackoverflow.com/questions/30031828/multiply-numpy-ndarray-with-1d-array-along-a-given-axis
def mult_along_axis(A, B, axis):

    # ensure we're working with Numpy arrays
    A = np.array(A)
    B = np.array(B)

    # shape check
    if axis >= A.ndim:
        raise Exception(axis, A.ndim)
    if A.shape[axis] != B.size:
        raise ValueError(
            "Length of 'A' along the given axis must be the same as B.size"
        )

    # np.broadcast_to puts the new axis as the last axis, so
    # we swap the given axis with the last one, to determine the
    # corresponding array shape. np.swapaxes only returns a view
    # of the supplied array, so no data is copied unneccessarily.
    shape = np.swapaxes(A, A.ndim - 1, axis).shape

    # Broadcast to an array with the shape as above. Again,
    # no data is copied, we only get a new look at the existing data.
    B_brc = np.broadcast_to(B, shape)

    # Swap back the axes. As before, this only changes our "point of view".
    B_brc = np.swapaxes(B_brc, A.ndim - 1, axis)

    return A * B_brc


def random_scale_noise(arr, std=0.01):
    tile_scale = np.random.normal(1, std, (arr.shape[0]))
    scaled = mult_along_axis(arr, tile_scale, 0)
    noise = scaled * np.random.normal(1, 0.001, arr.shape)
    return noise.astype(arr.dtype)


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
