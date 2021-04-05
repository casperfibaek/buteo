from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


def count_freq(arr: np.ndarray) -> np.ndarray:
    bins = np.bincount(arr)
    classes = np.nonzero(bins)[0]
    return np.vstack([classes, bins[classes]]).T


# Metrics for testing model accuracy
# def median_absolute_error(y_actual: float, y_pred: float) -> float:
#     return tfp.stats.percentile(tf.math.abs(y_actual - y_pred), 50.0)


# def median_absolute_percentage_error(y_actual: float, y_pred: float) -> float:
#     return tfp.stats.percentile(
#         tf.divide(
#             tf.abs(tf.subtract(y_actual, y_pred)), (y_actual + 1e-10)
#         ) * 100
#     , 50.0)


# Printing visuals
def pad(s, dl, dr):
    split = s.split('.')
    left = split[0]
    right = split[1]

    if len(left) < dl:
        left = ((dl - len(left)) * ' ') + left
    
    if len(right) < dr:
        right = right + ((dr - len(right)) * '0')
    
    return left + '.' + right


# https://stackoverflow.com/a/44233061/8564588
def minority_class_mask(arr, minority):
    return np.hstack(
        [
            np.random.choice(np.where(arr == l)[0], minority, replace=False)
            for l in np.unique(arr)
        ]
    )

def create_submask(arr, amount):
    z = np.zeros(len(arr) - int(amount), "bool")
    m = np.ones(amount, "bool")
    a = np.concatenate([z, m])
    np.random.shuffle(a)

    return a


def add_rotations(X, k=4, axes=(1, 2)):
    if k == 1:
        return X
    elif k == 2:
        return np.concatenate([
            X,
            np.rot90(X, k=2, axes=axes),
        ])
    elif k == 3:
        return np.concatenate(
            [
                X,
                np.rot90(X, k=1, axes=axes),
                np.rot90(X, k=2, axes=axes),
            ]
        )
    else:
        return np.concatenate(
            [
                X,
                np.rot90(X, k=1, axes=axes),
                np.rot90(X, k=2, axes=axes),
                np.rot90(X, k=3, axes=axes),
            ]
        )


def add_noise(X, amount=0.01):
    return X * np.random.normal(1, amount, X.shape)


def add_fixed_noise(X, center=0, amount=0.05):
    return X + np.random.normal(center, amount, X.shape)


def scale_to_01(X):
    return (X - X.min()) / (X.max() - X.min())


# stratify a regression split
def train_split_mask_regression(y, split=0.3, stratified=True):
    if stratified is True:
        strats = np.digitize(y, np.percentile(y, [10, 20, 30, 40, 50, 60, 70, 80, 90]))
        indices = np.arange(0, len(strats), 1)

        _X_train, _X_test, y_train, y_test = train_test_split(
            indices, strats, stratify=strats, test_size=split
        )

        return (y_train, y_test)

    split_amount = int(round(len(y) * split))

    positive = np.full(len(y) - split_amount, True)
    negative = np.full(split_amount, False)
    merged = np.append(positive, negative)
    np.random.shuffle(merged)

    return (merged, ~merged)


def add_randomness(arr):
    flips = int(round(len(arr) / 4))
    return np.concatenate(
        [
            arr[0:flips],
            np.rot90(arr[flips : 2 * flips], k=1, axes=(1, 2)),
            np.rot90(arr[flips * 2 : flips * 3], k=2, axes=(1, 2)),
            np.rot90(arr[flips * 3 :], k=3, axes=(1, 2)),
        ]
    )


def histogram_selection(
    y, zero_class=True, resolution=5, outliers=True, allow_gap=0.2, whisk_range=1.5
):

    indices = np.arange(0, len(y), 1)

    if outliers is True:
        if zero_class is True:
            q1 = np.quantile(y[y > 0], 0.25)
            q3 = np.quantile(y[y > 0], 0.75)
        else:
            q1 = np.quantile(y, 0.25)
            q3 = np.quantile(y, 0.75)
        iqr = q3 - q1
        whisk = iqr * whisk_range

        cl_start = q1 - whisk
        cl_end = q3 + whisk
        step_size = (cl_end - cl_start) / resolution
    else:
        step_size = (y.max() - y.min()) / resolution
        cl_start = step_size
        cl_end = y.max()

    if zero_class is True:
        classes = np.digitize(
            y, np.arange(0, cl_end + step_size, step_size), right=True
        )
    else:
        classes = np.digitize(
            y, np.arange(step_size, cl_end + step_size, step_size), right=True
        )

    frequency = count_freq(classes)
    minority = frequency.min(axis=0)[1]
    max_samples = int(round(minority * (1 + allow_gap)))

    samples = []

    for hist_class in np.unique(classes):
        sample_count = len(classes[classes == hist_class])
        if sample_count > max_samples:
            samples.append(
                np.random.choice(
                    indices[classes == hist_class], max_samples, replace=False
                )
            )
        else:
            samples.append(
                np.random.choice(
                    indices[classes == hist_class], sample_count, replace=False
                )
            )

    return np.hstack(samples)


def test_correlation(df: pd.DataFrame, cutoff: float=0.75) -> np.ndarray:
    abs_corr = df.corr().abs()
    triangle = np.array(np.triu(np.ones(abs_corr.shape), k=1), dtype=np.bool)
    upper_tri = abs_corr.where(triangle)
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > cutoff)]
    return to_drop
