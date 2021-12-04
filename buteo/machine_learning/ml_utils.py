import math
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Activation
from tensorflow.keras.utils import get_custom_objects


def create_step_decay(learning_rate=0.001, drop_rate=0.5, epochs_per_drop=10):
    def step_decay(epoch):
        initial_lrate = learning_rate
        drop = drop_rate
        epochs_drop = epochs_per_drop
        lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
        return lrate

    return step_decay


def count_freq(arr: np.ndarray) -> np.ndarray:
    bins = np.bincount(arr)
    classes = np.nonzero(bins)[0]
    return np.vstack([classes, bins[classes]]).T


# Printing visuals
def pad(s, dl, dr):
    split = s.split(".")
    left = split[0]
    right = split[1]

    if len(left) < dl:
        left = ((dl - len(left)) * " ") + left

    if len(right) < dr:
        right = right + ((dr - len(right)) * "0")

    return left + "." + right


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
        return np.concatenate(
            [
                X,
                np.rot90(X, k=2, axes=axes),
            ]
        )
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


def test_correlation(df: pd.DataFrame, cutoff: float = 0.75) -> np.ndarray:
    abs_corr = df.corr().abs()
    triangle = np.array(np.triu(np.ones(abs_corr.shape), k=1), dtype=np.bool)
    upper_tri = abs_corr.where(triangle)
    to_drop = [
        column for column in upper_tri.columns if any(upper_tri[column] > cutoff)
    ]
    return to_drop


def mish(inputs):
    return inputs * tf.math.tanh(tf.math.softplus(inputs))


class Mish(Activation):
    """
    Mish Activation Function.
    .. math::
        mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^{x}))
    Shape:
        - Input: Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
        - Output: Same shape as the input.
    Examples:
        >>> X = Activation('Mish', name="conv1_act")(X_input)
    """

    def __init__(self, activation, **kwargs):
        super(Mish, self).__init__(activation, **kwargs)
        self.__name__ = "Mish"


def load_mish():
    get_custom_objects().update({"Mish": Mish(mish)})


def mad_standard(arr):
    median = np.median(arr)
    absdev = np.abs(np.subtract(arr, median))
    madstd = np.median(absdev) * 1.4826

    return ((arr - median) / madstd).astype("float32")


def iqr_scale(arr):
    q1, median, q3 = np.percentile(arr, [25, 50, 75])

    return ((arr - median) / (q3 - q1)).astype("float32")


def mean_standard(arr):
    mean = np.mean(arr)
    std = np.std(arr)

    return ((arr - mean) / std).astype("float32")


def min_max(arr):
    return (arr / arr.max()).astype("float32")


def trunc_scale(arr, val):
    return (np.where(arr > val, val, arr) / val).astype("float32")


def scale_percentile(arr, percentile=98):
    percentile = np.percentile(arr, percentile)

    truncated = np.where(arr > percentile, percentile, arr)

    return (truncated / truncated.max()).astype("float32")


# np.set_printoptions(suppress=True)
def preprocess_optical(
    arr, cutoff_low=0, cutoff_high=8000, target_low=0, target_high=1
):
    clipped = np.where(
        arr > cutoff_high, cutoff_high, np.where(arr < cutoff_low, cutoff_low, arr)
    )
    val_a = (target_high - target_low) / (cutoff_high - cutoff_low)
    val_b = target_high - (val_a * cutoff_high)

    return ((val_a * clipped) + val_b).astype("float32")


def scale_to_range(arr, min, max):
    return MinMaxScaler(feature_range=(min, max)).fit_transform(arr)


def preprocess_coh(arr, target_low=0, target_high=1):
    return preprocess_optical(arr, 0, 1, target_low, target_high)


def get_offsets(size):
    high_mid = size // 2
    high_low = high_mid // 2
    high_high = high_mid + high_low

    return [
        [0, 0],
        [0, high_low],
        [0, high_mid],
        [0, high_high],
        [high_low, high_low],
        [high_mid, high_mid],
        [high_high, high_high],
        [high_low, 0],
        [high_mid, 0],
        [high_high, 0],
    ]


# Converts to decibel, thresholds between -30db and 10db and scales to -1 to 1
def preprocess_sar(
    arr, convert_db=True, cutoff_low=-30, cutoff_high=15, target_low=0, target_high=1
):
    if convert_db:
        with np.errstate(divide="ignore", invalid="ignore"):
            arr_adj = 10 * np.where(arr != 0, np.log10(arr), cutoff_low)
    else:
        arr_adj = arr

    thresholded = np.where(
        arr_adj > cutoff_high,
        cutoff_high,
        np.where(arr_adj < cutoff_low, cutoff_low, arr_adj),
    )

    scaled = preprocess_optical(
        thresholded,
        cutoff_low=cutoff_low,
        cutoff_high=cutoff_high,
        target_low=target_low,
        target_high=target_high,
    )

    return scaled


def mse(y_pred, y_true):
    y_pred = tf.cast(y_pred, tf.float64)
    y_true = tf.cast(y_true, tf.float64)

    return tf.math.reduce_mean(tf.math.squared_difference(y_pred, y_true))


def mse_biased(y_pred, y_true):
    y_pred = tf.cast(y_pred, tf.float32)
    y_true = tf.cast(y_true, tf.float32)
    bias = tf.cast(1.50, tf.float32)

    greater_one = tf.cast(tf.math.greater(y_true, y_pred), tf.float32)
    greater_two = tf.cast(tf.math.greater_equal(y_pred, y_true), tf.float32)

    # scale = (greater_one * bias) + greater_two
    scale = (greater_two * bias) + greater_one

    sqr_diff = ((y_pred - y_true) * scale) ** 2

    return tf.math.reduce_mean(sqr_diff)


def log_cosh(y_pred, y_true):
    y_pred = tf.cast(y_pred, tf.float64)
    y_true = tf.cast(y_true, tf.float64)

    y_pred = tf.nn.relu(y_pred)

    return tf.losses.log_cosh(y_true, y_pred)


def tpe(y_true, y_pred):
    y_pred = tf.cast(y_pred, tf.float64)
    y_true = tf.cast(y_true, tf.float64)

    espilon = 1e-7
    pred_sum = tf.math.reduce_sum(y_pred)
    true_sum = tf.math.reduce_sum(y_true)
    sdif = pred_sum - true_sum
    part = (sdif / (true_sum + espilon)) * 100
    clipped = tf.clip_by_value(part, -100.0, 100.0)

    return clipped


class SaveBestModel(tf.keras.callbacks.Callback):
    def __init__(self, save_best_metric="val_loss", this_max=False):
        self.save_best_metric = save_best_metric
        self.max = this_max
        if this_max:
            self.best = float("-inf")
        else:
            self.best = float("inf")

    def on_epoch_end(self, epoch, logs=None):
        metric_value = abs(logs[self.save_best_metric])
        if self.max:
            if metric_value > self.best:
                self.best = metric_value
                self.best_weights = self.model.get_weights()

        else:
            if metric_value < self.best:
                self.best = metric_value
                self.best_weights = self.model.get_weights()
