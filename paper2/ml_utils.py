import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np


def y_class(s):
    if s == "fid":
        return 0
    if s == "area" or s == "b_area":
        return 1
    if s == "volume" or s == "b_volume":
        return 2
    if s == "ppl" or s == "ppl_ha":
        return 3

def sar_class(s):
    if s == "asc" or s == "ascending":
        return 0
    if s == "desc" or s == "descending":
        return 1
    if s == "joined" or s == "max":
        return 2

def count_freq(arr):
    yy = np.bincount(arr)
    ii = np.nonzero(yy)[0]
    return np.vstack((ii, yy[ii])).T

# https://stackoverflow.com/a/44233061/8564588
def minority_class_mask(arr, minority):
    return np.hstack([
        np.random.choice(np.where(arr == l)[0], minority, replace=False) for l in np.unique(arr)
    ])

def get_shape(numpy_arr):
    return (numpy_arr.shape[1], numpy_arr.shape[2], numpy_arr.shape[3])

def add_rotations(X, k=4):
    if k == 1:
        return X
    elif k == 2:
        return np.concatenate([
            X,
            np.rot90(X, k=2, axes=(1, 2)),
        ])
    elif k == 3:
        return np.concatenate([
            X,
            np.rot90(X, k=1, axes=(1, 2)),
            np.rot90(X, k=2, axes=(1, 2)),
        ])
    else:
        return np.concatenate([
            np.rot90(X, k=1, axes=(1, 2)),
            np.rot90(X, k=2, axes=(1, 2)),
            np.rot90(X, k=3, axes=(1, 2)),
            X,
        ])


def add_noise(X, amount=0.01):
    return X * np.random.normal(1, amount, X.shape)


def scale_to_01(arr):
      return (arr - arr.min()) / (arr.max() - arr.min())


def train_split_mask(y, split=0.3, stratified=True):
    if stratified is True:
        strats = np.digitize(y, np.percentile(y, [10, 20, 30, 40, 50, 60, 70, 80, 90]))
        indices = np.arange(0, len(strats), 1)

        X_train, X_test, y_train, y_test = train_test_split(indices, strats, stratify=strats, test_size=split)

        return (y_train, y_test)

    split_amount = int(round(len(y) * split))

    positive = np.full(len(y) - split_amount, True)
    negative = np.full(split_amount, False)
    merged = np.append(positive, negative)
    np.random.shuffle(merged)

    return (merged, ~merged)


def add_randomness(arr):
    flips = int(round(len(arr) / 4))
    return np.concatenate([
        arr[0:flips],
        np.rot90(arr[flips:2 * flips], k=1, axes=(1, 2)),
        np.rot90(arr[flips * 2: flips * 3], k=2, axes=(1, 2)),
        np.rot90(arr[flips * 3:], k=3, axes=(1, 2)),
    ])


def viz(X, y, model, target='area'):
    truth = y.astype("float32")
    predicted = model.predict(X).squeeze().astype("float32")

    if target == "area":
        labels = [*range(140, 5740, 140)]
    else:
        labels = [*range(500, 20500, 500)]

    truth_labels = np.digitize(truth, labels, right=True)
    predicted_labels = np.digitize(predicted, labels, right=True)
    labels_unique = np.unique(truth_labels)

    residuals = (truth - predicted).astype('float32')
    residuals = residuals / 140 if target == 'area' else residuals / 700

    fig1, ax = plt.subplots()
    ax.set_title('violin area')

    per_class = []
    for cl in labels_unique:
        per_class.append(residuals[truth_labels == cl])

    ax.violinplot(per_class, showextrema=False, showmedians=True)

    plt.show()


def histogram_selection(y, zero_class=True, resolution=5, outliers=True, allow_gap=0.2, whisk_range=1.5):

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
        classes = np.digitize(y, np.arange(0, cl_end + step_size, step_size), right=True)
    else:
        classes = np.digitize(y, np.arange(step_size, cl_end + step_size, step_size), right=True)
    
    frequency = count_freq(classes)
    minority = frequency.min(axis=0)[1]
    max_samples = int(round(minority * (1 + allow_gap)))

    samples = []

    for hist_class in np.unique(classes):
        sample_count = len(classes[classes == hist_class])
        if sample_count > max_samples:
            samples.append(np.random.choice(indices[classes == hist_class], max_samples, replace=False))
        else:
            samples.append(np.random.choice(indices[classes == hist_class], sample_count, replace=False))

    return np.hstack(samples)
