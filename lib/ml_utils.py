import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np

# Tensorflow
import tensorflow as tf
import tensorflow_probability as tfp

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

# Metrics for testing model accuracy
def median_absolute_error(y_actual, y_pred):
    return tfp.stats.percentile(tf.math.abs(y_actual - y_pred), 50.0)

def median_absolute_percentage_error(y_actual, y_pred):
    return tfp.stats.percentile(
        tf.divide(
            tf.abs(tf.subtract(y_actual, y_pred)), (y_actual + 1e-10)
        ) * 100
    , 50.0)

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

def get_shape(numpy_arr):
    return (numpy_arr.shape[1], numpy_arr.shape[2], numpy_arr.shape[3])


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


def train_split_mask(y, split=0.3, stratified=True):
    if stratified is True:
        strats = np.digitize(y, np.percentile(y, [10, 20, 30, 40, 50, 60, 70, 80, 90]))
        indices = np.arange(0, len(strats), 1)

        X_train, X_test, y_train, y_test = train_test_split(
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


def viz(X, y, model, target="area"):
    truth = y.astype("float32")
    predicted = model.predict(X).squeeze().astype("float32")

    if target == "area":
        labels = [*range(140, 5740, 140)]
    else:
        labels = [*range(500, 20500, 500)]

    truth_labels = np.digitize(truth, labels, right=True)
    predicted_labels = np.digitize(predicted, labels, right=True)
    labels_unique = np.unique(truth_labels)

    residuals = (truth - predicted).astype("float32")
    residuals = residuals / 140 if target == "area" else residuals / 700

    fig1, ax = plt.subplots()
    ax.set_title("violin area")

    per_class = []
    for cl in labels_unique:
        per_class.append(residuals[truth_labels == cl])

    ax.violinplot(per_class, showextrema=False, showmedians=True)

    plt.show()


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


def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of deleted columns
    sets = {}
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (corr_matrix.iloc[i, j] >= threshold) and (
                corr_matrix.columns[j] not in col_corr
            ):
                colname = corr_matrix.columns[i]  # getting the name of column
                target_name = corr_matrix.columns[j]
                col_corr.add(f"{colname}:{target_name}")
                # if colname in dataset.columns:
                #     del dataset[colname]  # deleting the column from the dataset

    return col_corr


def best_param_random(X, y, n_iter=50, cv=3):
    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    max_features = ["log2", "sqrt"]
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    min_samples_split = [2, 4, 6]
    min_samples_leaf = [1, 2, 4]
    bootstrap = [True, False]

    random_grid = {
        "n_estimators": n_estimators,
        "max_features": max_features,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "bootstrap": bootstrap,
    }

    random_forest_tuned = RandomForestClassifier()

    rf_random = RandomizedSearchCV(
        estimator=random_forest_tuned,
        param_distributions=random_grid,
        n_iter=n_iter,
        cv=cv,
        verbose=2,
        random_state=42,
        n_jobs=-1,
    )

    rf_random.fit(X, y)

    return rf_random.best_params_
