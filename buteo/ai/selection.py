"""This module contains function to select features from a dataset for machine learning."""

# Standard library
import sys; sys.path.append("../../")
from typing import Optional, Union, Tuple

# External
import numpy as np


def split_train_val(
    X: np.ndarray,
    y: np.ndarray,
    val_size: float = 0.2,
    random_state: Optional[Union[float, int]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split a dataset into training and validation sets using random shuffling.

    Parameters
    ----------
    X : np.ndarray
        The input data.

    y : np.ndarray
        The target data.

    val_size : float, optional
        The size of the validation set as a fraction of the total dataset. Default: 0.2.

    random_state : Union[float, int], optional
        The random seed to use for the split. Default is None.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        A tuple containing the training and validation sets for input data and target data,
        respectively, in the following order: x_train, x_val, y_train, y_val.

    Raises
    ------
    AssertionError
        If X and y are not numpy arrays, X and y do not have the same number of rows, or
        val_size is not between 0 and 1.

    Notes
    -----
    The function always shuffles the data before splitting.
    """
    assert isinstance(X, np.ndarray), "X should be a numpy array."
    assert isinstance(y, np.ndarray), "y should be a numpy array."
    assert X.shape[0] == y.shape[0], "X and y should be the same axis=0 size."
    assert 0 < val_size < 1, "val_size should be between 0 and 1."

    if random_state is not None:
        np.random.seed(random_state)

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    split_idx = int(X.shape[0] * (1 - val_size))

    X_train = X[indices[:split_idx]]
    X_test = X[indices[split_idx:]]
    y_train = y[indices[:split_idx]]
    y_test = y[indices[split_idx:]]

    return X_train, X_test, y_train, y_test


def split_train_val_test(
    X: np.ndarray,
    y: np.ndarray,
    val_size: float = 0.1,
    test_size: float = 0.2,
    random_state: Optional[Union[float, int]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split a dataset into training, validation, and test sets using a specified random seed.

    Parameters
    ----------
    X : np.ndarray
        The input data to be split.

    y : np.ndarray
        The target data to be split.

    val_size : float, optional
        The proportion of the data to use for validation, default: 0.1.

    test_size : float, optional
        The proportion of the data to use for testing, default: 0.2.

    random_state : float or int, optional
        Seed for the random number generator used for shuffling, default: None.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        A tuple containing the following arrays:
        - x_train: training input data
        - x_val: validation input data
        - x_test: testing input data
        - y_train: training target data
        - y_val: validation target data
        - y_test: testing target data

    Raises
    ------
    AssertionError
        If X and y are not numpy arrays with the same shape[0], or if val_size or test_size are not between 0 and 1.

    Notes
    -----
    The function always shuffles the data before splitting.
    """
    assert isinstance(X, np.ndarray), "X should be a numpy array."
    assert isinstance(y, np.ndarray), "y should be a numpy array."
    assert X.shape[0] == y.shape[0], "X and y should be the same axis=0 size."
    assert 0 < val_size < 1, "val_size should be between 0 and 1."
    assert 0 < test_size < 1, "test_size should be between 0 and 1."

    if random_state is not None:
        np.random.seed(random_state)

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    test_split_idx = int(X.shape[0] * (1 - test_size))
    val_split_idx = int((1 - val_size) * test_split_idx)

    X_train = X[indices[:val_split_idx]]
    X_val = X[indices[val_split_idx:test_split_idx]]
    X_test = X[indices[test_split_idx:]]
    y_train = y[indices[:val_split_idx]]
    y_val = y[indices[val_split_idx:test_split_idx]]
    y_test = y[indices[test_split_idx:]]

    return X_train, X_val, X_test, y_train, y_val, y_test


def sampling_stratified(
    X: np.ndarray,
    y: np.ndarray,
    regression: bool = False,
    samples_per_class: bool = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Stratified sampling of a dataset.

    This function can be used for both regression and classification problems.

    Parameters
    ----------
    X : np.ndarray
        The input data.

    y : np.ndarray
        The target data.

    regression : bool, optional
        Whether the dataset is for regression or classification.
        Default: False.

    samples_per_class : int, optional
        The number of samples to take per class.
        If None, takes the same number of samples per class as the smallest class.
        Default: None.

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        (X_stratified, y_stratified) : The stratified input and target data.
    """
    assert isinstance(X, np.ndarray), "X should be a numpy array."
    assert isinstance(y, np.ndarray), "y should be a numpy array."
    assert X.shape[0] == y.shape[0], "X and y should be the same axis=0 size."
    assert samples_per_class is None or samples_per_class > 0, "samples_per_class should be greater than 0."

    y_classes = y

    if regression:
        assert y.dtype in [np.float64, np.float32, np.float16], "y should be a float array for regression."

        y_classes = np.digitize(y, np.percentile(y, [10, 20, 30, 40, 50, 60, 70, 80, 90]))

    unique_classes, counts = np.unique(y_classes, return_counts=True)

    if samples_per_class is None:
        samples_per_class = counts.min()

    stratified_indices = np.array([], dtype=int)

    for class_ in unique_classes:
        class_indices = np.where(y_classes == class_)[0]
        selected_indices = np.random.choice(class_indices, samples_per_class, replace=False)
        stratified_indices = np.concatenate((stratified_indices, selected_indices))

    return X[stratified_indices], y[stratified_indices]


def sampling_random(
    X: np.ndarray,
    y: np.ndarray,
    samples: Union[int, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Random sampling of a dataset.

    Parameters
    ----------
    X : np.ndarray
        The input data.

    y : np.ndarray
        The target data.

    samples : int or float
        The number of samples to take.
        If int, the number of samples to take.
        If float, the proportion of samples to take.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (X_random, y_random) : The random input and target data.
    """
    assert isinstance(X, np.ndarray), "X should be a numpy array."
    assert isinstance(y, np.ndarray), "y should be a numpy array."
    assert X.shape[0] == y.shape[0], "X and y should be the same axis=0 size."
    assert samples > 0, "samples should be greater than 0."
    assert isinstance(samples, (int, float)), "samples should be an int or float."

    if isinstance(samples, float) and samples <= 1:
        samples = int(X.shape[0] * samples)

    random_indices = np.random.choice(np.arange(X.shape[0]), samples, replace=False)

    return X[random_indices], y[random_indices]


# TODO: add a function to split a dataset into k folds
# TODO: add a function to split a dataset into k folds stratified by class
