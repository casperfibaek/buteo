"""
This module contains function to select features from a dataset for machine learning.
"""

# Standard library
import sys; sys.path.append("../../")
from typing import Optional, Union, Tuple

# External
import numpy as np


def train_val_split(
    X: np.ndarray,
    y: np.ndarray,
    val_size: float = 0.2,
    random_state: Optional[Union[float, int]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split a dataset into training and validation sets. Always shuffles.
    
    Args:
        X (np.ndarray): The input data.
        y (np.ndarray): The target data.
    
    Keyword Args:
        val_size (float=0.2): The size of the validation set as a fraction of the total dataset.
        random_state (float=None): The random seed to use for the split.

    Returns:
        x_train, x_val, y_train, y_val (np.ndarray): The training and validation sets in a tuple.
    """
    assert isinstance(X, np.ndarray), "X should be a numpy array."
    assert isinstance(y, np.ndarray), "y should be a numpy array."
    assert X.shape[0] == y.shape[0], "X and y should be the same axis=0 size."
    assert 0 < val_size < 1, "val_size should be between 0 and 1."

    if random_state is not None:
        np.random.seed(random_state)

    sample_indices = np.random.permutation(X.shape[0])

    split_idx = int(X.shape[0] * (1 - val_size))

    X_train = X[sample_indices[:split_idx]]
    X_test = X[sample_indices[split_idx:]]
    y_train = y[sample_indices[:split_idx]]
    y_test = y[sample_indices[split_idx:]]

    return X_train, X_test, y_train, y_test


def train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    val_size: float = 0.1,
    test_size: float = 0.2,
    random_state: Optional[Union[float, int]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split a dataset into training, validation, and test sets. Always shuffles.

    Args:
        X (np.ndarray): The input data.
        y (np.ndarray): The target data.

    Keyword Args:
        val_size (float=0.1): The size of the validation set as a fraction of the total dataset.
        test_size (float=0.2): The size of the test set as a fraction of the total dataset.
        random_state (float=None): The random seed to use for the split.
    
    Returns:
        x_train, x_val, x_test, y_train, y_val, y_test (np.ndarray): The training, validation, and test sets in a tuple.
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


def stratified_sampling(
    X: np.ndarray,
    y: np.ndarray,
    regression: bool = False,
    samples_per_class: bool = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stratified sampling of a dataset. Can be used for regression or classification.
    
    Args:
        X (np.ndarray): The input data.
        y (np.ndarray): The target data.
    
    Keyword Args:
        regression (bool=False): Whether the dataset is for regression or classification.
        samples_per_class (int=None): The number of samples to take per class. If None,
            takes the same number of samples per class as the smallest class.

    Returns:
        X, y (np.ndarray): The stratified dataset.
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
