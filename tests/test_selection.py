""" Tests for ai/selection.py """


# Standard library
import sys; sys.path.append("../")

# External
import numpy as np
import pytest

# Internal
from buteo.ai.selection import (
    split_train_val,
    split_train_val_test,
    stratified_sampling,
)


def test_split_proportions():
    """ Test that the split proportions are correct. """
    X = np.random.random((100, 4))
    y = np.random.randint(0, 2, 100)

    X_train, X_val, y_train, y_val = split_train_val(X, y, val_size=0.25, random_state=42)

    assert X_train.shape[0] == 75
    assert X_val.shape[0] == 25
    assert y_train.shape[0] == 75
    assert y_val.shape[0] == 25

def test_split_proportions_with_different_val_size():
    """ Test that the split proportions are correct with different val_size. """
    X = np.random.random((100, 4))
    y = np.random.randint(0, 2, 100)

    X_train, X_val, y_train, y_val = split_train_val(X, y, val_size=0.1, random_state=42)

    assert X_train.shape[0] == 90
    assert X_val.shape[0] == 10
    assert y_train.shape[0] == 90
    assert y_val.shape[0] == 10

def test_random_state_reproducibility():
    """ Test that the split is reproducible with the same random_state. """
    X = np.random.random((100, 4))
    y = np.random.randint(0, 2, 100)

    X_train1, X_val1, y_train1, y_val1 = split_train_val(X, y, val_size=0.25, random_state=42)
    X_train2, X_val2, y_train2, y_val2 = split_train_val(X, y, val_size=0.25, random_state=42)

    assert np.array_equal(X_train1, X_train2)
    assert np.array_equal(X_val1, X_val2)
    assert np.array_equal(y_train1, y_train2)
    assert np.array_equal(y_val1, y_val2)

def test_input_type_validation():
    """ Test that the input types are validated. """
    X = np.random.random((100, 4)).tolist()
    y = np.random.randint(0, 2, 100).tolist()

    with pytest.raises(AssertionError):
        split_train_val(X, y, val_size=0.25, random_state=42)

def test_input_size_validation():
    """ Test that the input sizes are validated. """
    X = np.random.random((100, 4))
    y = np.random.randint(0, 2, 101)

    with pytest.raises(AssertionError):
        split_train_val(X, y, val_size=0.25, random_state=42)

def test_val_size_validation():
    """ Test that the val_size is validated. """
    X = np.random.random((100, 4))
    y = np.random.randint(0, 2, 100)

    with pytest.raises(AssertionError):
        split_train_val(X, y, val_size=1.5, random_state=42)

    with pytest.raises(AssertionError):
        split_train_val(X, y, val_size=-0.1, random_state=42)

def test_split_proportions_with_different_val_size_3DX():
    """ Test that the split proportions are correct with different val_size. """
    X = np.random.random((100, 32, 32, 3))
    y = np.random.randint(0, 2, 100)

    X_train, X_val, y_train, y_val = split_train_val(X, y, val_size=0.1, random_state=42)

    assert X_train.shape[0] == 90
    assert X_val.shape[0] == 10
    assert y_train.shape[0] == 90
    assert y_val.shape[0] == 10

def test_train_val_test_split_proportions():
    """ Test that the split proportions are correct. """
    X = np.random.random((100, 4))
    y = np.random.randint(0, 2, 100)

    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(X, y, val_size=0.1, test_size=0.2, random_state=42)

    assert X_train.shape[0] == 72
    assert X_val.shape[0] == 8
    assert X_test.shape[0] == 20
    assert y_train.shape[0] == 72
    assert y_val.shape[0] == 8
    assert y_test.shape[0] == 20

def test_train_val_test_split_proportions_with_different_sizes():
    """ Test that the split proportions are correct with different val_size and test_size. """
    X = np.random.random((100, 4))
    y = np.random.randint(0, 2, 100)

    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(X, y, val_size=0.15, test_size=0.25, random_state=42)

    assert X_train.shape[0] == 63
    assert X_val.shape[0] == 12
    assert X_test.shape[0] == 25
    assert y_train.shape[0] == 63
    assert y_val.shape[0] == 12
    assert y_test.shape[0] == 25

def test_train_val_test_split_random_state_reproducibility():
    """ Test that the split is reproducible with the same random_state. """	
    X = np.random.random((100, 4))
    y = np.random.randint(0, 2, 100)

    X_train1, X_val1, X_test1, y_train1, y_val1, y_test1 = split_train_val_test(X, y, val_size=0.1, test_size=0.2, random_state=42)
    X_train2, X_val2, X_test2, y_train2, y_val2, y_test2 = split_train_val_test(X, y, val_size=0.1, test_size=0.2, random_state=42)

    assert np.array_equal(X_train1, X_train2)
    assert np.array_equal(X_val1, X_val2)
    assert np.array_equal(X_test1, X_test2)
    assert np.array_equal(y_train1, y_train2)
    assert np.array_equal(y_val1, y_val2)
    assert np.array_equal(y_test1, y_test2)

def test_train_val_test_split_input_type_validation():
    """ Test that the input types are validated. """
    X = np.random.random((100, 4)).tolist()
    y = np.random.randint(0, 2, 100).tolist()

    with pytest.raises(AssertionError):
        split_train_val_test(X, y, val_size=0.1, test_size=0.2, random_state=42)

def test_train_val_test_split_input_size_validation():
    """ Test that the input sizes are validated. """
    X = np.random.random((100, 4))
    y = np.random.randint(0, 2, 101)

    with pytest.raises(AssertionError):
        split_train_val_test(X, y, val_size=0.1, test_size=0.2, random_state=42)

def test_stratified_sampling_classification():
    """ Test that the stratified sampling works for classification. """
    X = np.random.random((100, 4))
    y = np.random.randint(0, 3, 100)

    X_strat, y_strat = stratified_sampling(X, y, regression=False, samples_per_class=10)

    assert X_strat.shape == (30, 4)
    assert y_strat.shape == (30,)

    unique_classes, counts = np.unique(y_strat, return_counts=True)
    assert np.all(counts == 10)


def test_stratified_sampling_regression():
    """ Test that the stratified sampling works for regression. """
    X = np.random.random((100, 4))
    y = np.random.random(100)

    X_strat, y_strat = stratified_sampling(X, y, regression=True, samples_per_class=5)

    assert X_strat.shape == (50, 4)
    assert y_strat.shape == (50,)


def test_stratified_sampling_input_validation():
    """ Test that the input types are validated. """
    X = np.random.random((100, 4)).tolist()
    y = np.random.randint(0, 2, 100).tolist()

    with pytest.raises(AssertionError):
        stratified_sampling(X, y, regression=False, samples_per_class=10)

    X = np.random.random((100, 4))
    y = np.random.randint(0, 2, 101)

    with pytest.raises(AssertionError):
        stratified_sampling(X, y, regression=False, samples_per_class=10)

    X = np.random.random((100, 4))
    y = np.random.random(100)

    with pytest.raises(AssertionError):
        stratified_sampling(X, y, regression=True, samples_per_class=0)


def test_stratified_sampling_automatic_samples_per_class():
    """ Test that the stratified sampling works for classification with automatic samples_per_class. """
    X = np.random.random((100, 4))
    y = np.hstack([np.zeros(50, dtype=int), np.ones(25, dtype=int), np.full(25, 2, dtype=int)])

    X_strat, y_strat = stratified_sampling(X, y, regression=False, samples_per_class=None)

    assert X_strat.shape == (75, 4)
    assert y_strat.shape == (75,)

    unique_classes, counts = np.unique(y_strat, return_counts=True)
    assert np.all(counts == 25)
