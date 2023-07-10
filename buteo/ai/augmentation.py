"""
This module contains functions for augmenting images that are
suited to remote sensing imagery.
"""
# Standard library
import sys; sys.path.append("../../")
from typing import Optional, List, Callable, Union

# External
import numpy as np

# Buteo
from buteo.array.loaders import MultiArray


class Dataset:
    """
    A dataset that does not apply any augmentations to the data.
    Allows a callback to be passed and can convert between
    channel formats.

    Parameters
    ----------
    X : np.ndarray or MultiArray
        The data to read.

    y : np.ndarray or MultiArray
        The labels for the data.

    callback : callable, optional
        A callback to apply to the data before returning.

    Returns
    -------
    Dataset
        A dataset to iterate over combining X and y.
    """
    def __init__(self,
        X: Union[np.ndarray, MultiArray],
        y: Union[np.ndarray, MultiArray],
        callback: Callable = None,
    ):
        assert len(X) == len(y), "X and y must have the same length."
        assert callback is None or callable(callback), "Callback must be callable."

        self.x_train = X
        self.y_train = y
        self.callback = callback

    def __getitem__(self, index):
        sample_x, sample_y = (self.x_train[index], self.y_train[index])
        
        sample_x = np.array(sample_x) if isinstance(sample_x, np.memmap) else sample_x
        sample_y = np.array(sample_y) if isinstance(sample_y, np.memmap) else sample_y

        # Apply callback if specified
        if self.callback is not None:
            sample_x, sample_y = self.callback(sample_x, sample_y)

        return sample_x, sample_y

    def __len__(self):
        return len(self.x_train)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


class DatasetAugmentation:
    """
    A dataset that does not apply any augmentations to the data.
    Allows a callback to be passed and can convert between
    channel formats.

    Parameters
    ----------
    X : np.ndarray or MultiArray
        The data to read.

    y : np.ndarray or MultiArray
        The labels for the data.

    callback_pre_augmentation : callable, optional
        A callback to apply before augmentation.

    callback_post_augmentation : callable, optional
        A callback to apply after augmentation.

    augmentations : list, optional
        The augmentations to apply.
        
    Returns
    -------
    Dataset
        A dataset to iterate over combining X and y.
    """
    def __init__(self,
        X: Union[np.ndarray, MultiArray],
        y: Union[np.ndarray, MultiArray],
        callback_pre_augmentation: Callable = None,
        callback_post_augmentation: Callable = None,
        augmentations: Optional[List] = None,
    ):
        assert len(X) == len(y), "X and y must have the same length."
        assert callback_pre_augmentation is None or callable(callback_pre_augmentation), "callback_pre_augmentation must be callable."
        assert callback_post_augmentation is None or callable(callback_post_augmentation), "callback_post_augmentation must be callable."

        if augmentations is not None:
            assert isinstance(augmentations, list), "Augmentations must be a list."
            self.augmentations = augmentations
        else:
            self.augmentations = []

        self.x_train = X
        self.y_train = y
        self.callback_pre_augmentation = callback_pre_augmentation
        self.callback_post_augmentation = callback_post_augmentation

    def __getitem__(self, index):
        sample_x, sample_y = (self.x_train[index], self.y_train[index])
        
        sample_x = np.array(sample_x) if isinstance(sample_x, np.memmap) else sample_x
        sample_y = np.array(sample_y) if isinstance(sample_y, np.memmap) else sample_y

        # Apply callback if specified
        if self.callback_pre_augmentation is not None:
            sample_x, sample_y = self.callback_pre_augmentation(sample_x, sample_y)

        # Apply augmentations
        for aug in self.augmentations:
            if not isinstance(aug, Callable):
                raise ValueError("Augmentations must be callable.")

            if aug.requires_dataset:
                source_idx = np.random.randint(len(self.x_train))
                source_x, source_y = (self.x_train[source_idx], self.y_train[source_idx])
                
                if self.callback_pre_augmentation is not None:
                    source_x, source_y = self.callback_pre_augmentation(source_x, source_y)

                sample_x, sample_y = aug(sample_x, sample_y, source_x, source_y)

            elif aug.applies_to_features and aug.applies_to_labels:
                sample_x, sample_y = aug(sample_x, sample_y)
            elif aug.applies_to_features:
                sample_x = aug(sample_x)
            elif aug.applies_to_labels:
                sample_y = aug(sample_y)

        # Apply callback if specified
        if self.callback_post_augmentation is not None:
            sample_x, sample_y = self.callback_post_augmentation(sample_x, sample_y)

        return sample_x, sample_y

    def __len__(self):
        return len(self.x_train)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
