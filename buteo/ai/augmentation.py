"""
This module contains functions for augmenting images that are
suited to remote sensing imagery.
"""
# Standard library
import sys; sys.path.append("../../")
from typing import Optional, List, Callable

# External
import numpy as np

# Buteo
from buteo.array.utils_array import channel_first_to_last, channel_last_to_first
from buteo.ai.augmentation_funcs import (
    augmentation_mirror,
    augmentation_mirror_xy,
    augmentation_rotation,
    augmentation_rotation_xy,
    augmentation_noise_uniform,
    augmentation_noise_normal,
    augmentation_channel_scale,
    augmentation_contrast,
    augmentation_drop_channel,
    augmentation_drop_pixel,
    augmentation_blur,
    augmentation_blur_xy,
    augmentation_sharpen,
    augmentation_sharpen_xy,
    augmentation_cutmix,
    augmentation_mixup,
    augmentation_misalign
)



class AugmentationDataset():
    """
    A dataset that applies augmentations to the data.
    Every augmentation added needs to be specified as a dictionary
    with the following keys:
        - name (str)
        - chance (float[0,1])

    The following augmentations are supported:
        - rotation
        - mirror
        - channel_scale (additive(bool), max_value(float[0,1]))
        - noise (additive(bool), max_value(float[0,1]))
        - contrast (max_value(float[0,1]))
        - drop_pixel (drop_probability(float[0,1]), drop_value(float))
        - drop_channel (drop_probability(float[0,1]), drop_value(float))
        - blur
        - blur_xy
        - sharpen
        - sharpen_xy
        - cutmix (min_size(float[0, 1]), max_size(float[0, 1]))
        - mixup (min_size(float[0, 1]), max_size(float[0, 1]), label_mix(int))

    Parameters
    ----------
    X : np.ndarray
        The data to augment.

    y : np.ndarray
        The labels for the data.

    augmentations : list, optional
        The augmentations to apply.

    callback : callable, optional
        A callback to apply to the data after augmentation.

    input_is_channel_last : bool, default: True
        Whether the data is in channel last format.

    output_is_channel_last : bool, default: False
        Whether the output should be in channel last format.

    Returns
    -------
    AugmentationDataset
        A dataset yielding batches of augmented data. For Pytorch,
        convert the batches to tensors before ingestion.

    Example
    -------
    >>> def callback(x, y):
    ...     return (
    ...         torch.from_numpy(x).float(),
    ...         torch.from_numpy(y).float(),
    ...     )
    ...
    >>> dataset = AugmentationDataset(
    ...     x_train,
    ...     y_train,
    ...     callback=callback,
    ...     input_is_channel_last=True,
    ...     output_is_channel_last=False,
    ...     augmentations=[
    ...         { "name": "rotation", "chance": 0.2},
    ...         { "name": "mirror", "chance": 0.2 },
    ...         { "name": "noise", "chance": 0.2 },
    ...         { "name": "cutmix", "chance": 0.2 },
    ...     ],
    ... )
    >>>
    >>> from torch.utils.data import DataLoader
    >>> dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    """
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        augmentations: Optional[List] = None,
        callback: Callable = None,
        input_is_channel_last: bool = True,
        output_is_channel_last: bool = False,
    ):
        assert len(X) == len(y), "X and y must have the same length."
        assert X.dtype in [np.float16, np.float32, np.float64], "X must be a float array."
        assert y.dtype in [np.float16, np.float32, np.float64], "y must be a float array."

        # Convert data format if necessary
        if input_is_channel_last:
            self.x_train = channel_last_to_first(X)
            self.y_train = channel_last_to_first(y)
        else:
            self.x_train = X
            self.y_train = y

        # Set augmentations and callback
        self.augmentations = augmentations or []
        self.callback = callback
        self.channel_last = False
        self.output_is_channel_last = output_is_channel_last

        for aug in self.augmentations:

            # Check if augmentation is valid
            if "name" not in aug:
                raise ValueError("Augmentation name not specified.")

            if "p" not in aug and "chance" not in aug:
                raise ValueError("Augmentation chance not specified.")

            aug_name = aug["name"]
            if "chance" in aug:
                aug_change = aug["chance"]
            elif "p" in aug:
                aug_change = aug["p"]

            assert aug_change is not None, "Augmentation chance cannot be None."
            assert 0 <= aug_change <= 1, "Augmentation chance must be between 0 and 1."
            assert aug_name is not None, "Augmentation name cannot be None."

    def __len__(self):
        return len(self.x_train)

    def __getitem__(self, index):
        x = self.x_train[index].copy()
        y = self.y_train[index].copy()

        # Apply augmentations
        for aug in self.augmentations:
            aug_name = aug["name"]
            if "chance" in aug:
                aug_change = aug["chance"]
            elif "p" in aug:
                aug_change = aug["p"]
            func = None

            # Check if augmentation should be applied
            if np.random.rand() > aug_change:
                break

            # Mapping augmentation names to their respective functions
            if aug_name == "rotation":
                func = augmentation_rotation
            elif aug_name == "rotation_xy":
                func = augmentation_rotation_xy
            elif aug_name == "mirror":
                func = augmentation_mirror
            elif aug_name == "mirror_xy":
                func = augmentation_mirror_xy
            elif aug_name == "channel_scale":
                func = augmentation_channel_scale
            elif aug_name == "noise_uniform":
                func = augmentation_noise_uniform
            elif aug_name == "noise_normal":
                func = augmentation_noise_normal
            elif aug_name == "contrast":
                func = augmentation_contrast
            elif aug_name == "drop_pixel":
                func = augmentation_drop_pixel
            elif aug_name == "drop_channel":
                func = augmentation_drop_channel
            elif aug_name == "blur":
                func = augmentation_blur
            elif aug_name == "blur_xy":
                func = augmentation_blur_xy
            elif aug_name == "sharpen":
                func = augmentation_sharpen
            elif aug_name == "sharpen_xy":
                func = augmentation_sharpen_xy
            elif aug_name == "misalign":
                func = augmentation_misalign
            elif aug_name == "cutmix":
                func = augmentation_cutmix
            elif aug_name == "mixup":
                func = augmentation_mixup

            if func is None:
                raise ValueError(f"Augmentation {aug['name']} not supported.")

            channel_last = self.channel_last
            kwargs = {key: value for key, value in aug.items() if key not in ["name", "chance", "inplace", "p"]}

            # Augmentations that apply to both image and label
            if aug_name in ["rotation_xy", "mirror_xy", "blur_xy", "sharpen_xy"]:
                x, y = func(x, y, channel_last=channel_last, inplace=True, **kwargs)

            # Augmentations that needs two images
            elif aug_name in ["cutmix", "mixup"]:
                idx_source = np.random.randint(len(self.x_train))
                xx = self.x_train[idx_source]
                yy = self.y_train[idx_source]

                x, y = func(x, y, xx, yy, channel_last=channel_last, inplace=True, **kwargs)

            # Augmentations that only apply to image
            else:
                x = func(x, channel_last=channel_last, inplace=True, **kwargs)

        # Apply callback if specified
        if self.callback is not None:
            x, y = self.callback(x, y)

        # Convert output format if necessary
        if self.output_is_channel_last:
            x = channel_first_to_last(x)
            y = channel_first_to_last(y)

        return x, y


class Dataset():
    """
    A dataset that does not apply any augmentations to the data.
    Allows a callback to be passed and can convert between
    channel formats.

    Parameters
    ----------
    X : np.ndarray
        The data to augment.

    y : np.ndarray
        The labels for the data.

    callback : callable, optional
        A callback to apply to the data after augmentation.

    input_is_channel_last : bool, default: True
        Whether the data is in channel last format.

    output_is_channel_last : bool, default: False
        Whether the output should be in channel last format.

    Returns
    -------
    Dataset
        A dataset yielding batches of data. For Pytorch,
        convert the batches to tensors before ingestion.
    """
    def __init__(self,
        X: np.ndarray,
        y: np.ndarray,
        callback: Callable = None,
        input_is_channel_last: bool = True,
        output_is_channel_last: bool = False,
    ):
        # Convert input format if necessary
        if input_is_channel_last:
            self.x_train = channel_last_to_first(X)
            self.y_train = channel_last_to_first(y)
        else:
            self.x_train = X
            self.y_train = y

        self.callback = callback
        self.channel_last = False
        self.output_is_channel_last = output_is_channel_last

    def __len__(self):
        return len(self.x_train)

    def __getitem__(self, index):
        x = self.x_train[index]
        y = self.y_train[index]

        # Apply callback if specified
        if self.callback is not None:
            x, y = self.callback(x, y)

        # Convert output format if necessary
        if self.output_is_channel_last:
            x = channel_first_to_last(x)
            y = channel_first_to_last(y)

        return x, y
