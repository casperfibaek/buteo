"""
This module contains functions for augmenting images that are
suited to remote sensing imagery.
"""
# Standard library
import sys; sys.path.append("../../")
from typing import Optional, List, Callable

import numpy as np
import buteo as beo

from buteo.ai.augmentation_funcs import (
    augmentation_rotation,
    augmentation_mirror,
    augmentation_channel_scale,
    augmentation_noise,
    augmentation_contrast,
    augmentation_blur,
    augmentation_blur_xy,
    augmentation_sharpen,
    augmentation_sharpen_xy,
    augmentation_drop_pixel,
    augmentation_drop_channel,
    augmentation_misalign,
    augmentation_cutmix,
    augmentation_mixup,
)


class AugmentationDataset():
    """
    A dataset that applies augmentations to the data.

    Args:
        X (np.ndarray): The data to augment.
        y (np.ndarray): The labels for the data.

    Keyword Args:
        augmentations (list): The augmentations to apply.
        callback (callable): A callback to apply to the data after augmentation.
        input_is_channel_last (bool=True): Whether the data is in channel last format.
        output_is_channel_last (bool=False): Whether the output should be in channel last format.
    
    Returns:
        A dataset yielding batches of augmented data. For Pytorch,
            convert the batches to tensors before ingestion.
    """
    def __init__(self,
        X: np.ndarray,
        y: np.ndarray,
        augmentations: Optional[List] = None,
        callback: Callable = None,
        input_is_channel_last: bool = True,
        output_is_channel_last: bool = False,
    ):
        if input_is_channel_last:
            self.x_train = beo.channel_last_to_first(X)
            self.y_train = beo.channel_last_to_first(y)
        else:
            self.x_train = X
            self.y_train = y

        if augmentations is None:
            self.augmentations = []
        else:
            self.augmentations = augmentations

        self.callback = callback
        self.channel_last = False
        self.output_is_channel_last = output_is_channel_last

    def __len__(self):
        return len(self.x_train)

    def __getitem__(self, index):
        x = self.x_train[index]
        y = self.y_train[index]

        for aug in self.augmentations:
            aug_name = aug["name"]
            if aug_name == "rotation":
                func = augmentation_rotation
            elif aug_name == "mirror":
                func = augmentation_mirror
            elif aug_name == "channel_scale":
                func = augmentation_channel_scale
            elif aug_name == "noise":
                func = augmentation_noise
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
            kwargs = {key: value for key, value in aug.items() if key != "name"}

            if aug_name in ["cutmix", "mixup"]:
                idx_source = np.random.randint(len(self.x_train))
                xx = self.x_train[idx_source]
                yy = self.y_train[idx_source]

                x, y = func(x, y, xx, yy, channel_last=channel_last, **kwargs)
            else:
                x, y = func(x, y, channel_last=channel_last, **kwargs)

        if self.callback is not None:
            x, y = self.callback(x, y)

        if self.output_is_channel_last:
            x = beo.channel_first_to_last(x)
            y = beo.channel_first_to_last(y)

        return x, y
