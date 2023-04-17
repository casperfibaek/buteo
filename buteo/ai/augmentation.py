"""
This module contains functions for augmenting images that are
suited to remote sensing imagery.
"""
# Standard library
import sys; sys.path.append("../../")
from typing import Optional, Tuple, List

# External
import numpy as np

# Internal
from buteo.ai.augmentation_batch import (
    augmentation_batch_rotation,
    augmentation_batch_mirror,
    augmentation_batch_channel_scale,
    augmentation_batch_noise,
    augmentation_batch_contrast,
    augmentation_batch_blur,
    augmentation_batch_sharpen,
    augmentation_batch_drop_pixel,
    augmentation_batch_drop_channel,
    augmentation_batch_misalign,
    augmentation_batch_cutmix,
    augmentation_batch_mixup,
)


def apply_augmentations(
    batch_x: np.ndarray,
    batch_y: Optional[np.ndarray],
    *,
    augmentations: Optional[List[dict]] = None,
    channel_last: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Apply a list of augmentations to a batch of images.
    
    Args:
        batch_x (np.ndarray): The batch of images to augment.
        batch_y (np.ndarray/None=None): The batch of labels to augment.

    Keyword Args:
        augmentations (list/dict=None): The list of augmentations to apply.
        channel_last (bool=True): Whether the image is (channels, height, width) or (height, width, channels).

    Returns:
        Tuple(np.ndarray, np.ndarray): The augmented batch of images and labels (if provided).
    """

    if augmentations is None or len(augmentations) == 0:
        return batch_x, batch_y

    batch_x, batch_y = batch_x.copy(), batch_y.copy()

    for aug in augmentations:
        if aug["name"] == "rotation":
            kwargs = {}
            if "chance" in aug:
                kwargs["chance"] = aug["chance"]
            if "channel_last" in aug:
                kwargs["channel_last"] = aug["channel_last"]
            else:
                kwargs["channel_last"] = channel_last

            batch_x, batch_y = augmentation_batch_rotation(
                batch_x,
                batch_y,
                **kwargs,
            )

        elif aug["name"] == "mirror":
            kwargs = {}
            if "chance" in aug:
                kwargs["chance"] = aug["chance"]
            if "channel_last" in aug:
                kwargs["channel_last"] = aug["channel_last"]
            else:
                kwargs["channel_last"] = channel_last

            batch_x, batch_y = augmentation_batch_mirror(
                batch_x,
                batch_y,
                **kwargs,
            )

        elif aug["name"] == "channel_scale":
            kwargs = {}
            if "chance" in aug:
                kwargs["chance"] = aug["chance"]
            if "amount" in aug:
                kwargs["amount"] = aug["amount"]
            if "additive" in aug:
                kwargs["additive"] = aug["additive"]
            if "channel_last" in aug:
                kwargs["channel_last"] = aug["channel_last"]
            else:
                kwargs["channel_last"] = channel_last

            batch_x, batch_y = augmentation_batch_channel_scale(
                batch_x,
                batch_y,
                **kwargs,
            )

        elif aug["name"] == "noise":
            kwargs = {}
            if "chance" in aug:
                kwargs["chance"] = aug["chance"]
            if "amount" in aug:
                kwargs["amount"] = aug["amount"]
            if "additive" in aug:
                kwargs["additive"] = aug["additive"]
            if "channel_last" in aug:
                kwargs["channel_last"] = aug["channel_last"]
            else:
                kwargs["channel_last"] = channel_last

            batch_x, batch_y = augmentation_batch_noise(
                batch_x,
                batch_y,
                **kwargs,
            )

        elif aug["name"] == "contrast":
            kwargs = {}
            if "chance" in aug:
                kwargs["chance"] = aug["chance"]
            if "contrast_factor" in aug:
                kwargs["contrast_factor"] = aug["contrast_factor"]
            if "channel_last" in aug:
                kwargs["channel_last"] = aug["channel_last"]
            else:
                kwargs["channel_last"] = channel_last

            batch_x, batch_y = augmentation_batch_contrast(
                batch_x,
                batch_y,
                **kwargs,
            )

        elif aug["name"] == "drop_pixel":
            kwargs = {}
            if "chance" in aug:
                kwargs["chance"] = aug["chance"]
            if "drop_probability" in aug:
                kwargs["drop_probability"] = aug["drop_probability"]
            if "drop_value" in aug:
                kwargs["drop_value"] = aug["drop_value"]
            if "channel_last" in aug:
                kwargs["channel_last"] = aug["channel_last"]
            else:
                kwargs["channel_last"] = channel_last

            batch_x, batch_y = augmentation_batch_drop_pixel(
                batch_x,
                batch_y,
                **kwargs,
            )

        elif aug["name"] == "drop_channel":
            kwargs = {}
            if "chance" in aug:
                kwargs["chance"] = aug["chance"]
            if "drop_probability" in aug:
                kwargs["drop_probability"] = aug["drop_probability"]
            if "drop_value" in aug:
                kwargs["drop_value"] = aug["drop_value"]
            if "channel_last" in aug:
                kwargs["channel_last"] = aug["channel_last"]
            else:
                kwargs["channel_last"] = channel_last

            batch_x, batch_y = augmentation_batch_drop_channel(
                batch_x,
                batch_y,
                **kwargs,
            )

        elif aug["name"] == "blur":
            kwargs = {}
            if "chance" in aug:
                kwargs["chance"] = aug["chance"]
            if "intensity" in aug:
                kwargs["intensity"] = aug["intensity"]
            if "channel_last" in aug:
                kwargs["channel_last"] = aug["channel_last"]
            else:
                kwargs["channel_last"] = channel_last

            batch_x, batch_y = augmentation_batch_blur(
                batch_x,
                batch_y,
                **kwargs,
            )

        elif aug["name"] == "sharpen":
            kwargs = {}
            if "chance" in aug:
                kwargs["chance"] = aug["chance"]
            if "intensity" in aug:
                kwargs["intensity"] = aug["intensity"]
            if "channel_last" in aug:
                kwargs["channel_last"] = aug["channel_last"]
            else:
                kwargs["channel_last"] = channel_last

            batch_x, batch_y = augmentation_batch_sharpen(
                batch_x,
                batch_y,
                **kwargs,
            )

        elif aug["name"] == "misalign":
            kwargs = {}
            if "chance" in aug:
                kwargs["chance"] = aug["chance"]
            if "max_offset" in aug:
                kwargs["max_offset"] = aug["max_offset"]
            if "max_images" in aug:
                kwargs["max_images"] = aug["max_images"]
            if "max_channels" in aug:
                kwargs["max_channels"] = aug["max_channels"]
            if "channel_last" in aug:
                kwargs["channel_last"] = aug["channel_last"]

            batch_x, batch_y = augmentation_batch_misalign(
                batch_x,
                batch_y,
                **kwargs,
            )

        elif aug["name"] == "cutmix":
            kwargs = {}
            if "chance" in aug:
                kwargs["chance"] = aug["chance"]
            if "max_size" in aug:
                kwargs["max_size"] = aug["max_size"]
            if "max_mixes" in aug:
                kwargs["max_mixes"] = aug["max_mixes"]
            if "feather" in aug:
                kwargs["feather"] = aug["feather"]
            if "feather_dist" in aug:
                kwargs["feather_dist"] = aug["feather_dist"]
            if "channel_last" in aug:
                kwargs["channel_last"] = aug["channel_last"]
            else:
                kwargs["channel_last"] = channel_last

            batch_x, batch_y = augmentation_batch_cutmix(
                batch_x,
                batch_y,
                **kwargs,
            )

        elif aug["name"] == "mixup":
            kwargs = {}
            if "chance" in aug:
                kwargs["chance"] = aug["chance"]
            if "max_mixes" in aug:
                kwargs["max_mixes"] = aug["max_mixes"]
            if "channel_last" in aug:
                kwargs["channel_last"] = aug["channel_last"]
            else:
                kwargs["channel_last"] = channel_last

            batch_x, batch_y = augmentation_batch_mixup(
                batch_x,
                batch_y,
                **kwargs,
            )

    return batch_x, batch_y


def augmentation_generator(
    X: np.ndarray,
    y: Optional[np.ndarray],
    batch_size: int = 64,
    augmentations: Optional[List[dict]] = None,
    shuffle: bool = True,
    seed: Optional[int] = None,
    channel_last: bool = True,
) -> np.ndarray:
    """
    Generate batches of augmented data.
    
    Args:
        X (np.ndarray): The data to augment.

    Keyword Args:
        y (np.ndarray): The labels for the data.
        batch_size (int): The size of the batches to generate.
        augmentations (list): The augmentations to apply.
        shuffle (bool): Whether to shuffle the data before generating batches.
        seed (int): The seed to use for shuffling.
        channel_last (bool): Whether the data is in channel last format.

    Returns:
        A generator yielding batches of augmented data.
    """
    if seed is not None:
        np.random.seed(seed)

    if shuffle:
        idx = np.random.permutation(X.shape[0])
        X = X[idx]
        y = y[idx]

    num_batches = (X.shape[0] + batch_size - 1) // batch_size

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, X.shape[0])

        batch_x = X[start_idx:end_idx]
        batch_y = y[start_idx:end_idx]

        batch_augmented_x, batch_augmented_y = apply_augmentations(
            batch_x,
            batch_y,
            augmentations=augmentations,
            channel_last=channel_last,
        )

        yield batch_augmented_x, batch_augmented_y


class AugmentationDataset:
    """
    A dataset that applies augmentations to the data.

    Args:
        X (np.ndarray): The data to augment.

    Keyword Args:
        y (np.ndarray): The labels for the data.
        batch_size (int): The size of the batches to generate.
        augmentations (list): The augmentations to apply.
        shuffle (bool): Whether to shuffle the data before generating batches.
            and whenever the dataset is iterated over.
        seed (int): The seed to use for shuffling.
        channel_last (bool): Whether the data is in channel last format.
    
    Returns:
        A dataset yielding batches of augmented data. For Pytorch,
            convert the batches to tensors before ingestion.
    """
    def __init__(
        self,
        X: np.ndarray,
        *,
        y: Optional[np.ndarray],
        batch_size: int = 64,
        augmentations: Optional[List[dict]] = None,
        shuffle: bool = True,
        seed: Optional[int] = None,
        channel_last: bool = True,
    ):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.augmentations = augmentations
        self.shuffle = shuffle
        self.seed = seed
        self.channel_last = channel_last
        self.generator = None

    def __iter__(self):
        self.generator = augmentation_generator(
            self.X,
            y=self.y,
            batch_size=self.batch_size,
            augmentations=self.augmentations,
            shuffle=self.shuffle,
            seed=self.seed,
            channel_last=self.channel_last,
        )
        return self

    def __next__(self):
        try:
            return next(self.generator)
        except StopIteration as e:
            raise StopIteration from e

    def __len__(self):
        num_batches = (self.X.shape[0] + self.batch_size - 1) // self.batch_size

        return num_batches
