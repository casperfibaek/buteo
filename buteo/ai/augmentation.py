"""
This module contains functions for augmenting images that are
suited to remote sensing imagery.
"""
# Standard library
import sys; sys.path.append("../../")
from typing import Optional, List, Callable, Union, Tuple

# External
import numpy as np

# Buteo
from buteo.array.utils_array import channel_first_to_last, channel_last_to_first
from buteo.array.loaders import MultiArray
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
    def __init__(self,
        X: Union[Union[np.ndarray, MultiArray], List[Union[np.ndarray, MultiArray]]],
        y: Union[Union[np.ndarray, MultiArray], List[Union[np.ndarray, MultiArray]]],
        augmentations: Optional[List] = None,
        callback: Callable = None,
        callback_pre: Callable = None,
        input_is_channel_last: Optional[bool] = None,
        output_is_channel_last: Optional[bool] = None,
    ):
        self.x_train = X
        self.y_train = y

        self.augmentations = augmentations or []
        self.callback = callback
        self.callback_pre = callback_pre
        self.input_is_channel_last = input_is_channel_last
        self.output_is_channel_last = output_is_channel_last

        # Read the first sample to determine if it is multi input
        self.x_is_multi_input = isinstance(self.x_train[0], list) and len(self.x_train[0]) > 1
        self.y_is_multi_input = isinstance(self.y_train[0], list) and len(self.y_train[0]) > 1

        assert len(self.x_train) == len(self.y_train), "X and y must have the same length."
        assert input_is_channel_last is not None, "Input channel format must be specified."
        assert output_is_channel_last is not None, "Output channel format must be specified."

        # If X is more than one array, then we need to make sure that the
        # number of list of augmentations is the same as the number of arrays.
        if self.x_is_multi_input or self.y_is_multi_input:
            if len(self.augmentations) == 0:
                pass
            elif not isinstance(self.augmentations[0], list):
                x_len = len(self.x_train[0]) if self.x_is_multi_input else 1
                y_len = len(self.y_train[0]) if self.y_is_multi_input else 1

                self.augmentations = [self.augmentations] * (x_len + y_len)
            else:
                assert len(self.augmentations) == len(X), "Number of augmentations must match number of arrays."

        test_augs = [self.augmentations] if not (self.x_is_multi_input or self.y_is_multi_input) else self.augmentations
        for aug_outer in test_augs:
            for aug in aug_outer:

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

                # Check if augmentation is valid for multi input
                if (aug_name[-2:] == "xy" or aug_name in ["cutmix", "mixup"]) and (self.x_is_multi_input or self.y_is_multi_input):
                    raise ValueError("Augmentation that target labels are not supported for multi input. (_xy augmentations)")

    def __len__(self):
        return len(self.x_train)

    def __getitem__(self, index):
        sample_x = self.x_train[index]
        sample_y = self.y_train[index]

        # Remeber to copy the data, otherwise it will be changed inplace
        converted_x = False
        if not isinstance(sample_x, list):
            sample_x = [sample_x]
            converted_x = True
        
        converted_y = False
        if not isinstance(sample_y, list):
            sample_y = [sample_y]
            converted_y = True

        # Copy the data. For Pytorch.
        for i in range(len(sample_x)):
            sample_x[i] = sample_x[i].copy()
        
        for j in range(len(sample_y)):
            sample_y[j] = sample_y[j].copy()

        if self.input_is_channel_last:
            for i in range(len(sample_x)):
                sample_x[i] = channel_last_to_first(sample_x[i])
            
            for j in range(len(sample_y)):
                sample_y[j] = channel_last_to_first(sample_y[j])

        # Apply callback_pre if specified. For normalisation.
        if self.callback_pre is not None:
            preconv_x = sample_x
            if converted_x:
                preconv_x = sample_x[0]
            
            preconv_y = sample_y
            if converted_y:
                preconv_y = sample_y[0]

            sample_x, sample_y = self.callback_pre(preconv_x, preconv_y)

            if converted_x:
                sample_x = [sample_x]
            
            if converted_y:
                sample_y = [sample_y]

        if self.x_is_multi_input or self.y_is_multi_input:
            # Apply augmentations
            for i in range(len(sample_x)):
                for aug in self.augmentations[i]:
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
                    elif aug_name == "mirror":
                        func = augmentation_mirror
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
                    elif aug_name == "sharpen":
                        func = augmentation_sharpen
                    elif aug_name == "misalign":
                        func = augmentation_misalign

                    if func is None:
                        raise ValueError(f"Augmentation {aug['name']} not supported.")

                    kwargs = {key: value for key, value in aug.items() if key not in ["name", "chance", "inplace", "p"]}

                    sample_x[i] = func(sample_x[i], channel_last=False, inplace=True, **kwargs)
        else:
            x = sample_x[0]
            y = sample_y[0]

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

                kwargs = {key: value for key, value in aug.items() if key not in ["name", "chance", "inplace", "p"]}

                # Augmentations that apply to both image and label
                if aug_name in ["rotation_xy", "mirror_xy", "blur_xy", "sharpen_xy"]:
                    x, y = func(x, y, channel_last=False, inplace=True, **kwargs)

                # Augmentations that needs two images
                elif aug_name in ["cutmix", "mixup"]:
                    idx_source = np.random.randint(len(self.x_train))
                    xx = self.x_train[idx_source]
                    yy = self.y_train[idx_source]

                    if isinstance(xx, list):
                        xx = xx[0]
                    if isinstance(yy, list):
                        yy = yy[0]

                    if self.input_is_channel_last:
                        xx = channel_last_to_first(xx)
                        yy = channel_last_to_first(yy)

                    x, y = func(x, y, xx, yy, channel_last=False, inplace=True, **kwargs)

                # Augmentations that only apply to image
                else:
                    x = func(x, channel_last=False, inplace=True, **kwargs)
                
                sample_x = [x]
                sample_y = [y]

        # Apply callback if specified
        if self.callback is not None:
            if converted_x:
                sample_x = sample_x[0]
        
            if converted_y:
                sample_y = sample_y[0]

            sample_x, sample_y = self.callback(sample_x, sample_y)
        
        if self.callback is None and converted_x:
            sample_x = sample_x[0]
        
        if self.callback is None and converted_y:
            sample_y = sample_y[0]

        if self.output_is_channel_last:
            for i in range(len(sample_x)):
                sample_x[i] = channel_first_to_last(sample_x[i])
            
            for j in range(len(sample_y)):
                sample_y[j] = channel_first_to_last(sample_y[j])

        if converted_x:
            sample_x = sample_x[0]
        
        if converted_y:
            sample_y = sample_y[0]

        return sample_x, sample_y


class Dataset():
    """
    A dataset that does not apply any augmentations to the data.
    Allows a callback to be passed and can convert between
    channel formats.

    Parameters
    ----------
    X : np.ndarray
        The data to read.

    y : np.ndarray
        The labels for the data.

    callback : callable, optional
        A callback to apply to the data before returning.
        Inside the callback, the format will always be channel first.

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
        X: Union[Union[np.ndarray, MultiArray], List[Union[np.ndarray, MultiArray]]],
        y: Union[Union[np.ndarray, MultiArray], List[Union[np.ndarray, MultiArray]]],
        callback: Callable = None,
        input_is_channel_last: Optional[bool] = None,
        output_is_channel_last: Optional[bool] = None,
    ):
        self.x_train = X
        self.y_train = y

        self.callback = callback
        self.channel_last = False
        self.input_is_channel_last = input_is_channel_last
        self.output_is_channel_last = output_is_channel_last

        assert len(self.x_train) == len(self.y_train), "X and y must have the same length."
        assert input_is_channel_last is not None, "Input channel format must be specified."
        assert output_is_channel_last is not None, "Output channel format must be specified."

    def __len__(self):
        return len(self.x_train)

    def __getitem__(self, index):
        sample_x = self.x_train[index]
        sample_y = self.y_train[index]

        converted_x = False
        if not isinstance(sample_x, list):
            sample_x = [sample_x]
            converted_x = True
        
        converted_y = False
        if not isinstance(sample_y, list):
            sample_y = [sample_y]
            converted_y = True

        # Copy the data. For Pytorch.
        for i in range(len(sample_x)):
            sample_x[i] = sample_x[i].copy()
        
        for j in range(len(sample_y)):
            sample_y[j] = sample_y[j].copy()

        if self.input_is_channel_last:
            for i in range(len(sample_x)):
                sample_x[i] = channel_last_to_first(sample_x[i])
            
            for j in range(len(sample_y)):
                sample_y[j] = channel_last_to_first(sample_y[j])

        # Apply callback if specified
        if self.callback is not None:
            preconv_x = sample_x
            if converted_x:
                preconv_x = sample_x[0]
            
            preconv_y = sample_y
            if converted_y:
                preconv_y = sample_y[0]

            sample_x, sample_y = self.callback(preconv_x, preconv_y)

            if converted_x:
                sample_x = [sample_x]
            
            if converted_y:
                sample_y = [sample_y]

        # Convert output format if necessary
        if self.output_is_channel_last:
            for i in range(len(sample_x)):
                sample_x[i] = channel_first_to_last(sample_x[i])
            
            for j in range(len(sample_y)):
                sample_y[j] = channel_first_to_last(sample_y[j])

        if converted_x:
            sample_x = sample_x[0]
        
        if converted_y:
            sample_y = sample_y[0]

        return sample_x, sample_y
