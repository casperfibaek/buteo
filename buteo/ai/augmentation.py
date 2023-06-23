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
        assert callback is None or callable(callback), "Callback must be callable."
        assert callback_pre is None or callable(callback_pre), "Callback_pre must be callable."

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
        for aug in test_augs:
            if not self._validate_augmentations(aug):
                raise ValueError("Augmentations are invalid.")

    def __len__(self):
        return len(self.x_train)
    
    def _validate_augmentations(self, augmentations) -> bool:
        """ Validates the augmentations. """
        multi_input = self.x_is_multi_input or self.y_is_multi_input

        if len(augmentations) == 0:
            return True
        
        for aug in augmentations:
                # Check if augmentation is valid
            if "name" not in aug:
                return False

            if "p" not in aug and "chance" not in aug:
                return False

            aug_name = aug["name"]
            if "chance" in aug:
                aug_change = aug["chance"]
            elif "p" in aug:
                aug_change = aug["p"]

            if aug_change is None:
                return False
            
            if not 0 <= aug_change <= 1:
                return False
            
            if aug_name is None:
                return False

            # Check if augmentation is valid for multi input
            if (aug_name[-2:] == "xy" or aug_name in ["cutmix", "mixup"]) and multi_input:
                return False
            
        return True

    def _apply_augmentations(self,
        sample_x: List[np.ndarray],
    ) -> List[np.ndarray]:
        """ Applies the augmentations to the sample. Inplace. Only x is possible. """
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

        return sample_x

    def _apply_augmentations_xy(self,
        sample_x: List[np.ndarray],
        sample_y: List[np.ndarray],
    ) -> List[np.ndarray]:
        """ Applies the augmentations to the sample. Inplace. Both x and y are possible. """
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

                xx, yy = (self.x_train[idx_source], self.y_train[idx_source])

                if isinstance(xx, list): xx = xx[0]
                if isinstance(yy, list): yy = yy[0]

                # Convert the format of the input data if necessary
                if self.input_is_channel_last:
                    xx = channel_last_to_first(xx)
                    yy = channel_last_to_first(yy)

                # If using cutmix/mixup, we have to draw samples from other images.
                # This means that we have to apply the callback_pre those samples
                # before applying the augmentation.
                if self.callback_pre is not None:
                    xx, yy = self.callback_pre(xx, yy)
                
                xx = np.array(xx) if isinstance(xx, np.memmap) else xx
                yy = np.array(yy) if isinstance(yy, np.memmap) else yy

                x, y = func(x, y, xx, yy, channel_last=False, inplace=True, **kwargs)

            # Augmentations that only apply to image
            else:
                x = func(x, channel_last=False, inplace=True, **kwargs)
            
            sample_x = [x]
            sample_y = [y]
        
        return sample_x, sample_y

    def __getitem__(self, index):
        # Ensure the samples are in list format
        sample_x, sample_y = self._ensure_list_format(self.x_train[index], self.y_train[index])

        # Convert the format of the input data if necessary
        if self.input_is_channel_last:
            sample_x = [channel_last_to_first(x) for x in sample_x]
            sample_y = [channel_last_to_first(y) for y in sample_y]

        # Apply callback_pre if specified. For normalization.
        if self.callback_pre is not None:
            sample_x, sample_y = self._apply_callback(self.callback_pre, sample_x, sample_y)

        sample_x = [np.array(x) if isinstance(x, np.memmap) else x.copy() for x in sample_x]
        sample_y = [np.array(y) if isinstance(y, np.memmap) else y.copy() for y in sample_y]

        # Apply augmentations
        if self.x_is_multi_input or self.y_is_multi_input:
            sample_x = self._apply_augmentations(sample_x)
        else:
            sample_x, sample_y = self._apply_augmentations_xy(sample_x, sample_y)

        # Apply callback if specified
        if self.callback is not None:
            sample_x, sample_y = self._apply_callback(self.callback, sample_x, sample_y)

        # Convert the format of the output data if necessary
        if self.output_is_channel_last:
            sample_x = [channel_first_to_last(x) for x in sample_x]
            sample_y = [channel_first_to_last(y) for y in sample_y]

        # Restore the samples to their original format
        return self._restore_original_format(sample_x), self._restore_original_format(sample_y)

    def _ensure_list_format(self, x, y):
        return (x if isinstance(x, list) else [x]), (y if isinstance(y, list) else [y])

    def _copy_and_convert_format(self, data, format_flag, converter_func):
        return converter_func(data) if format_flag else data

    def _apply_callback(self, callback_func, sample_x, sample_y):
        preconv_x, preconv_y = sample_x[0], sample_y[0]
        callback_x, callback_y = callback_func(preconv_x, preconv_y)
        return [callback_x], [callback_y]

    def _restore_original_format(self, data):
        return data[0] if len(data) == 1 else data


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
        Inside the callback, the format will always be channel first.

    input_is_channel_last : bool, default: True
        Whether the data is in channel last format.

    output_is_channel_last : bool, default: False
        Whether the output should be in channel last format.

    Returns
    -------
    Dataset
        A dataset to iterate over combining X and y.
    """
    def __init__(self,
        X: Union[Union[np.ndarray, MultiArray], List[Union[np.ndarray, MultiArray]]],
        y: Union[Union[np.ndarray, MultiArray], List[Union[np.ndarray, MultiArray]]],
        callback: Callable = None,
        input_is_channel_last: Optional[bool] = True,
        output_is_channel_last: Optional[bool] = False,
    ):
        assert len(X) == len(y), "X and y must have the same length."
        assert input_is_channel_last is not None, "Input channel format must be specified."
        assert output_is_channel_last is not None, "Output channel format must be specified."
        assert callback is None or callable(callback), "Callback must be callable."

        self.x_train = X
        self.y_train = y
        self.callback = callback
        self.input_is_channel_last = input_is_channel_last
        self.output_is_channel_last = output_is_channel_last

    def __len__(self):
        return len(self.x_train)

    def __getitem__(self, index):
        sample_x, sample_y = self._ensure_list_format(self.x_train[index], self.y_train[index])

        # Copy and optionally convert the format of the input data
        sample_x = [self._copy_and_convert_format(x, self.input_is_channel_last, channel_last_to_first) for x in sample_x]
        sample_y = [self._copy_and_convert_format(y, self.input_is_channel_last, channel_last_to_first) for y in sample_y]

        sample_x = [np.array(x) if isinstance(x, np.memmap) else x.copy() for x in sample_x]
        sample_y = [np.array(y) if isinstance(y, np.memmap) else y.copy() for y in sample_y]

        # Apply callback if specified
        if self.callback is not None:
            sample_x, sample_y = self._apply_callback(sample_x, sample_y)

        # Convert the output format if necessary
        sample_x = [self._copy_and_convert_format(x, self.output_is_channel_last, channel_first_to_last) for x in sample_x]
        sample_y = [self._copy_and_convert_format(y, self.output_is_channel_last, channel_first_to_last) for y in sample_y]

        return self._restore_original_format(sample_x), self._restore_original_format(sample_y)

    def _ensure_list_format(self, x, y):
        return (x if isinstance(x, list) else [x]), (y if isinstance(y, list) else [y])

    def _copy_and_convert_format(self, data, format_flag, converter_func):
        return converter_func(data) if format_flag else data

    def _apply_callback(self, sample_x, sample_y):
        preconv_x, preconv_y = sample_x[0], sample_y[0]
        callback_x, callback_y = self.callback(preconv_x, preconv_y)
        return [callback_x], [callback_y]

    def _restore_original_format(self, data):
        return data[0] if len(data) == 1 else data
