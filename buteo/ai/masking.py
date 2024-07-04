"""This module contains functions for masking images that are
suited to remote sensing imagery.
"""
# Standard library
import sys; sys.path.append("../../")
from typing import List

# External
import numpy as np

# Internal
from buteo.ai.masking_funcs import mask_replace_2d, mask_replace_3d


class MaskImages():
    """A class that masks images when called using a list of masking functions.
    Parameters.
    NOTE: The images that are being masked should be floating point arrays.

    ----------
    masking_functions : List, optional
        A list of masking functions to apply to the image, default: None.

    method : int, optional
        The method to use for replacing pixels.
        0. replace with 0
        1. replace with 1
        2. replace with val
        3. replace with random value between min_val and max_val
        4. replace with random value binary value, either min_val or max_val
        5. replace with a blurred version of the image

    per_channel : bool, optional
        Whether to replace pixels per channel or for the same of each channel, default: True.

    val : float, optional
        The value to replace pixels with if method is 2, default: 0.0.

    min_val : float, optional
        The minimum value to replace pixels with if method is 3 or 4, default: 0.0.

    max_val : float, optional
        The maximum value to replace pixels with if method is 3 or 4, default: 1.0.

    inplace : bool, optional
        Whether to replace the pixels in the original array or return a copy, default: False.

    channel_last : bool, optional
        Whether the image is (channels, height, width) or (height, width, channels), default: True.

    Returns
    -------
    np.ndarray
        The array with replaced pixels.
    """
    def __init__(self,
        masking_functions: List = None, *,
        method: int = 0,
        per_channel: bool = True,
        val: float = 0.0,
        min_val: float = 0.0,
        max_val: float = 1.0,
        channel_last: bool = True,
        inplace: bool = False,
    ):
        self.masking_functions = masking_functions
        self.method = method
        self.per_channel = per_channel
        self.val = val
        self.min_val = min_val
        self.max_val = max_val
        self.channel_last = channel_last
        self.inplace = inplace

        if self.masking_functions is None:
            self.masking_functions = []


    def __call__(self, X):
        """Applies the masking functions to the input image.
        Parameters
        ----------
        X : np.ndarray
            The input image to mask.

        Returns
        -------
        np.ndarray
            The masked image.
        """
        X = X.astype(np.float32, copy=False)
        masks = None
        for i, func in enumerate(self.masking_functions):
            if i == 0:
                masks = func(X)
            else:
                masks = masks * func(X)

        if self.per_channel:
            return mask_replace_3d(
                X,
                masks,
                method=self.method,
                val=self.val,
                min_val=self.min_val,
                max_val=self.max_val,
                channel_last=self.channel_last,
                inplace=self.inplace,
            )
        
        return mask_replace_2d(
            X,
            masks,
            method=self.method,
            val=self.val,
            min_val=self.min_val,
            max_val=self.max_val,
            channel_last=self.channel_last,
            inplace=self.inplace,
        )

