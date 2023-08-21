# Standard library
import sys; sys.path.append("../")


import numpy as np
from buteo.ai.masking import MaskImages
from buteo.ai.masking_funcs import MaskRectangle2D, MaskLines3D

def test_mask_images():
    # Create a 3D array with shape (3, 4, 4) and random values between 0 and 1
    arr = np.random.rand(3, 4, 4)

    # Create a MaskImages object with a list of masking functions
    masker = MaskImages(
        masking_functions=[
            MaskLines3D(p=1.0),
        ],
        per_channel=False,
        method=3,
        min_val=0,
        max_val=255,
        channel_last=True,
        inplace=False,
    )

    # Call the MaskImages object with the array
    masked_arr = masker(arr)

    # Check that the masked array has the same shape as the original array
    assert masked_arr.shape == arr.shape

    # Create a 2D array with shape (4, 4) and random values between 0 and 1
    arr = np.random.rand(5, 5, 1)

    # Create a MaskImages object with a list of masking functions
    masker = MaskImages(
        masking_functions=[
            MaskRectangle2D(p=1.0),
        ],
        per_channel=False,
        method=3,
        min_val=0,
        max_val=255,
        channel_last=True,
        inplace=False,
    )
    # Call the MaskImages object with the array
    masked_arr = masker(arr)

    # Check that the masked array has the same shape as the original array
    assert masked_arr.shape == arr.shape
  