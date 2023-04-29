"""
### Perform morphological operations on arrays and rasters.  ###
"""

# Standard library
import sys; sys.path.append("../../")
from typing import Union

# External
import numpy as np

# Internal
from buteo.array.convolution import convolve_array
from buteo.array.convolution_kernels import get_kernel_sobel, get_offsets_and_weights

# problem med kanterne.
def edge_detection(
    arr: np.ndarray,
    radius: Union[int, float] = 2,
    scale: Union[int, float] = 2,
    gradient: bool = False,
) -> np.ndarray:
    """ Perform an detection method. """
    arr = arr.astype(np.float32, copy=False)
    mask = None
    if np.ma.isMaskedArray(arr):
        nodata = True
        nodata_value = arr.fill_value
        arr = np.ma.getdata(arr)
        mask = np.ma.getmask(arr)
    else:
        nodata = False
        nodata_value = 0.0

    kernel_gx, kernel_gy = get_kernel_sobel(radius, scale)

    offsets_gx, weights_gx, _ = get_offsets_and_weights(kernel_gx)
    offsets_gy, weights_gy, _ = get_offsets_and_weights(kernel_gy)

    arr_gx = convolve_array(
        arr,
        offsets_gx,
        weights_gx,
        nodata=nodata,
        nodata_value=nodata_value,
    )

    arr_gy = convolve_array(
        arr,
        offsets_gy,
        weights_gy,
        nodata=nodata,
        nodata_value=nodata_value,
    )

    magnitude = np.sqrt(arr_gx ** 2 + arr_gy ** 2)

    if mask is not None:
        magnitude = np.ma.array(magnitude, mask=mask)

    if gradient:
        gradient = np.arctan2(arr_gy, arr_gx)

        return magnitude, gradient

    return magnitude


if __name__ == "__main__":
    import os
    from buteo.raster.core_raster import raster_to_array, array_to_raster

    FOLDER = "C:/Users/casper.fibaek/OneDrive - ESA/Desktop/buteo/tests/"
    file = os.path.join(FOLDER, "test_image_rgb_8bit.tif")

    arr = raster_to_array(file, filled=True, fill_value=0.0, cast=np.float32)

    magnitude, gradient = edge_detection(arr, gradient=True)

    array_to_raster(
        magnitude,
        reference=os.path.join(FOLDER, "test_image_rgb_8bit.tif"),
        out_path=os.path.join(FOLDER, "test_image_rgb_8bit_sobel_magnitude_5.tif"),
    )

    array_to_raster(
        gradient,
        reference=os.path.join(FOLDER, "test_image_rgb_8bit.tif"),
        out_path=os.path.join(FOLDER, "test_image_rgb_8bit_sobel_gradient_5.tif"),
    )
