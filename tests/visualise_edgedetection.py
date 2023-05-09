import os
import numpy as np
from buteo.raster.core_raster_io import raster_to_array, array_to_raster
from buteo.array.convolution import convolve_array_channels
from buteo.array.edge_detection import filter_edge_detection


FOLDER = "./"
path = os.path.join(FOLDER, "/features/test_image_rgb_8bit.tif")

arr = raster_to_array(path, filled=True, fill_value=0.0, cast=np.float32)

magnitude, gradient = filter_edge_detection(arr, gradient=True)

array_to_raster(
    magnitude,
    reference=path,
    out_path=os.path.join(FOLDER, "test_image_rgb_8bit_sobel_magnitude_3.tif"),
)

collapsed_magnitude = convolve_array_channels(
    magnitude, method=1,
)

array_to_raster(
    collapsed_magnitude,
    reference=path,
    out_path=os.path.join(FOLDER, "test_image_rgb_8bit_sobel_magnitude_3_collapsed_mean.tif"),
)

collapsed_gradient = convolve_array_channels(
    gradient, method=1,
)

array_to_raster(
    collapsed_gradient,
    reference=path,
    out_path=os.path.join(FOLDER, "test_image_rgb_8bit_sobel_gradient_3_collapsed_mean.tif"),
)
