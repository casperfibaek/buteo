""" ### Patches module for working with raster patches. ###"""

from buteo.array.patches.extraction import array_to_patches
from buteo.array.patches.prediction import predict_array, predict_array_pixel

__all__ = [
    "array_to_patches",
    "predict_array",
    "predict_array_pixel",
]
