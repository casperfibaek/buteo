""" ### Create patches from rasters. ###

This module is maintained for backward compatibility.
New code should import from buteo.array.patches.* submodules directly.
"""

# Internal
from buteo.array.patches.extraction import array_to_patches, _array_to_patches_single, _patches_to_array_single
from buteo.array.patches.prediction import predict_array, predict_array_pixel
from buteo.array.patches.util import (
    _get_kernel_weights,
    _get_offsets,
    _borders_are_necessary,
    _borders_are_necessary_list, 
    _patches_to_weights,
    _unique_values,
)
from buteo.array.patches.merging import (
    _merge_weighted_median,
    _merge_weighted_average,
    _merge_weighted_minmax,
    _merge_weighted_olympic,
    _merge_weighted_mad,
    _merge_weighted_mode,
)


__all__ = [
    "array_to_patches",
    "predict_array",
    "predict_array_pixel",
]
