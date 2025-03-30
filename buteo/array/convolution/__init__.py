"""### Convolution module for array operations. ###"""

from .base import (
    pad_array,
    convolve_array,
    convolve_array_simple,
    convolve_array_channels
)

from .kernels import (
    kernel_base,
    kernel_shift,
    kernel_unsharp,
    kernel_sobel,
    kernel_get_offsets_and_weights
)

__all__ = [
    "pad_array",
    "convolve_array",
    "convolve_array_simple",
    "convolve_array_channels",
    "kernel_base",
    "kernel_shift",
    "kernel_unsharp",
    "kernel_sobel",
    "kernel_get_offsets_and_weights"
]
