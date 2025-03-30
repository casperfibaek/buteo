""" ### Functions for transforming rasters represented as arrays. ### """

from .color import *
from .convolution import *
from .distance import *
from .edge_detection import *
from .fill import *
from .filters import *
from .loaders import MultiArray
from .morphology import *
from .patches import *
from .timeseries import *
from .coordinate_encoding import *
from .utils_array import *

__all__ = [
    # From color
    "color_rgb_to_hsl",
    "color_hsl_to_rgb",
    
    # From convolution
    "pad_array",
    "convolve_array",
    "convolve_array_simple",
    "convolve_array_channels",
    "kernel_base",
    "kernel_shift",
    "kernel_unsharp",
    "kernel_sobel",
    "kernel_get_offsets_and_weights",
    
    # From loaders
    "MultiArray",
    
    # From morphology
    "morph_erode",
    "morph_dilate",
    "morph_open",
    "morph_close",
    "morph_tophat",
    "morph_bothat",
    "morph_difference",
    
    # Imported from other modules
    # These will also be available with "from buteo.array import *"
]
