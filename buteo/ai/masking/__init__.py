"""### Masking operations for images. ###

This module provides various masking functions for image data.
"""

# Import masking functionality 
# TODO: Create a proper masking implementation within submodules
# and reference that instead of this placeholder comment

# Import from noise module
from .noise import (
    _get_matched_noise_2d,
    _get_matched_noise_2d_binary,
    _get_matched_noise_3d,
    _get_matched_noise_3d_binary,
    _get_blurred_image,
)

# Import from pixel_masking module
from .pixel_masking import (
    mask_pixels_2d,
    MaskPixels2D,
    mask_pixels_3d,
    MaskPixels3D,
    mask_channels,
    MaskChannels,
)

# Import from line_masking module
from .line_masking import (
    mask_lines_2d,
    MaskLines2D,
    mask_lines_3d,
    MaskLines3D,
    mask_lines_2d_bezier,
    MaskLines2DBezier,
)

# Import from shape_masking module
from .shape_masking import (
    mask_elipse_2d,
    MaskElipse2D,
    mask_elipse_3d,
    MaskElipse3D,
    mask_rectangle_2d,
    MaskRectangle2D,
    mask_rectangle_3d,
    MaskRectangle3D,
)
