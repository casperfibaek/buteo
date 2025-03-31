"""### Bounding Box Module ###

Provides functions for creating, manipulating, and validating bounding boxes.

This module offers tools for working with bounding boxes in various formats,
including extraction from geospatial datasets, coordinate transformations,
format conversions, and geometric operations.

Public Functions:
----------------
get_bbox_from_dataset : Extracts the bounding box from a raster or vector dataset
"""

# Public API
from .source import get_bbox_from_dataset

# Optional utility functions (public but specialized, not included in __all__)
from .source import _get_utm_zone_from_dataset, _get_utm_zone_from_dataset_list

# Internal functions are not exported by default.
# If specific internal functions are needed elsewhere, they should be imported directly
# from their respective submodules (e.g., from buteo.bbox.operations import _get_pixel_offsets).

__all__ = ["get_bbox_from_dataset"]
