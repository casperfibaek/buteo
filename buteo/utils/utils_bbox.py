"""### Bounding box utility functions. ###

Various utility functions to work with bounding boxes and gdal.

There are two different formats for bounding boxes used by GDAL:</br>
OGR:  `[x_min, x_max, y_min, y_max]`</br>
WARP: `[x_min, y_min, x_max, y_max]`</br>

_If nothing else is stated, the OGR format is used._
"""

# Re-export validation functions
from buteo.utils.bbox_validation import (
    _check_is_valid_bbox,
    _check_is_valid_bbox_latlng,
    _check_is_valid_geotransform,
    _check_bboxes_intersect,
    _check_bboxes_within,
)

# Re-export operations functions
from buteo.utils.bbox_operations import (
    _get_pixel_offsets,
    _get_bbox_from_geotransform,
    _get_intersection_bboxes,
    _get_union_bboxes,
    _get_aligned_bbox_to_pixel_size,
    _get_gdal_bbox_from_ogr_bbox,
    _get_ogr_bbox_from_gdal_bbox,
    _get_geotransform_from_bbox,
    _get_sub_geotransform,
)

# Re-export conversion functions
from buteo.utils.bbox_conversion import (
    _get_geom_from_bbox,
    _get_bbox_from_geom,
    _get_wkt_from_bbox,
    _get_geojson_from_bbox,
    _get_vector_from_bbox,
    _transform_point,
    _transform_bbox_coordinates,
    _create_polygon_from_points,
    _get_bounds_from_bbox_as_geom,
    _get_bounds_from_bbox_as_wkt,
    _get_vector_from_geom,
)

# Re-export source functions
from buteo.utils.bbox_source import (
    _get_bbox_from_raster,
    _get_bbox_from_vector,
    _get_bbox_from_vector_layer,
    get_bbox_from_dataset,
    _get_utm_zone_from_bbox,
    _get_utm_zone_from_dataset,
    _get_utm_zone_from_dataset_list,
)

# Define public API
__all__ = [
    # Validation
    "_check_is_valid_bbox",
    "_check_is_valid_bbox_latlng",
    "_check_is_valid_geotransform",
    "_check_bboxes_intersect",
    "_check_bboxes_within",

    # Operations
    "_get_pixel_offsets",
    "_get_bbox_from_geotransform",
    "_get_intersection_bboxes",
    "_get_union_bboxes",
    "_get_aligned_bbox_to_pixel_size",
    "_get_gdal_bbox_from_ogr_bbox",
    "_get_ogr_bbox_from_gdal_bbox",
    "_get_geotransform_from_bbox",
    "_get_sub_geotransform",

    # Conversion
    "_get_geom_from_bbox",
    "_get_bbox_from_geom",
    "_get_wkt_from_bbox",
    "_get_geojson_from_bbox",
    "_get_vector_from_bbox",
    "_transform_point",
    "_transform_bbox_coordinates",
    "_create_polygon_from_points",
    "_get_bounds_from_bbox_as_geom",
    "_get_bounds_from_bbox_as_wkt",
    "_get_vector_from_geom",

    # Source
    "_get_bbox_from_raster",
    "_get_bbox_from_vector",
    "_get_bbox_from_vector_layer",
    "get_bbox_from_dataset",
    "_get_utm_zone_from_bbox",
    "_get_utm_zone_from_dataset",
    "_get_utm_zone_from_dataset_list",
]
