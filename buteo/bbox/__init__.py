"""### Bounding Box Module ###

Provides tools for creating, manipulating, and validating bounding boxes.

This module offers functionality for working with bounding boxes in various formats,
including extraction from geospatial datasets, coordinate transformations,
format conversions, and geometric operations.

The module provides both function-based and object-oriented interfaces:
- Function-based: Direct manipulation of bbox coordinate arrays
- Object-oriented: BBox class with methods for common operations

Public Classes:
--------------
BBox : Class for consistent handling of bounding boxes with utility methods

Public Functions:
----------------
get_bbox_from_dataset : Extracts the bounding box from a raster or vector dataset
union_bboxes : Calculates the union of two bounding boxes
intersection_bboxes : Calculates the intersection of two bounding boxes
bbox_to_geom : Converts a bounding box to an OGR geometry
bbox_from_geom : Extracts a bounding box from an OGR geometry
bbox_to_wkt : Converts a bounding box to WKT polygon
bbox_to_geojson : Converts a bounding box to GeoJSON polygon
align_bbox : Aligns a bounding box to pixel grid
validate_bbox : Validates a bounding box
validate_bbox_latlng : Validates a bounding box in lat/long coordinates

Utility Functions:
----------------
create_bbox_from_points : Creates a bbox from a collection of points
convert_bbox_ogr_to_gdal : Converts from OGR to GDAL bbox format
convert_bbox_gdal_to_ogr : Converts from GDAL to OGR bbox format
get_bbox_center : Gets the center point of a bbox
buffer_bbox : Buffers a bbox by a fixed distance
get_bbox_aspect_ratio : Gets the aspect ratio of a bbox
bbox_contains_point : Checks if a bbox contains a point
"""

from beartype import beartype
from typing import List, Union, Dict, Sequence, Tuple
from osgeo import ogr, osr

# Type aliases
BboxType = Sequence[Union[int, float]]
NumType = Union[int, float]
PointsType = Sequence[Sequence[NumType]]

# Import core functions
from .source import get_bbox_from_dataset
from .operations import _get_union_bboxes, _get_intersection_bboxes, _get_aligned_bbox_to_pixel_size
from .conversion import (_get_geom_from_bbox, _get_bbox_from_geom, _get_wkt_from_bbox, 
                         _get_geojson_from_bbox)
from .validation import _check_is_valid_bbox, _check_is_valid_bbox_latlng

# Optional utility functions (public but specialized, not included in __all__)
from .source import _get_utm_zone_from_dataset, _get_utm_zone_from_dataset_list

# Public API wrapper functions

@beartype
def union_bboxes(
    bbox1_ogr: BboxType,
    bbox2_ogr: BboxType
) -> List[float]:
    """Calculates the union (bounding hull) of two OGR formatted bboxes.

    Parameters
    ----------
    bbox1_ogr : BboxType
        The first OGR formatted bbox: `[x_min, x_max, y_min, y_max]`.
    bbox2_ogr : BboxType
        The second OGR formatted bbox: `[x_min, x_max, y_min, y_max]`.

    Returns
    -------
    List[float]
        An OGR formatted bbox representing the union:
        `[x_min, x_max, y_min, y_max]`.

    Raises
    ------
    ValueError
        If either input is not a valid OGR formatted bbox.

    Examples
    --------
    >>> union_bboxes([0, 1, 0, 1], [1, 2, 1, 2])
    [0.0, 2.0, 0.0, 2.0]
    >>> union_bboxes([-10, 0, -10, 0], [0, 10, 0, 10])
    [-10.0, 10.0, -10.0, 10.0]
    """
    return _get_union_bboxes(bbox1_ogr, bbox2_ogr)


@beartype
def intersection_bboxes(
    bbox1_ogr: BboxType,
    bbox2_ogr: BboxType
) -> List[float]:
    """Calculates the intersection of two OGR formatted bounding boxes.

    Parameters
    ----------
    bbox1_ogr : BboxType
        The first OGR formatted bbox: `[x_min, x_max, y_min, y_max]`.
    bbox2_ogr : BboxType
        The second OGR formatted bbox: `[x_min, x_max, y_min, y_max]`.

    Returns
    -------
    List[float]
        An OGR formatted bbox representing the intersection:
        `[x_min, x_max, y_min, y_max]`.

    Raises
    ------
    ValueError
        If either input is not a valid OGR bbox, or if the
        bounding boxes do not intersect.

    Examples
    --------
    >>> intersection_bboxes([0, 2, 0, 2], [1, 3, 1, 3])
    [1.0, 2.0, 1.0, 2.0]
    >>> intersection_bboxes([0, 1, 0, 1], [1, 2, 1, 2]) # Corner touch
    [1.0, 1.0, 1.0, 1.0]
    """
    return _get_intersection_bboxes(bbox1_ogr, bbox2_ogr)


@beartype
def bbox_to_geom(
    bbox_ogr: BboxType
) -> ogr.Geometry:
    """Converts an OGR formatted bounding box to an OGR Polygon Geometry.

    Parameters
    ----------
    bbox_ogr : BboxType
        An OGR formatted bounding box: `[x_min, x_max, y_min, y_max]`.

    Returns
    -------
    ogr.Geometry
        An OGR Polygon geometry representing the bounding box.

    Raises
    ------
    ValueError
        If `bbox_ogr` is not a valid OGR bbox, contains NaN values,
        or if geometry creation fails.

    Examples
    --------
    >>> bbox = [0.0, 1.0, 0.0, 1.0]
    >>> geom = bbox_to_geom(bbox)
    >>> isinstance(geom, ogr.Geometry)
    True
    >>> geom.GetGeometryName()
    'POLYGON'
    """
    return _get_geom_from_bbox(bbox_ogr)


@beartype
def bbox_from_geom(geom: ogr.Geometry) -> List[float]:
    """Extracts the OGR bounding box from an OGR Geometry.

    Parameters
    ----------
    geom : ogr.Geometry
        An OGR geometry object.

    Returns
    -------
    List[float]
        An OGR formatted bounding box: `[x_min, x_max, y_min, y_max]`.

    Raises
    ------
    TypeError
        If `geom` is not a valid `ogr.Geometry` object.
    ValueError
        If the geometry envelope cannot be computed or contains invalid
        (NaN) values, or if the resulting bbox is invalid.
    """
    return _get_bbox_from_geom(geom)


@beartype
def bbox_to_wkt(
    bbox_ogr: BboxType
) -> str:
    """Converts an OGR formatted bounding box to a WKT Polygon string.

    Parameters
    ----------
    bbox_ogr : BboxType
        An OGR formatted bounding box: `[x_min, x_max, y_min, y_max]`.

    Returns
    -------
    str
        A WKT string representing the bounding box as a Polygon.
        Example: `'POLYGON ((x_min y_min, x_max y_min, x_max y_max, x_min y_max, x_min y_min))'`

    Raises
    ------
    ValueError
        If `bbox_ogr` is not a valid OGR bbox or contains NaN values.

    Examples
    --------
    >>> bbox_to_wkt([0.0, 1.0, 2.0, 3.0])
    'POLYGON ((0.000000 2.000000, 1.000000 2.000000, 1.000000 3.000000, 0.000000 3.000000, 0.000000 2.000000))'
    """
    return _get_wkt_from_bbox(bbox_ogr)


@beartype
def bbox_to_geojson(
    bbox_ogr: BboxType
) -> Dict[str, Union[str, List[List[List[float]]]]]:
    """Converts an OGR formatted bounding box to a GeoJSON Polygon dictionary.

    Parameters
    ----------
    bbox_ogr : BboxType
        An OGR formatted bounding box: `[x_min, x_max, y_min, y_max]`.

    Returns
    -------
    Dict[str, Union[str, List[List[List[float]]]]]
        A GeoJSON dictionary representing the bounding box as a Polygon.
        Structure: `{ "type": "Polygon", "coordinates": [[[x, y], ...]] }`

    Raises
    ------
    ValueError
        If `bbox_ogr` is not a valid OGR bbox or contains NaN values.

    Examples
    --------
    >>> bbox_to_geojson([0.0, 1.0, 2.0, 3.0])
    {'type': 'Polygon', 'coordinates': [[[0.0, 2.0], [1.0, 2.0], [1.0, 3.0], [0.0, 3.0], [0.0, 2.0]]]}
    """
    return _get_geojson_from_bbox(bbox_ogr)


@beartype
def align_bbox(
    bbox_to_align_to_ogr: BboxType,
    bbox_to_be_aligned_ogr: BboxType,
    pixel_width: float,
    pixel_height: float,
) -> List[float]:
    """Aligns a bounding box (`bbox_to_be_aligned_ogr`) to the pixel grid
    defined by another bounding box (`bbox_to_align_to_ogr`) and pixel dimensions.

    The output bounding box encompasses `bbox_to_be_aligned_ogr` but its
    edges snap outwards to the grid defined by `bbox_to_align_to_ogr`.

    Parameters
    ----------
    bbox_to_align_to_ogr : BboxType
        The reference OGR bbox defining the grid origin: `[x_min, x_max, y_min, y_max]`.
    bbox_to_be_aligned_ogr : BboxType
        The OGR bbox to be aligned: `[x_min, x_max, y_min, y_max]`.
    pixel_width : float
        The target pixel width (must be positive).
    pixel_height : float
        The target pixel height (can be negative for north-up).

    Returns
    -------
    List[float]
        The aligned OGR formatted bbox: `[x_min, x_max, y_min, y_max]`.

    Raises
    ------
    ValueError
        If input bboxes are invalid, `pixel_width` is not positive,
        `pixel_height` is zero, or alignment results in NaN/infinite values.
    TypeError
        If pixel dimensions are not numeric.

    Examples
    --------
    >>> ref_bbox = [0.0, 4.0, 0.0, 4.0]
    >>> target_bbox = [1.2, 3.7, 1.2, 3.7]
    >>> # Align target to ref grid with 1x1 pixel size (pixel_height=-1)
    >>> align_bbox(ref_bbox, target_bbox, 1.0, -1.0)
    [1.0, 4.0, 1.0, 4.0]
    """
    return _get_aligned_bbox_to_pixel_size(
        bbox_to_align_to_ogr, bbox_to_be_aligned_ogr, pixel_width, pixel_height
    )


@beartype
def validate_bbox(bbox_ogr: BboxType) -> bool:
    """Validates if a bounding box is in valid OGR format.

    A valid OGR formatted bbox has the form: `[x_min, x_max, y_min, y_max]`.

    Validation Rules:
        - Must be a sequence (list or tuple) of 4 numbers.
        - `x_min` must be less than or equal to `x_max` (unless crossing dateline).
        - `y_min` must be less than or equal to `y_max`.
        - Infinite values are allowed.
        - None or NaN values are not allowed.
        - Non-numeric values are not allowed.

    Parameters
    ----------
    bbox_ogr : BboxType
        An OGR formatted bounding box: `[x_min, x_max, y_min, y_max]`.
        Note: This function uses beartype validation, so passing None or
        non-sequence values will raise a BeartypeCallHintParamViolation
        rather than returning False.

    Returns
    -------
    bool
        True if the bbox is valid, False otherwise.

    Raises
    ------
    beartype.roar.BeartypeCallHintParamViolation
        If bbox_ogr is not a sequence (list or tuple).

    Examples
    --------
    >>> validate_bbox([0, 1, 0, 1])
    True
    >>> validate_bbox([0, 1, 1, 0]) # y_min > y_max
    False
    """
    return _check_is_valid_bbox(bbox_ogr)


@beartype
def validate_bbox_latlng(bbox_ogr_latlng: BboxType) -> bool:
    """Validates if a bbox is in valid lat/long (WGS84) coordinates.

    A valid OGR formatted bbox in lat/long coordinates has the form:
    `[longitude_min, longitude_max, latitude_min, latitude_max]`
    where:
        - It must be a valid bbox according to `validate_bbox`.
        - Longitude values must be between -180 and 180 (inclusive).
        - Latitude values must be between -90 and 90 (inclusive).
        - Allows for dateline crossing (x_min > x_max is valid if within bounds).

    Parameters
    ----------
    bbox_ogr_latlng : BboxType
        An OGR formatted bbox potentially in lat/long coordinates:
        `[lng_min, lng_max, lat_min, lat_max]`.

    Returns
    -------
    bool
        True if the bbox is valid and within lat/long bounds, False otherwise.

    Examples
    --------
    >>> validate_bbox_latlng([-180, 180, -90, 90])
    True
    >>> validate_bbox_latlng([170, -170, -10, 10]) # Dateline crossing
    True
    >>> validate_bbox_latlng([-181, 180, -90, 90]) # Invalid longitude
    False
    """
    return _check_is_valid_bbox_latlng(bbox_ogr_latlng)


# Import BBox class and utility functions
from .bbox_class import (
    BBox, create_bbox_from_points, convert_bbox_ogr_to_gdal,
    convert_bbox_gdal_to_ogr, get_bbox_center, buffer_bbox,
    get_bbox_aspect_ratio, bbox_contains_point
)

# Define the functions exported when using wildcard import (*)
__all__ = [
    # Core bbox functions
    "get_bbox_from_dataset",
    "union_bboxes",
    "intersection_bboxes",
    "bbox_to_geom",
    "bbox_from_geom",
    "bbox_to_wkt",
    "bbox_to_geojson",
    "align_bbox",
    "validate_bbox",
    "validate_bbox_latlng",
    
    # BBox class
    "BBox",
    
    # Utility functions
    "create_bbox_from_points",
    "convert_bbox_ogr_to_gdal",
    "convert_bbox_gdal_to_ogr",
    "get_bbox_center",
    "buffer_bbox",
    "get_bbox_aspect_ratio",
    "bbox_contains_point"
]
