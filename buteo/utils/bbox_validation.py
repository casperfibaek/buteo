"""### Bounding box validation functions. ###

Functions to validate bounding boxes.

There are two different formats for bounding boxes used by GDAL:</br>
OGR:  `[x_min, x_max, y_min, y_max]`</br>
WARP: `[x_min, y_min, x_max, y_max]`</br>

_If nothing else is stated, the OGR format is used._
"""

# Standard library
from typing import List, Union

# External
import numpy as np


def _check_is_valid_bbox(bbox_ogr: List[Union[int, float]]) -> bool:
    """Checks if a bbox is valid.

    A valid OGR formatted bbox has the form:
        `[x_min, x_max, y_min, y_max]`

    The following rules apply:
    - Must be a list of 4 numbers
    - x_min must be less than or equal to x_max
    - y_min must be less than or equal to y_max
    - Infinite values are allowed
    - None values are not allowed
    - Non-numeric values are not allowed

    Parameters
    ----------
    bbox_ogr : List[Union[int, float]]
        An OGR formatted bbox.
        `[x_min, x_max, y_min, y_max]`

    Returns
    -------
    bool
        True if the bbox is valid, False otherwise.

    Examples
    --------
    >>> _check_is_valid_bbox([0, 1, 0, 1])
    True
    >>> _check_is_valid_bbox([1, 0, 0, 1]) # x_min > x_max
    False
    >>> _check_is_valid_bbox([0, 1, None, 1])
    False
    >>> _check_is_valid_bbox([-np.inf, np.inf, -90, 90])
    True
    """
    # Check if input is None
    if bbox_ogr is None:
        return False

    # Check if input is a list/tuple
    if not isinstance(bbox_ogr, (list, tuple)):
        return False

    # Check if length is 4
    if len(bbox_ogr) != 4:
        return False

    # Check if all values are numbers or infinite
    for val in bbox_ogr:
        if val is None:
            return False
        if not isinstance(val, (int, float)):
            return False

    # Unpack values
    try:
        x_min, x_max, y_min, y_max = bbox_ogr
    except ValueError:
        return False

    # Check for NaN values
    if any(np.isnan(val) for val in bbox_ogr):
        return False

    # Allow infinite values
    if any(np.isinf(val) for val in bbox_ogr):
        return True

    # Check if min values are less than or equal to max values
    if x_min > x_max or y_min > y_max:
        return False

    return True


def _check_is_valid_bbox_latlng(bbox_ogr_latlng: List[Union[int, float]]) -> bool:
    """Checks if a bbox is valid and in lat/long coordinates.

    A valid OGR formatted bbox in lat/long coordinates has the form:
        `[x_min, x_max, y_min, y_max]`
    where:
        - x values (longitude) must be between -180 and 180
        - y values (latitude) must be between -90 and 90

    Parameters
    ----------
    bbox_ogr_latlng : List[Union[int, float]]
        An OGR formatted bbox in lat/long coordinates.
        `[x_min, x_max, y_min, y_max]`

    Returns
    -------
    bool
        True if the bbox is valid lat/long coordinates, False otherwise.

    Examples
    --------
    >>> _check_is_valid_bbox_latlng([-180, 180, -90, 90])
    True
    >>> _check_is_valid_bbox_latlng([-181, 180, -90, 90])
    False
    >>> _check_is_valid_bbox_latlng([0, 1, -91, 90])
    False
    >>> _check_is_valid_bbox_latlng(None)
    False
    """
    # Check if input is None
    if bbox_ogr_latlng is None:
        return False

    # First check if it's a valid bbox in general
    if not _check_is_valid_bbox(bbox_ogr_latlng):
        return False

    try:
        x_min, x_max, y_min, y_max = bbox_ogr_latlng
    except ValueError:
        return False

    # Check for NaN values
    if any(np.isnan(val) for val in [x_min, x_max, y_min, y_max]):
        return False

    # Check longitude bounds (-180 to 180)
    if not (-180 <= x_min <= 180 and -180 <= x_max <= 180):
        return False

    # Check latitude bounds (-90 to 90)
    if not (-90 <= y_min <= 90 and -90 <= y_max <= 90):
        return False

    return True


def _check_is_valid_geotransform(geotransform: List[Union[int, float]]) -> bool:
    """Checks if a geotransform is valid.

    A valid geotransform has the form:
    `[x_min, pixel_width, row_skew, y_max, column_skew, pixel_height]`

    Parameters
    ----------
    geotransform : List[Union[int, float]]
        A GDAL formatted geotransform.
        `[x_min, pixel_width, row_skew, y_max, column_skew, pixel_height]`
        pixel_height is negative for north up.

    Returns
    -------
    bool
        True if the geotransform is valid, False otherwise.

    Examples
    --------
    >>> _check_is_valid_geotransform([0, 1, 0, 0, 0, -1])
    True
    >>> _check_is_valid_geotransform(None)
    False
    >>> _check_is_valid_geotransform([0, 1, 0, 0, 0]) # Wrong length
    False
    >>> _check_is_valid_geotransform([0, None, 0, 0, 0, -1]) # Contains None
    False
    """
    # Check if input is None
    if geotransform is None:
        return False

    # Check if input is list or tuple
    if not isinstance(geotransform, (list, tuple)):
        return False

    # Check length
    if len(geotransform) != 6:
        return False

    # Check if all values are numbers and not None
    try:
        for val in geotransform:
            # Check for None
            if val is None:
                return False

            # check if val is int or float, and not infinite
            if not isinstance(val, (int, float)):
                return False

            # Check if value is finite
            if not np.isfinite(float(val)) and not np.isinf(float(val)):
                return False

    except (TypeError, ValueError):
        return False

    # Verify pixel width is not zero
    if float(geotransform[1]) == 0:
        return False

    # Verify pixel height is not zero
    if float(geotransform[5]) == 0:
        return False

    return True


def _check_bboxes_intersect(
    bbox1_ogr: List[Union[int, float]],
    bbox2_ogr: List[Union[int, float]]
) -> bool:
    """Checks if two OGR formatted bboxes intersect.

    Parameters
    ----------
    bbox1_ogr : List[Union[int, float]]
        An OGR formatted bbox.
        `[x_min, x_max, y_min, y_max]`
    bbox2_ogr : List[Union[int, float]]
        An OGR formatted bbox.
        `[x_min, x_max, y_min, y_max]`

    Returns
    -------
    bool
        True if the bboxes intersect, False otherwise.

    Raises
    ------
    TypeError
        If either bbox is None or not a list/tuple
    ValueError
        If either bbox is not a valid OGR formatted bbox

    Examples
    --------
    >>> _check_bboxes_intersect([0, 1, 0, 1], [0.5, 1.5, 0.5, 1.5])
    True
    >>> _check_bboxes_intersect([0, 1, 0, 1], [2, 3, 2, 3])
    False
    >>> _check_bboxes_intersect(None, [0, 1, 0, 1])
    Raises TypeError
    >>> _check_bboxes_intersect([0, 1, None, 1], [0, 1, 0, 1])
    Raises ValueError
    """
    # Input validation
    if bbox1_ogr is None or bbox2_ogr is None:
        raise TypeError("bbox arguments cannot be None")

    if not isinstance(bbox1_ogr, (list, tuple)) or not isinstance(bbox2_ogr, (list, tuple)):
        raise TypeError("bbox arguments must be lists or tuples")

    if not _check_is_valid_bbox(bbox1_ogr):
        raise ValueError(f"bbox1_ogr is not a valid OGR bbox: {bbox1_ogr}")

    if not _check_is_valid_bbox(bbox2_ogr):
        raise ValueError(f"bbox2_ogr is not a valid OGR bbox: {bbox2_ogr}")

    try:
        # Unpack coordinates
        bbox1_x_min, bbox1_x_max, bbox1_y_min, bbox1_y_max = map(float, bbox1_ogr)
        bbox2_x_min, bbox2_x_max, bbox2_y_min, bbox2_y_max = map(float, bbox2_ogr)

        # Handle infinite values
        if any(np.isinf(val) for val in bbox1_ogr + bbox2_ogr):
            # If any value is infinite, at least one box extends infinitely
            # and they must intersect unless separated by an infinite gap
            return True

        # Check for non-intersection conditions
        if (bbox2_x_min > bbox1_x_max or
            bbox2_y_min > bbox1_y_max or
            bbox2_x_max < bbox1_x_min or
            bbox2_y_max < bbox1_y_min):
            return False

        return True

    except (TypeError, ValueError) as e:
        raise ValueError(f"Error comparing bboxes: {str(e)}") from e


def _check_bboxes_within(
    bbox1_ogr: List[Union[int, float]],
    bbox2_ogr: List[Union[int, float]]
) -> bool:
    """Checks if bbox1_ogr is completely within bbox2_ogr.

    Parameters
    ----------
    bbox1_ogr : List[Union[int, float]]
        An OGR formatted bbox to check if it's contained.
        `[x_min, x_max, y_min, y_max]`
    bbox2_ogr : List[Union[int, float]]
        An OGR formatted bbox to check against.
        `[x_min, x_max, y_min, y_max]`

    Returns
    -------
    bool
        True if bbox1_ogr is completely within bbox2_ogr, False otherwise.

    Raises
    ------
    TypeError
        If either bbox is None or not a list/tuple
    ValueError
        If either bbox is not a valid OGR formatted bbox

    Examples
    --------
    >>> _check_bboxes_within([1, 2, 1, 2], [0, 3, 0, 3])
    True
    >>> _check_bboxes_within([0, 4, 0, 4], [1, 3, 1, 3])
    False
    >>> _check_bboxes_within(None, [0, 1, 0, 1])
    Raises TypeError
    >>> _check_bboxes_within([0, 1, None, 1], [0, 1, 0, 1])
    Raises ValueError
    """
    # Input validation
    if bbox1_ogr is None or bbox2_ogr is None:
        raise TypeError("bbox arguments cannot be None")

    if not isinstance(bbox1_ogr, (list, tuple)) or not isinstance(bbox2_ogr, (list, tuple)):
        raise TypeError("bbox arguments must be lists or tuples")

    # Validate bbox formats
    if not _check_is_valid_bbox(bbox1_ogr):
        raise ValueError(f"bbox1_ogr is not a valid OGR bbox: {bbox1_ogr}")

    if not _check_is_valid_bbox(bbox2_ogr):
        raise ValueError(f"bbox2_ogr is not a valid OGR bbox: {bbox2_ogr}")

    try:
        # Convert all values to float for consistent comparison
        bbox1_x_min, bbox1_x_max, bbox1_y_min, bbox1_y_max = map(float, bbox1_ogr)
        bbox2_x_min, bbox2_x_max, bbox2_y_min, bbox2_y_max = map(float, bbox2_ogr)

        # Handle infinite values
        if any(np.isinf(val) for val in bbox1_ogr):
            return False  # A bbox with infinite bounds cannot be contained

        if all(np.isinf(val) for val in bbox2_ogr):
            return True  # An infinite bbox contains everything

        # Regular comparison
        return (bbox1_x_min >= bbox2_x_min and
                bbox1_x_max <= bbox2_x_max and
                bbox1_y_min >= bbox2_y_min and
                bbox1_y_max <= bbox2_y_max)

    except (TypeError, ValueError) as e:
        raise ValueError(f"Error comparing bboxes: {str(e)}") from e
