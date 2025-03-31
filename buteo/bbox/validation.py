"""Internal validation functions for bounding boxes and geotransforms."""

# Standard library
from typing import Union, Sequence

# External
import numpy as np

# Type Aliases for clarity
BboxType = Sequence[Union[int, float]]
GeoTransformType = Sequence[Union[int, float]]


def _check_is_valid_bbox(bbox_ogr: BboxType) -> bool:
    """Checks if a bounding box is valid in OGR format.

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

    Returns
    -------
    bool
        True if the bbox is valid, False otherwise.

    Examples
    --------
    >>> _check_is_valid_bbox([0, 1, 0, 1])
    True
    >>> _check_is_valid_bbox((0, 1, 0, 1)) # Tuple is also valid
    True
    >>> _check_is_valid_bbox([1, 0, 0, 1]) # x_min > x_max (dateline crossing allowed if latlng)
    True # Note: This function alone doesn't check latlng bounds
    >>> _check_is_valid_bbox([170, -170, -10, 10]) # Valid dateline crossing
    True
    >>> _check_is_valid_bbox([0, 1, 1, 0]) # y_min > y_max
    False
    >>> _check_is_valid_bbox([0, 1, 0]) # Invalid length
    False
    >>> _check_is_valid_bbox([0, 1, None, 1]) # Contains None
    False
    >>> _check_is_valid_bbox([0, 1, np.nan, 1]) # Contains NaN
    False
    >>> _check_is_valid_bbox([-np.inf, np.inf, -90, 90]) # Infinite values are valid
    True
    """

    valid = False
    if isinstance(bbox_ogr, (list, tuple)) and len(bbox_ogr) == 4:
        # Check if all values are numeric (int or float) and not None
        is_numeric = all(isinstance(val, (int, float)) for val in bbox_ogr)

        if is_numeric:
            # Check for NaN values explicitly
            has_nan = any(np.isnan(v) for v in bbox_ogr)

            if not has_nan:
                # Unpack values
                x_min, x_max, y_min, y_max = bbox_ogr

                # Allow infinite values - if any are infinite, only y order matters
                has_inf = any(np.isinf(v) for v in bbox_ogr)
                if has_inf:
                    # If infinite, only check that y_min <= y_max
                    if y_min <= y_max:
                        valid = True
                else:
                    # Check if y_min <= y_max. x_min <= x_max is generally true,
                    # but allow x_min > x_max for potential dateline crossing cases
                    # (further validation needed in _check_is_valid_bbox_latlng).
                    if y_min <= y_max:
                        valid = True

    return valid


def _check_is_valid_bbox_latlng(bbox_ogr_latlng: BboxType) -> bool:
    """Checks if a bbox is valid and represents geographic lat/long coordinates.

    A valid OGR formatted bbox in lat/long coordinates has the form:
    `[longitude_min, longitude_max, latitude_min, latitude_max]`
    where:
        - It must be a valid bbox according to `_check_is_valid_bbox`.
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
    >>> _check_is_valid_bbox_latlng([-180, 180, -90, 90])
    True
    >>> _check_is_valid_bbox_latlng([170, -170, -10, 10]) # Dateline crossing
    True
    >>> _check_is_valid_bbox_latlng([-10, 10, -10, 10])
    True
    >>> _check_is_valid_bbox_latlng([-181, 180, -90, 90]) # Invalid longitude
    False
    >>> _check_is_valid_bbox_latlng([0, 1, -91, 90]) # Invalid latitude
    False
    >>> _check_is_valid_bbox_latlng([0, 1, 0, 1]) # Valid bbox, also valid lat/lng
    True
    >>> _check_is_valid_bbox_latlng(None) # Input validation handled by _check_is_valid_bbox
    False
    """
    # First, check if it's a generally valid bbox (handles None, length, numeric, NaN, y-order)
    # Note: _check_is_valid_bbox now allows x_min > x_max
    if not _check_is_valid_bbox(bbox_ogr_latlng):
        return False

    # Unpack values
    lng_min, lng_max, lat_min, lat_max = bbox_ogr_latlng

    # Check longitude bounds (-180 to 180)
    if not (-180 <= lng_min <= 180 and -180 <= lng_max <= 180):
        return False

    # Check latitude bounds (-90 to 90)
    if not (-90 <= lat_min <= 90 and -90 <= lat_max <= 90):
        return False

    # Check y-order again (already done in _check_is_valid_bbox, but explicit here)
    if lat_min > lat_max:
        return False

    # x_min <= x_max check is implicitly handled by bounds check unless crossing dateline
    # If not crossing dateline, _check_is_valid_bbox ensures x_min <= x_max is not needed here.
    # If crossing dateline, x_min > x_max is allowed if within bounds.

    return True


def _check_is_valid_geotransform(geotransform: GeoTransformType) -> bool:
    """Checks if a GDAL geotransform is valid.

    A valid geotransform is a sequence (list or tuple) of 6 numbers:
    `[origin_x, pixel_width, row_skew, origin_y, column_skew, pixel_height]`

    Validation Rules:
        - Must be a sequence of 6 numbers.
        - All values must be numeric (int or float).
        - None or NaN values are not allowed.
        - `pixel_width` (index 1) cannot be zero.
        - `pixel_height` (index 5) cannot be zero.

    Parameters
    ----------
    geotransform : GeoTransformType
        A GDAL formatted geotransform sequence.

    Returns
    -------
    bool
        True if the geotransform is valid, False otherwise.

    Examples
    --------
    >>> _check_is_valid_geotransform([0, 1, 0, 10, 0, -1])
    True
    >>> _check_is_valid_geotransform((0.0, 1.0, 0.0, 10.0, 0.0, -1.0)) # Tuple is valid
    True
    >>> _check_is_valid_geotransform(None)
    False
    >>> _check_is_valid_geotransform([0, 1, 0, 10, 0]) # Wrong length
    False
    >>> _check_is_valid_geotransform([0, 1, 0, 10, 0, None]) # Contains None
    False
    >>> _check_is_valid_geotransform([0, 1, 0, 10, 0, np.nan]) # Contains NaN
    False
    >>> _check_is_valid_geotransform([0, 0, 0, 10, 0, -1]) # Zero pixel width
    False
    >>> _check_is_valid_geotransform([0, 1, 0, 10, 0, 0]) # Zero pixel height
    False
    """
    # Check if input is a sequence (list or tuple)
    if not isinstance(geotransform, (list, tuple)):
        return False

    # Check length
    if len(geotransform) != 6:
        return False

    # Check if all values are numeric (int or float) and not None or NaN
    for val in geotransform:
        if not isinstance(val, (int, float)) or np.isnan(val):
            return False

    # Verify pixel width (index 1) is not zero
    if abs(float(geotransform[1])) < 1e-15: # Use tolerance for float comparison
        return False

    # Verify pixel height (index 5) is not zero
    if abs(float(geotransform[5])) < 1e-15: # Use tolerance for float comparison
        return False

    return True


def _check_bboxes_intersect(
    bbox1_ogr: BboxType,
    bbox2_ogr: BboxType
) -> bool:
    """Checks if two OGR formatted bounding boxes intersect.

    Parameters
    ----------
    bbox1_ogr : BboxType
        The first OGR formatted bbox: `[x_min, x_max, y_min, y_max]`.
    bbox2_ogr : BboxType
        The second OGR formatted bbox: `[x_min, x_max, y_min, y_max]`.

    Returns
    -------
    bool
        True if the bounding boxes intersect, False otherwise.

    Raises
    ------
    ValueError
        If either input is not a valid OGR formatted bbox according
        to `_check_is_valid_bbox`.

    Notes
    -----
    Intersection is defined as having any overlapping area, including
    sharing just an edge or a corner. Bboxes containing infinite values
    are considered to intersect unless separated by an infinite gap.
    Handles dateline crossing (where x_min > x_max).

    Examples
    --------
    >>> _check_bboxes_intersect([0, 1, 0, 1], [0.5, 1.5, 0.5, 1.5]) # Overlap
    True
    >>> _check_bboxes_intersect([0, 1, 0, 1], [1, 2, 0, 1]) # Touch edge
    True
    >>> _check_bboxes_intersect([170, -170, 0, 1], [-175, -172, 0, 1]) # Dateline overlap
    True
    >>> _check_bboxes_intersect([170, -170, 0, 1], [160, 165, 0, 1]) # Dateline no overlap
    False
    >>> _check_bboxes_intersect([0, 1, 0, 1], [2, 3, 2, 3]) # Separate
    False
    >>> _check_bboxes_intersect([0, 1, 0, 1], [-np.inf, np.inf, -np.inf, np.inf]) # Infinite box
    True
    >>> _check_bboxes_intersect([0, 1, 0, 1], None)
    Raises ValueError: bbox2_ogr is not a valid OGR bbox: None
    """
    # Validate inputs first
    if not _check_is_valid_bbox(bbox1_ogr):
        raise ValueError(f"bbox1_ogr is not a valid OGR bbox: {bbox1_ogr}")
    if not _check_is_valid_bbox(bbox2_ogr):
        raise ValueError(f"bbox2_ogr is not a valid OGR bbox: {bbox2_ogr}")

    # Unpack coordinates (already validated as numeric)
    bbox1_x_min, bbox1_x_max, bbox1_y_min, bbox1_y_max = bbox1_ogr
    bbox2_x_min, bbox2_x_max, bbox2_y_min, bbox2_y_max = bbox2_ogr

    # Handle infinite values - they always intersect unless separated by infinity
    if any(np.isinf(v) for v in bbox1_ogr) or any(np.isinf(v) for v in bbox2_ogr):
        # Check for infinite separation
        if (bbox1_x_max == -np.inf and bbox2_x_min == np.inf) or \
           (bbox2_x_max == -np.inf and bbox1_x_min == np.inf) or \
           (bbox1_y_max == -np.inf and bbox2_y_min == np.inf) or \
           (bbox2_y_max == -np.inf and bbox1_y_min == np.inf):
            return False
        return True

    # Check y-overlap first (simpler)
    y_overlap = not (bbox1_y_max < bbox2_y_min or bbox1_y_min > bbox2_y_max)
    if not y_overlap:
        return False

    # Check x-overlap, considering dateline crossing
    bbox1_crosses_dateline = bbox1_x_min > bbox1_x_max
    bbox2_crosses_dateline = bbox2_x_min > bbox2_x_max

    if bbox1_crosses_dateline and bbox2_crosses_dateline:
        # Both cross dateline - they always overlap in x if y overlaps
        return True
    elif bbox1_crosses_dateline:
        # bbox1 crosses, bbox2 does not. Overlap if bbox2 overlaps with either part of bbox1
        return (bbox2_x_max >= bbox1_x_min or bbox2_x_min <= bbox1_x_max)
    elif bbox2_crosses_dateline:
        # bbox2 crosses, bbox1 does not. Overlap if bbox1 overlaps with either part of bbox2
        return (bbox1_x_max >= bbox2_x_min or bbox1_x_min <= bbox2_x_max)
    else:
        # Neither crosses dateline - standard overlap check
        return not (bbox1_x_max < bbox2_x_min or bbox1_x_min > bbox2_x_max)


def _check_bboxes_within(
    bbox1_ogr: BboxType,
    bbox2_ogr: BboxType
) -> bool:
    """Checks if the first bounding box (bbox1_ogr) is completely within
    the second bounding box (bbox2_ogr).

    Parameters
    ----------
    bbox1_ogr : BboxType
        The OGR formatted bbox to check if it's contained:
        `[x_min, x_max, y_min, y_max]`.
    bbox2_ogr : BboxType
        The OGR formatted bbox to check against (the container):
        `[x_min, x_max, y_min, y_max]`.

    Returns
    -------
    bool
        True if bbox1_ogr is completely within bbox2_ogr, False otherwise.

    Raises
    ------
    ValueError
        If either input is not a valid OGR formatted bbox according
        to `_check_is_valid_bbox`.

    Notes
    -----
    A bbox with infinite bounds cannot be contained within any finite bbox.
    A finite bbox is always contained within a bbox with infinite bounds.
    Handles dateline crossing for the container bbox (bbox2_ogr).

    Examples
    --------
    >>> _check_bboxes_within([1, 2, 1, 2], [0, 3, 0, 3]) # Fully contained
    True
    >>> _check_bboxes_within([175, -175, 0, 1], [170, -170, -1, 2]) # Contained across dateline
    True
    >>> _check_bboxes_within([170, -170, 0, 1], [175, -175, -1, 2]) # Container smaller across dateline
    False
    >>> _check_bboxes_within([0, 4, 0, 4], [1, 3, 1, 3]) # Container is smaller
    False
    >>> _check_bboxes_within([-np.inf, 1, 0, 1], [0, 3, 0, 3]) # Infinite cannot be contained
    False
    >>> _check_bboxes_within([1, 2, 1, 2], [-np.inf, np.inf, -np.inf, np.inf]) # Contained in infinite
    True
    >>> _check_bboxes_within([0, 1, 0, 1], None)
    Raises ValueError: bbox2_ogr is not a valid OGR bbox: None
    """
    # Validate inputs first
    if not _check_is_valid_bbox(bbox1_ogr):
        raise ValueError(f"bbox1_ogr is not a valid OGR bbox: {bbox1_ogr}")
    if not _check_is_valid_bbox(bbox2_ogr):
        raise ValueError(f"bbox2_ogr is not a valid OGR bbox: {bbox2_ogr}")

    # Unpack coordinates (already validated as numeric)
    bbox1_x_min, bbox1_x_max, bbox1_y_min, bbox1_y_max = bbox1_ogr
    bbox2_x_min, bbox2_x_max, bbox2_y_min, bbox2_y_max = bbox2_ogr

    # Handle infinite values
    if any(np.isinf(v) for v in bbox1_ogr):
        return False  # An infinite box cannot be contained

    if all(np.isinf(v) for v in bbox2_ogr):
        return True  # Any finite box is contained within an infinite box
    elif any(np.isinf(v) for v in bbox2_ogr):
        # If container has some infinite bounds but bbox1 is finite,
        # containment depends on the finite bounds.
        pass # Continue to standard checks

    # Standard y-containment check
    y_within = bbox1_y_min >= bbox2_y_min and bbox1_y_max <= bbox2_y_max
    if not y_within:
        return False

    # Standard x-containment check (handles non-dateline crossing)
    x_within_standard = bbox1_x_min >= bbox2_x_min and bbox1_x_max <= bbox2_x_max

    # Handle dateline crossing for the container (bbox2)
    if bbox2_x_min > bbox2_x_max: # bbox2 crosses dateline
        # bbox1 must be entirely within one of the two parts of bbox2
        x_within_dateline = (bbox1_x_min >= bbox2_x_min and bbox1_x_max <= 180.0) or \
                            (bbox1_x_min >= -180.0 and bbox1_x_max <= bbox2_x_max)
        return x_within_dateline and y_within
    else: # bbox2 does not cross dateline
        # bbox1 also cannot cross dateline if it's within bbox2
        if bbox1_x_min > bbox1_x_max:
            return False
        return x_within_standard and y_within
