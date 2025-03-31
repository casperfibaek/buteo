"""Internal bounding box operation functions."""

# Standard library
from typing import List, Union, Dict, Tuple, Sequence

# External
import numpy as np

# Internal
from buteo.bbox.validation import (_check_is_valid_bbox, _check_is_valid_geotransform,
                                  _check_bboxes_intersect)
from buteo.utils import utils_base

# Type Aliases (consistent with validation.py)
BboxType = Sequence[Union[int, float]]
GeoTransformType = Sequence[Union[int, float]]


def _get_pixel_offsets(
    geotransform: GeoTransformType,
    bbox_ogr: BboxType,
) -> Tuple[int, int, int, int]:
    """Calculates pixel offsets for a bounding box within a geotransform grid.

    Parameters
    ----------
    geotransform : GeoTransformType
        A GDAL formatted geotransform:
        `[origin_x, pixel_width, row_skew, origin_y, column_skew, pixel_height]`.
        `pixel_height` is typically negative for north-up images.
    bbox_ogr : BboxType
        An OGR formatted bounding box: `[x_min, x_max, y_min, y_max]`.

    Returns
    -------
    Tuple[int, int, int, int]
        Pixel offsets as `(x_start, y_start, x_size, y_size)`:
            x_start : int
                Starting pixel column index.
            y_start : int
                Starting pixel row index.
            x_size : int
                Number of pixels in the x direction (width).
            y_size : int
                Number of pixels in the y direction (height).

    Raises
    ------
    ValueError
        If `geotransform` or `bbox_ogr` are invalid, if pixel width/height
        in geotransform are zero or near-zero, or if calculation results
        in overflow or floating point errors.

    Examples
    --------
    >>> gt = [0.0, 1.0, 0.0, 10.0, 0.0, -1.0]
    >>> bbox = [2.0, 4.0, 4.0, 8.0]
    >>> _get_pixel_offsets(gt, bbox)
    (2, 2, 2, 4)
    >>> gt_zero_pixel = [0.0, 0.0, 0.0, 10.0, 0.0, -1.0]
    >>> _get_pixel_offsets(gt_zero_pixel, bbox)
    Raises ValueError: Pixel width and height cannot be zero or near-zero
    """

    # Input validation using imported validation functions
    if not _check_is_valid_geotransform(geotransform):
        raise ValueError(f"Invalid geotransform provided: {geotransform}")

    if not _check_is_valid_bbox(bbox_ogr):
        raise ValueError(f"Invalid OGR bounding box provided: {bbox_ogr}")

    # Unpack values, converting to float for calculations
    x_min, x_max, y_min, y_max = map(float, bbox_ogr)
    origin_x = float(geotransform[0])
    origin_y = float(geotransform[3])
    pixel_width = float(geotransform[1])
    pixel_height = float(geotransform[5])

    # Pixel dimensions were already checked for non-zero in _check_is_valid_geotransform

    # Calculate pixel offsets, rounding to nearest integer
    try:
        x_start = int(np.rint((x_min - origin_x) / pixel_width))
        y_start = int(np.rint((y_max - origin_y) / pixel_height)) # Use y_max for top-left origin
        x_size = int(np.rint((x_max - x_min) / pixel_width))
        y_size = int(np.rint((y_min - y_max) / pixel_height)) # Use y_min - y_max for height
    except (OverflowError, FloatingPointError, ZeroDivisionError) as e:
        # ZeroDivisionError should be caught by validation, but included for safety
        raise ValueError(f"Error calculating pixel offsets: {str(e)}") from e

    # Ensure calculated sizes are non-negative
    x_size = abs(x_size)
    y_size = abs(y_size)

    return (x_start, y_start, x_size, y_size)


def _get_bbox_from_geotransform(
    geotransform: GeoTransformType,
    raster_x_size: int,
    raster_y_size: int,
) -> List[float]:
    """Calculates the OGR bounding box from a geotransform and raster dimensions.

    Parameters
    ----------
    geotransform : GeoTransformType
        A GDAL formatted geotransform:
        `[origin_x, pixel_width, row_skew, origin_y, column_skew, pixel_height]`.
    raster_x_size : int
        The number of pixels (columns) in the x direction (width).
    raster_y_size : int
        The number of pixels (rows) in the y direction (height).

    Returns
    -------
    List[float]
        An OGR formatted bounding box: `[x_min, x_max, y_min, y_max]`.

    Raises
    ------
    ValueError
        If `geotransform` is invalid or `raster_x_size`/`raster_y_size` are negative.
    TypeError
        If `raster_x_size` or `raster_y_size` are not integers.

    Examples
    --------
    >>> gt = [0.0, 1.0, 0.0, 10.0, 0.0, -1.0]
    >>> _get_bbox_from_geotransform(gt, 5, 5)
    [0.0, 5.0, 5.0, 10.0]
    >>> _get_bbox_from_geotransform(gt, 100, 200)
    [0.0, 100.0, -190.0, 10.0]
    >>> _get_bbox_from_geotransform(gt, 0, 0) # Zero size raster
    [0.0, 0.0, 10.0, 10.0]
    >>> _get_bbox_from_geotransform(gt, -5, 5)
    Raises ValueError: raster sizes cannot be negative
    """
    # Input validation
    if not isinstance(raster_x_size, int) or not isinstance(raster_y_size, int):
        raise TypeError("raster_x_size and raster_y_size must be integers.")
    if raster_x_size < 0 or raster_y_size < 0:
        raise ValueError("raster sizes cannot be negative.")
    if not _check_is_valid_geotransform(geotransform):
        raise ValueError(f"Invalid geotransform provided: {geotransform}")

    # Unpack geotransform, converting to float
    origin_x = float(geotransform[0])
    pixel_width = float(geotransform[1])
    origin_y = float(geotransform[3])
    pixel_height = float(geotransform[5]) # Note: usually negative

    # Calculate extent corners
    x_min = origin_x
    y_max = origin_y
    x_max = origin_x + (raster_x_size * pixel_width)
    y_min = origin_y + (raster_y_size * pixel_height)

    # Ensure correct OGR order [x_min, x_max, y_min, y_max]
    # Handle cases where pixel width or height might be positive/negative
    if x_max < x_min:
        x_min, x_max = x_max, x_min
    if y_max < y_min: # This happens if pixel_height is positive
        y_min, y_max = y_max, y_min

    return [x_min, x_max, y_min, y_max]


def _get_intersection_bboxes(
    bbox1_ogr: BboxType,
    bbox2_ogr: BboxType,
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
    >>> _get_intersection_bboxes([0, 2, 0, 2], [1, 3, 1, 3])
    [1.0, 2.0, 1.0, 2.0]
    >>> _get_intersection_bboxes([0, 1, 0, 1], [1, 2, 1, 2]) # Corner touch
    [1.0, 1.0, 1.0, 1.0]
    >>> _get_intersection_bboxes([0, 1, 0, 1], [2, 3, 2, 3]) # No intersection
    Raises ValueError: Bounding boxes do not intersect
    >>> _get_intersection_bboxes([0, 1, 0, 1], None)
    Raises ValueError: Invalid OGR bounding box provided: None
    """
    # Input validation (raises ValueError if invalid)
    if not _check_is_valid_bbox(bbox1_ogr):
        raise ValueError(f"Invalid OGR bounding box provided: {bbox1_ogr}")
    if not _check_is_valid_bbox(bbox2_ogr):
        raise ValueError(f"Invalid OGR bounding box provided: {bbox2_ogr}")

    # Check for intersection (raises ValueError if they don't intersect)
    if not _check_bboxes_intersect(bbox1_ogr, bbox2_ogr):
        raise ValueError("Bounding boxes do not intersect")

    # Convert to float for calculation
    bbox1_x_min, bbox1_x_max, bbox1_y_min, bbox1_y_max = map(float, bbox1_ogr)
    bbox2_x_min, bbox2_x_max, bbox2_y_min, bbox2_y_max = map(float, bbox2_ogr)

    # Calculate intersection bounds
    intersect_x_min = max(bbox1_x_min, bbox2_x_min)
    intersect_x_max = min(bbox1_x_max, bbox2_x_max)
    intersect_y_min = max(bbox1_y_min, bbox2_y_min)
    intersect_y_max = min(bbox1_y_max, bbox2_y_max)

    # Result should always be valid if inputs are valid and intersect
    return [intersect_x_min, intersect_x_max, intersect_y_min, intersect_y_max]


def _get_union_bboxes(
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
    >>> _get_union_bboxes([0, 1, 0, 1], [1, 2, 1, 2])
    [0.0, 2.0, 0.0, 2.0]
    >>> _get_union_bboxes([-10, 0, -10, 0], [0, 10, 0, 10])
    [-10.0, 10.0, -10.0, 10.0]
    >>> _get_union_bboxes([0, 1, 0, 1], [-np.inf, 0, 0, 1])
    [-inf, 1.0, 0.0, 1.0]
    >>> _get_union_bboxes([0, 1, 0, 1], None)
    Raises ValueError: Invalid OGR bounding box provided: None
    """
    # Input validation (raises ValueError if invalid)
    if not _check_is_valid_bbox(bbox1_ogr):
        raise ValueError(f"Invalid OGR bounding box provided: {bbox1_ogr}")
    if not _check_is_valid_bbox(bbox2_ogr):
        raise ValueError(f"Invalid OGR bounding box provided: {bbox2_ogr}")

    # Convert to float for calculation
    bbox1_x_min, bbox1_x_max, bbox1_y_min, bbox1_y_max = map(float, bbox1_ogr)
    bbox2_x_min, bbox2_x_max, bbox2_y_min, bbox2_y_max = map(float, bbox2_ogr)

    # Calculate union bounds, handling potential infinite values
    union_x_min = min(bbox1_x_min, bbox2_x_min)
    union_x_max = max(bbox1_x_max, bbox2_x_max)
    union_y_min = min(bbox1_y_min, bbox2_y_min)
    union_y_max = max(bbox1_y_max, bbox2_y_max)

    # Result is always valid if inputs are valid
    return [union_x_min, union_x_max, union_y_min, union_y_max]


def _get_aligned_bbox_to_pixel_size(
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
    >>> _get_aligned_bbox_to_pixel_size(ref_bbox, target_bbox, 1.0, -1.0)
    [1.0, 4.0, 1.0, 4.0]
    >>> # Align with 0.5x0.5 pixel size
    >>> _get_aligned_bbox_to_pixel_size(ref_bbox, target_bbox, 0.5, -0.5)
    [1.0, 4.0, 1.0, 4.0] # Snaps to nearest 0.5 outwards
    >>> _get_aligned_bbox_to_pixel_size(ref_bbox, target_bbox, 0.0, -1.0)
    Raises ValueError: pixel_width must be positive, got: 0.0
    """
    # Input validation
    if not isinstance(pixel_width, (int, float)) or not isinstance(pixel_height, (int, float)):
        raise TypeError("Pixel dimensions must be numeric.")
    if not _check_is_valid_bbox(bbox_to_align_to_ogr):
        raise ValueError(f"Invalid reference bbox provided: {bbox_to_align_to_ogr}")
    if not _check_is_valid_bbox(bbox_to_be_aligned_ogr):
        raise ValueError(f"Invalid target bbox provided: {bbox_to_be_aligned_ogr}")

    # Validate pixel dimensions
    if float(pixel_width) <= 0:
        raise ValueError(f"pixel_width must be positive, got: {pixel_width}")
    if abs(float(pixel_height)) < 1e-15: # Check against zero with tolerance
        raise ValueError("pixel_height cannot be zero.")

    # Convert inputs to float
    ref_x_min, _, _, ref_y_max = map(float, bbox_to_align_to_ogr) # ref_y_min is unused
    target_x_min, target_x_max, target_y_min, target_y_max = map(float, bbox_to_be_aligned_ogr)
    # Use pixel_width and pixel_height directly to reduce local variables
    pixel_width_f = float(pixel_width)
    pixel_height_f = float(pixel_height) # Keep sign for y-alignment calculation

    # Calculate aligned coordinates by snapping outwards to the grid
    # defined by the reference bbox origin (ref_x_min, ref_y_max for top-left)
    # and pixel dimensions.

    # Align x_min: Find the largest grid line <= target_x_min
    aligned_x_min = ref_x_min + np.floor((target_x_min - ref_x_min) / pixel_width_f) * pixel_width_f

    # Align x_max: Find the smallest grid line >= target_x_max
    aligned_x_max = ref_x_min + np.ceil((target_x_max - ref_x_min) / pixel_width_f) * pixel_width_f

    # Align y_max (top edge): Find the smallest grid line >= target_y_max (using negative pixel_height_f)
    # Grid lines are ref_y_max + n * pixel_height_f. We want ref_y_max + n * pixel_height_f >= target_y_max
    # n * pixel_height_f >= target_y_max - ref_y_max
    # n <= (target_y_max - ref_y_max) / pixel_height_f (since pixel_height_f is negative, inequality flips)
    # So, n = floor((target_y_max - ref_y_max) / pixel_height_f)
    aligned_y_max = ref_y_max + np.floor((target_y_max - ref_y_max) / pixel_height_f) * pixel_height_f

    # Align y_min (bottom edge): Find the largest grid line <= target_y_min
    # Grid lines are ref_y_max + n * pixel_height_f. We want ref_y_max + n * pixel_height_f <= target_y_min
    # n * pixel_height_f <= target_y_min - ref_y_max
    # n >= (target_y_min - ref_y_max) / pixel_height_f (inequality flips)
    # So, n = ceil((target_y_min - ref_y_max) / pixel_height_f)
    aligned_y_min = ref_y_max + np.ceil((target_y_min - ref_y_max) / pixel_height_f) * pixel_height_f


    # Check for NaN or infinite values in result
    result = [aligned_x_min, aligned_x_max, aligned_y_min, aligned_y_max]
    if any(np.isnan(val) or np.isinf(val) for val in result):
        raise ValueError("Alignment resulted in NaN or infinite values.")

    # Ensure correct order just in case (though alignment logic should handle it)
    if result[0] > result[1]: result[0], result[1] = result[1], result[0]
    if result[2] > result[3]: result[2], result[3] = result[3], result[2]

    return result


def _get_gdal_bbox_from_ogr_bbox(
    bbox_ogr: BboxType
) -> List[float]:
    """Converts an OGR formatted bbox to a GDAL warp tool formatted bbox.

    OGR format: `[x_min, x_max, y_min, y_max]`
    GDAL format: `[x_min, y_min, x_max, y_max]`

    Parameters
    ----------
    bbox_ogr : BboxType
        An OGR formatted bbox: `[x_min, x_max, y_min, y_max]`.

    Returns
    -------
    List[float]
        A GDAL formatted bbox: `[x_min, y_min, x_max, y_max]`.

    Raises
    ------
    ValueError
        If `bbox_ogr` is not a valid OGR formatted bbox.

    Examples
    --------
    >>> _get_gdal_bbox_from_ogr_bbox([0.0, 10.0, 20.0, 30.0])
    [0.0, 20.0, 10.0, 30.0]
    >>> _get_gdal_bbox_from_ogr_bbox(None)
    Raises ValueError: Invalid OGR bounding box provided: None
    """
    # Input validation (raises ValueError if invalid)
    if not _check_is_valid_bbox(bbox_ogr):
        raise ValueError(f"Invalid OGR bounding box provided: {bbox_ogr}")

    # Unpack and reorder, converting to float
    x_min, x_max, y_min, y_max = map(float, bbox_ogr)
    return [x_min, y_min, x_max, y_max]


def _get_ogr_bbox_from_gdal_bbox(
    bbox_gdal: BboxType,
) -> List[float]:
    """Converts a GDAL warp tool formatted bbox to an OGR formatted bbox.

    GDAL format: `[x_min, y_min, x_max, y_max]`
    OGR format: `[x_min, x_max, y_min, y_max]`

    Parameters
    ----------
    bbox_gdal : BboxType
        A GDAL formatted bbox: `[x_min, y_min, x_max, y_max]`.

    Returns
    -------
    List[float]
        An OGR formatted bbox: `[x_min, x_max, y_min, y_max]`.

    Raises
    ------
    ValueError
        If `bbox_gdal` is not a sequence of 4 numeric, non-NaN values,
        or if `x_min > x_max` or `y_min > y_max` after conversion.

    Examples
    --------
    >>> _get_ogr_bbox_from_gdal_bbox([0.0, 20.0, 10.0, 30.0])
    [0.0, 10.0, 20.0, 30.0]
    >>> _get_ogr_bbox_from_gdal_bbox(None)
    Raises ValueError: Input must be a sequence of 4 numbers.
    >>> _get_ogr_bbox_from_gdal_bbox([0, 0, 1, None])
    Raises ValueError: Input sequence cannot contain None or NaN values.
    """
    # Basic input validation
    if not isinstance(bbox_gdal, (list, tuple)) or len(bbox_gdal) != 4:
        raise ValueError("Input must be a sequence of 4 numbers.")
    if any(not isinstance(v, (int, float)) or np.isnan(v) for v in bbox_gdal):
        raise ValueError("Input sequence cannot contain None or NaN values.")

    # Unpack and reorder, converting to float
    x_min, y_min, x_max, y_max = map(float, bbox_gdal)
    ogr_bbox = [x_min, x_max, y_min, y_max]

    # Validate the resulting OGR bbox
    if not _check_is_valid_bbox(ogr_bbox):
        # This primarily checks x_min <= x_max and y_min <= y_max
        raise ValueError(f"Converted OGR bbox is invalid: {ogr_bbox}")

    return ogr_bbox


def _get_geotransform_from_bbox(
    bbox_ogr: BboxType,
    raster_x_size: int,
    raster_y_size: int,
) -> List[float]:
    """Calculates a GDAL GeoTransform from an OGR bounding box and raster dimensions.

    Assumes a north-up orientation (negative pixel height) and no rotation/skew.

    Parameters
    ----------
    bbox_ogr : BboxType
        An OGR formatted bbox: `[x_min, x_max, y_min, y_max]`.
    raster_x_size : int
        The number of pixels (columns) in the x direction (width).
    raster_y_size : int
        The number of pixels (rows) in the y direction (height).

    Returns
    -------
    List[float]
        A GDAL GeoTransform:
        `[origin_x, pixel_width, 0.0, origin_y, 0.0, pixel_height]`.

    Raises
    ------
    ValueError
        If `bbox_ogr` is invalid, `raster_x_size` or `raster_y_size`
        are not positive integers, or if calculation leads to zero pixel size.
    TypeError
        If raster dimensions are not integers.

    Examples
    --------
    >>> bbox = [0.0, 100.0, 50.0, 150.0] # 100x100 extent
    >>> _get_geotransform_from_bbox(bbox, 100, 100) # 100x100 pixels
    [0.0, 1.0, 0.0, 150.0, 0.0, -1.0]
    >>> _get_geotransform_from_bbox(bbox, 200, 50) # 200x50 pixels
    [0.0, 0.5, 0.0, 150.0, 0.0, -2.0]
    >>> _get_geotransform_from_bbox(bbox, 0, 100)
    Raises ValueError: raster dimensions must be positive
    """
    # Input validation
    if not isinstance(raster_x_size, int) or not isinstance(raster_y_size, int):
        raise TypeError("raster dimensions must be integers.")
    if raster_x_size <= 0 or raster_y_size <= 0:
        raise ValueError("raster dimensions must be positive.")
    if not _check_is_valid_bbox(bbox_ogr):
        raise ValueError(f"Invalid OGR bounding box provided: {bbox_ogr}")

    # Unpack bbox, converting to float
    x_min, x_max, y_min, y_max = map(float, bbox_ogr)

    # Calculate pixel dimensions
    # Use float division for potentially non-integer pixel sizes
    pixel_width = (x_max - x_min) / float(raster_x_size)
    pixel_height = (y_max - y_min) / float(raster_y_size) # This will be positive

    # Check for zero pixel dimensions (already checked raster sizes > 0)
    # This catches cases where bbox has zero width or height
    epsilon = 1e-15
    if abs(pixel_width) < epsilon or abs(pixel_height) < epsilon:
        raise ValueError("Calculated pixel width or height is zero or near-zero.")

    # Construct geotransform: [origin_x, pixel_w, skew_x, origin_y, skew_y, pixel_h]
    # Origin is top-left corner (x_min, y_max)
    # Pixel height needs to be negative for north-up
    geotransform = [
        x_min,
        pixel_width,
        0.0,
        y_max,
        0.0,
        utils_base._ensure_negative(pixel_height) # Ensure negative height
    ]

    return geotransform


def _get_sub_geotransform(
    geotransform: GeoTransformType,
    bbox_ogr: BboxType,
) -> Dict[str, Union[List[float], int]]:
    """Creates a new GeoTransform and raster dimensions corresponding to a
    sub-region defined by an OGR bounding box within an existing geotransform grid.

    Parameters
    ----------
    geotransform : GeoTransformType
        The original GDAL geotransform of the larger grid:
        `[origin_x, pixel_width, row_skew, origin_y, column_skew, pixel_height]`.
    bbox_ogr : BboxType
        The OGR formatted bbox defining the sub-region:
        `[x_min, x_max, y_min, y_max]`.

    Returns
    -------
    Dict[str, Union[List[float], int]]
        A dictionary containing:
            "Transform" : List[float]
                The new geotransform for the sub-region.
            "RasterXSize" : int
                The width of the sub-region in pixels.
            "RasterYSize" : int
                The height of the sub-region in pixels.

    Raises
    ------
    ValueError
        If `geotransform` or `bbox_ogr` are invalid, or if pixel
        dimensions derived from the geotransform are zero or near-zero.

    Examples
    --------
    >>> gt = [0.0, 1.0, 0.0, 10.0, 0.0, -1.0] # 1m pixels, origin (0, 10)
    >>> sub_bbox = [2.0, 4.0, 4.0, 8.0] # Sub region
    >>> result = _get_sub_geotransform(gt, sub_bbox)
    >>> result["Transform"]
    [2.0, 1.0, 0.0, 8.0, 0.0, -1.0]
    >>> result["RasterXSize"]
    2
    >>> result["RasterYSize"]
    4
    """
    # Input validation
    if not _check_is_valid_geotransform(geotransform):
        raise ValueError(f"Invalid geotransform provided: {geotransform}")
    if not _check_is_valid_bbox(bbox_ogr):
        raise ValueError(f"Invalid OGR bounding box provided: {bbox_ogr}")

    # Extract values, converting to float
    sub_x_min, sub_x_max, sub_y_min, sub_y_max = map(float, bbox_ogr)
    pixel_width = float(geotransform[1])
    pixel_height = float(geotransform[5]) # Keep sign

    # Pixel dimensions already validated by _check_is_valid_geotransform

    # Calculate raster dimensions for the sub-region
    try:
        # Use np.rint for rounding before casting to int
        sub_raster_x_size = int(np.rint((sub_x_max - sub_x_min) / pixel_width))
        # Use absolute pixel height for size calculation
        sub_raster_y_size = int(np.rint((sub_y_max - sub_y_min) / abs(pixel_height)))
    except (OverflowError, ValueError, ZeroDivisionError) as e:
        raise ValueError(f"Error calculating sub-raster dimensions: {str(e)}") from e

    # Create the new geotransform for the sub-region
    # Origin is the top-left corner (sub_x_min, sub_y_max)
    # Pixel width/height and skew remain the same as the original
    sub_transform = [
        sub_x_min,
        pixel_width,
        float(geotransform[2]), # row_skew
        sub_y_max,
        float(geotransform[4]), # column_skew
        pixel_height # Keep original pixel height (sign matters)
    ]

    return {
        "Transform": sub_transform,
        "RasterXSize": abs(sub_raster_x_size), # Ensure positive size
        "RasterYSize": abs(sub_raster_y_size), # Ensure positive size
    }
