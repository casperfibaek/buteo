"""### Bounding box operation functions. ###

Functions to perform operations on bounding boxes.

There are two different formats for bounding boxes used by GDAL:</br>
OGR:  `[x_min, x_max, y_min, y_max]`</br>
WARP: `[x_min, y_min, x_max, y_max]`</br>

_If nothing else is stated, the OGR format is used._
"""

# Standard library
from typing import List, Union, Dict

# External
import numpy as np

# Internal
from buteo.utils.bbox_validation import (_check_is_valid_bbox, _check_is_valid_geotransform,
                                         _check_bboxes_intersect)
from buteo.utils import utils_base


def _get_pixel_offsets(
    geotransform: List[Union[int, float]],
    bbox_ogr: List[Union[int, float]],
) -> tuple[int, int, int, int]:
    """Get the pixels offsets for a bbox and a geotransform.

    Parameters
    ----------
    geotransform : List[Union[int, float]]
        A GDAL formatted geotransform.
        `[x_min, pixel_width, row_skew, y_max, column_skew, pixel_height]`
        pixel_height is negative for north up.
    bbox_ogr : List[Union[int, float]]
        An OGR formatted bbox.
        `[x_min, x_max, y_min, y_max]`

    Returns
    -------
    Tuple[int, int, int, int]
        A tuple of pixel offsets. `(x_start, y_start, x_size, y_size)`
        Where:
            x_start: Starting pixel in x direction
            y_start: Starting pixel in y direction
            x_size: Number of pixels in x direction
            y_size: Number of pixels in y direction

    Raises
    ------
    AssertionError
        If geotransform or bbox_ogr are invalid.
    ValueError
        If pixel width or height is zero.

    Examples
    --------
    >>> _get_pixel_offsets([0, 1, 0, 10, 0, -1], [2, 4, 4, 8])
    (2, 2, 2, 4)
    >>> _get_pixel_offsets([0, 0, 0, 0, 0, 0], [0, 1, 0, 1])
    Raises ValueError
    """
    # Input validation
    if geotransform is None or bbox_ogr is None:
        raise ValueError("geotransform and bbox_ogr cannot be None")

    if not _check_is_valid_geotransform(geotransform):
        raise ValueError(f"geotransform must be a valid GDAL geotransform. Received: {geotransform}")

    if not _check_is_valid_bbox(bbox_ogr):
        raise ValueError(f"bbox_ogr must be a valid OGR formatted bbox. Received: {bbox_ogr}")

    # Unpack values
    x_min, x_max, y_min, y_max = bbox_ogr
    origin_x = float(geotransform[0])
    origin_y = float(geotransform[3])
    pixel_width = float(geotransform[1])
    pixel_height = float(geotransform[5])

    # Check for zero pixel sizes
    if abs(pixel_width) < 1e-10 or abs(pixel_height) < 1e-10:
        raise ValueError("Pixel width and height cannot be zero or near-zero")

    # Calculate pixel offsets
    try:
        x_start = int(np.rint((x_min - origin_x) / pixel_width))
        y_start = int(np.rint((y_max - origin_y) / pixel_height))
        x_size = int(np.rint((x_max - x_min) / pixel_width))
        y_size = int(np.rint((y_min - y_max) / pixel_height))
    except (OverflowError, FloatingPointError) as e:
        raise ValueError(f"Error calculating pixel offsets: {str(e)}") from e

    # Ensure positive sizes
    x_size = abs(x_size)
    y_size = abs(y_size)

    return (x_start, y_start, x_size, y_size)


def _get_bbox_from_geotransform(
    geotransform: List[Union[int, float]],
    raster_x_size: int,
    raster_y_size: int,
) -> List[Union[int, float]]:
    """Get an OGR bounding box from a geotransform and raster sizes.

    Parameters
    ----------
    geotransform : List[Union[int, float]]
        A GDAL formatted geotransform.
        `[x_min, pixel_width, row_skew, y_max, column_skew, pixel_height]`
    raster_x_size : int
        The number of pixels in the x direction.
    raster_y_size : int
        The number of pixels in the y direction.

    Returns
    -------
    List[float]
        An OGR formatted bbox. `[x_min, x_max, y_min, y_max]`

    Raises
    ------
    ValueError
        If inputs are None or invalid
    AssertionError
        If geotransform is invalid
    TypeError
        If raster sizes are not integers

    Examples
    --------
    >>> _get_bbox_from_geotransform([0, 1, 0, 10, 0, -1], 5, 5)
    [0.0, 5.0, 5.0, 10.0]
    >>> _get_bbox_from_geotransform([0, 1, 0, 10, 0, -1], 0, 0)
    [0.0, 0.0, 10.0, 10.0]
    """
    # Input validation
    if geotransform is None:
        raise ValueError("geotransform cannot be None")

    if not isinstance(raster_x_size, int) or not isinstance(raster_y_size, int):
        raise TypeError("raster_x_size and raster_y_size must be integers")

    if raster_x_size < 0 or raster_y_size < 0:
        raise ValueError("raster sizes cannot be negative")

    # Verify geotransform validity
    if not _check_is_valid_geotransform(geotransform):
        raise ValueError(f"geotransform must be a valid GDAL geotransform. Received: {geotransform}")

    # Convert all values to float for consistent return type
    x_min = float(geotransform[0])
    pixel_width = float(geotransform[1])
    y_max = float(geotransform[3])
    pixel_height = float(geotransform[5])

    # Calculate extent
    x_max = x_min + (raster_x_size * pixel_width)
    y_min = y_max + (raster_y_size * pixel_height)

    # Ensure correct ordering when pixel width/height are negative
    if x_max < x_min:
        x_min, x_max = x_max, x_min
    if y_max < y_min:
        y_min, y_max = y_max, y_min

    return [x_min, x_max, y_min, y_max]


def _get_intersection_bboxes(
    bbox1_ogr: List[Union[int, float]],
    bbox2_ogr: List[Union[int, float]],
) -> List[Union[int, float]]:
    """Get the intersection of two OGR formatted bboxes.

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
    List[Union[int, float]]
        An OGR formatted bbox of the intersection.
        All values are returned as floats for consistency.

    Raises
    ------
    TypeError
        If either bbox is None or not a list/tuple
    ValueError
        If either bbox is not a valid OGR bbox format
        If the bboxes do not intersect
        If any coordinate values are NaN

    Examples
    --------
    >>> _get_intersection_bboxes([0, 2, 0, 2], [1, 3, 1, 3])
    [1.0, 2.0, 1.0, 2.0]
    >>> _get_intersection_bboxes([0, 1, 0, 1], [2, 3, 2, 3])
    Raises ValueError: Bounding boxes do not intersect
    >>> _get_intersection_bboxes(None, [0, 1, 0, 1])
    Raises TypeError
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

    # Check for intersection
    if not _check_bboxes_intersect(bbox1_ogr, bbox2_ogr):
        raise ValueError("Bounding boxes do not intersect")

    try:
        # Convert all values to float for consistent return type
        bbox1_x_min, bbox1_x_max, bbox1_y_min, bbox1_y_max = map(float, bbox1_ogr)
        bbox2_x_min, bbox2_x_max, bbox2_y_min, bbox2_y_max = map(float, bbox2_ogr)

        # Calculate intersection
        x_min = max(bbox1_x_min, bbox2_x_min)
        x_max = min(bbox1_x_max, bbox2_x_max)
        y_min = max(bbox1_y_min, bbox2_y_min)
        y_max = min(bbox1_y_max, bbox2_y_max)

        # Check for NaN values in result
        if any(np.isnan([x_min, x_max, y_min, y_max])):
            raise ValueError("Intersection resulted in NaN values")

        return [x_min, x_max, y_min, y_max]

    except (TypeError, ValueError) as e:
        raise ValueError(f"Error computing intersection: {str(e)}") from e


def _get_union_bboxes(
    bbox1_ogr: List[Union[int, float]],
    bbox2_ogr: List[Union[int, float]]
) -> List[Union[int, float]]:
    """Get the union of two OGR formatted bboxes.

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
    List[Union[int, float]]
        An OGR formatted bbox of the union.
        All values are returned as floats for consistency.

    Raises
    ------
    TypeError
        If either bbox is None or not a list/tuple
    ValueError
        If either bbox is not a valid OGR bbox format
        If any coordinate values are NaN

    Examples
    --------
    >>> _get_union_bboxes([0, 1, 0, 1], [1, 2, 1, 2])
    [0.0, 2.0, 0.0, 2.0]
    >>> _get_union_bboxes([0, 1, 0, 1], None)
    Raises TypeError
    >>> _get_union_bboxes([0, 1, None, 1], [0, 1, 0, 1])
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
        # Convert all values to float for consistent return type
        bbox1_x_min, bbox1_x_max, bbox1_y_min, bbox1_y_max = map(float, bbox1_ogr)
        bbox2_x_min, bbox2_x_max, bbox2_y_min, bbox2_y_max = map(float, bbox2_ogr)

        # Handle infinite values consistently
        if np.isinf(bbox1_x_min) or np.isinf(bbox2_x_min):
            x_min = -np.inf
        else:
            x_min = min(bbox1_x_min, bbox2_x_min)

        if np.isinf(bbox1_x_max) or np.isinf(bbox2_x_max):
            x_max = np.inf
        else:
            x_max = max(bbox1_x_max, bbox2_x_max)

        if np.isinf(bbox1_y_min) or np.isinf(bbox2_y_min):
            y_min = -np.inf
        else:
            y_min = min(bbox1_y_min, bbox2_y_min)

        if np.isinf(bbox1_y_max) or np.isinf(bbox2_y_max):
            y_max = np.inf
        else:
            y_max = max(bbox1_y_max, bbox2_y_max)

        # Check for NaN values in result
        if any(np.isnan([x_min, x_max, y_min, y_max])):
            raise ValueError("Union resulted in NaN values")

        return [x_min, x_max, y_min, y_max]

    except (TypeError, ValueError) as e:
        raise ValueError(f"Error computing union: {str(e)}") from e


def _get_aligned_bbox_to_pixel_size(
    bbox1_ogr: List[Union[int, float]],
    bbox2_ogr: List[Union[int, float]],
    pixel_width: Union[int, float],
    pixel_height: Union[int, float],
) -> List[Union[int, float]]:
    """Aligns two OGR formatted bboxes to a pixel size. Output is an augmented version
    of bbox2 aligned to bbox1's grid.

    Parameters
    ----------
    bbox1_ogr : List[Union[int, float]]
        An OGR formatted bbox to align to.
        `[x_min, x_max, y_min, y_max]`
    bbox2_ogr : List[Union[int, float]]
        An OGR formatted bbox to be aligned.
        `[x_min, x_max, y_min, y_max]`
    pixel_width : Union[int, float]
        The width of the pixel. Must be positive.
    pixel_height : Union[int, float]
        The height of the pixel. Can be negative for north-up orientation.

    Returns
    -------
    List[Union[int, float]]
        An OGR formatted bbox of the alignment.
        `[x_min, x_max, y_min, y_max]`
        All values are returned as float for consistency.

    Raises
    ------
    TypeError
        If inputs are None or of invalid type
    ValueError
        If bboxes are invalid or pixel dimensions are zero
        If pixel_width is negative

    Examples
    --------
    >>> _get_aligned_bbox_to_pixel_size([0, 4, 0, 4], [1, 3, 1, 3], 1.0, -1.0)
    [1.0, 3.0, 1.0, 3.0]
    >>> _get_aligned_bbox_to_pixel_size([0, 4, 0, 4], [1.2, 3.7, 1.2, 3.7], 1.0, -1.0)
    [1.0, 4.0, 1.0, 4.0]
    """
    # Input validation
    if any(x is None for x in [bbox1_ogr, bbox2_ogr, pixel_width, pixel_height]):
        raise TypeError("None values are not allowed in input parameters")

    if not isinstance(bbox1_ogr, (list, tuple)) or not isinstance(bbox2_ogr, (list, tuple)):
        raise TypeError("Bounding boxes must be lists or tuples")

    if not isinstance(pixel_width, (int, float)) or not isinstance(pixel_height, (int, float)):
        raise TypeError("Pixel dimensions must be numeric")

    # Validate bboxes
    if not _check_is_valid_bbox(bbox1_ogr):
        raise ValueError(f"Invalid bbox1_ogr format: {bbox1_ogr}")

    if not _check_is_valid_bbox(bbox2_ogr):
        raise ValueError(f"Invalid bbox2_ogr format: {bbox2_ogr}")

    # Validate pixel dimensions
    if pixel_width <= 0:
        raise ValueError(f"pixel_width must be positive, got: {pixel_width}")

    if pixel_height == 0:
        raise ValueError("pixel_height cannot be zero")

    try:
        # Convert all values to float for consistent calculations
        bbox1_x_min, bbox1_x_max, bbox1_y_min, bbox1_y_max = map(float, bbox1_ogr)
        bbox2_x_min, bbox2_x_max, bbox2_y_min, bbox2_y_max = map(float, bbox2_ogr)
        pixel_width = float(pixel_width)
        pixel_height = float(pixel_height)

        # Calculate alignments with pixel grid
        x_min = bbox2_x_min - ((bbox2_x_min - bbox1_x_min) % pixel_width)
        x_max = bbox2_x_max + ((bbox1_x_max - bbox2_x_max) % pixel_width)

        abs_pixel_height = abs(pixel_height)
        y_min = bbox2_y_min - ((bbox2_y_min - bbox1_y_min) % abs_pixel_height)
        y_max = bbox2_y_max + ((bbox1_y_max - bbox2_y_max) % abs_pixel_height)

        # Check for NaN or infinite values in result
        result = [x_min, x_max, y_min, y_max]
        if any(np.isnan(val) for val in result):
            raise ValueError("Alignment resulted in NaN values")

        if any(np.isinf(val) for val in result):
            raise ValueError("Alignment resulted in infinite values")

        return result

    except (TypeError, ValueError) as e:
        raise ValueError(f"Error during bbox alignment: {str(e)}") from e


def _get_gdal_bbox_from_ogr_bbox(
    bbox_ogr: List[Union[int, float]]
) -> List[float]:
    """Converts an OGR formatted bbox to a GDAL formatted one.
    `[x_min, x_max, y_min, y_max] -> [x_min, y_min, x_max, y_max]`

    Parameters
    ----------
    bbox_ogr : List[Union[int, float]]
        An OGR formatted bbox.
        `[x_min, x_max, y_min, y_max]`

    Returns
    -------
    List[float]
        A GDAL formatted bbox. `[x_min, y_min, x_max, y_max]`

    Raises
    ------
    TypeError
        If bbox_ogr is None or not a list/tuple
    ValueError
        If bbox_ogr is not a valid bbox format

    Examples
    --------
    >>> _get_gdal_bbox_from_ogr_bbox([0, 1, 0, 1])
    [0.0, 0.0, 1.0, 1.0]
    >>> _get_gdal_bbox_from_ogr_bbox(None)
    Raises TypeError
    """
    # Input validation
    if bbox_ogr is None:
        raise TypeError("bbox_ogr cannot be None")

    if not isinstance(bbox_ogr, (list, tuple)):
        raise TypeError(f"bbox_ogr must be a list or tuple, got {type(bbox_ogr)}")

    # Convert tuple to list if necessary
    if isinstance(bbox_ogr, tuple):
        bbox_ogr = list(bbox_ogr)

    # Validate bbox format
    if not _check_is_valid_bbox(bbox_ogr):
        raise ValueError(f"Invalid bbox format: {bbox_ogr}")

    try:
        # Convert all values to float for consistent return type
        x_min = float(bbox_ogr[0])
        x_max = float(bbox_ogr[1])
        y_min = float(bbox_ogr[2])
        y_max = float(bbox_ogr[3])
    except (TypeError, ValueError) as e:
        raise ValueError(f"Could not convert bbox values to float: {str(e)}") from e

    return [x_min, y_min, x_max, y_max]


def _get_ogr_bbox_from_gdal_bbox(
    bbox_gdal: List[Union[int, float]],
) -> List[float]:
    """Converts a GDAL formatted bbox to an OGR formatted one.
    `[x_min, y_min, x_max, y_max] -> [x_min, x_max, y_min, y_max]`

    Parameters
    ----------
    bbox_gdal : List[Union[int, float]]
        A GDAL formatted bbox.
        `[x_min, y_min, x_max, y_max]`

    Returns
    -------
    List[float]
        An OGR formatted bbox. `[x_min, x_max, y_min, y_max]`

    Raises
    ------
    TypeError
        If bbox_gdal is None or not a list/tuple
    ValueError
        If bbox_gdal doesn't contain exactly 4 numeric values
        If bbox_gdal contains None values

    Examples
    --------
    >>> _get_ogr_bbox_from_gdal_bbox([0, 0, 1, 1])
    [0.0, 1.0, 0.0, 1.0]
    >>> _get_ogr_bbox_from_gdal_bbox(None)
    Raises TypeError
    >>> _get_ogr_bbox_from_gdal_bbox([0, None, 1, 1])
    Raises ValueError
    """
    # Input validation
    if bbox_gdal is None:
        raise TypeError("bbox_gdal cannot be None")

    if not isinstance(bbox_gdal, (list, tuple)):
        raise TypeError(f"bbox_gdal must be a list or tuple, got {type(bbox_gdal)}")

    # Convert tuple to list if necessary
    if isinstance(bbox_gdal, tuple):
        bbox_gdal = list(bbox_gdal)

    # Check length
    if len(bbox_gdal) != 4:
        raise ValueError(f"bbox_gdal must have exactly 4 values, got {len(bbox_gdal)}")

    # Check for None values and convert to float
    try:
        x_min, y_min, x_max, y_max = [float(val) if val is not None else None for val in bbox_gdal]
    except (TypeError, ValueError) as e:
        raise ValueError(f"All bbox values must be numeric: {str(e)}") from e

    if any(val is None for val in [x_min, y_min, x_max, y_max]):
        raise ValueError("bbox_gdal cannot contain None values")

    return [float(x_min), float(x_max), float(y_min), float(y_max)] # type: ignore


def _get_geotransform_from_bbox(
    bbox_ogr: List[Union[int, float]],
    raster_x_size: int,
    raster_y_size: int,
) -> List[float]:
    """Convert an OGR formatted bounding box to a GDAL GeoTransform.
    `[x_min, x_max, y_min, y_max] -> [x_min, pixel_width, x_skew, y_max, y_skew, pixel_height]`

    Parameters
    ----------
    bbox_ogr : List[Union[int, float]]
        An OGR formatted bbox.
        `[x_min, x_max, y_min, y_max]`
    raster_x_size : int
        The number of pixels in the x direction.
    raster_y_size : int
        The number of pixels in the y direction.

    Returns
    -------
    List[float]
        A GDAL GeoTransform. `[x_min, pixel_width, x_skew, y_max, y_skew, pixel_height]`

    Raises
    ------
    TypeError
        If inputs are None or of invalid type
    ValueError
        If bbox is invalid or raster dimensions are invalid
        If division by zero would occur
    """
    # Input validation
    if bbox_ogr is None:
        raise TypeError("bbox_ogr cannot be None")

    if not isinstance(raster_x_size, int) or not isinstance(raster_y_size, int):
        raise TypeError("raster dimensions must be integers")

    if raster_x_size <= 0 or raster_y_size <= 0:
        raise ValueError("raster dimensions must be positive")

    if not _check_is_valid_bbox(bbox_ogr):
        raise ValueError(f"Invalid bbox format: {bbox_ogr}")

    try:
        x_min, x_max, y_min, y_max = map(float, bbox_ogr)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Could not convert bbox values to float: {str(e)}") from e

    # Calculate pixel dimensions
    try:
        pixel_width = (x_max - x_min) / float(raster_x_size)
        pixel_height = (y_max - y_min) / float(raster_y_size)
    except ZeroDivisionError as e:
        raise ValueError("Cannot calculate pixel size with zero raster dimensions") from e

    # Check for very small pixel dimensions that might cause precision issues
    if abs(pixel_width) < 1e-10 or abs(pixel_height) < 1e-10:
        raise ValueError("Computed pixel dimensions are too small")

    # Return consistent float values with no skew
    return [
        float(x_min),       # origin x
        float(pixel_width), # pixel width
        0.0,                # x skew
        float(y_max),       # origin y
        0.0,                # y skew
        utils_base._ensure_negative(float(pixel_height)) # pixel height (negative for north-up)
    ]


def _get_sub_geotransform(
    geotransform: List[Union[int, float]],
    bbox_ogr: List[Union[int, float]],
) -> Dict[str, Union[List[Union[int, float]], Union[int, float]]]:
    """Create a GeoTransform and the raster sizes for an OGR formatted bbox.

    Parameters
    ----------
    geotransform : List[Union[int, float]]
        A GDAL geotransform.
        `[top_left_x, pixel_width, rotation_x, top_left_y, rotation_y, pixel_height]`
    bbox_ogr : List[Union[int, float]]
        An OGR formatted bbox.
        `[x_min, x_max, y_min, y_max]`

    Returns
    -------
    Dict[str, Union[List[Union[int, float]], Union[int, float]]]
        Dictionary containing:
        - "Transform": List[Union[int, float]] - The geotransform parameters
        - "RasterXSize": Union[int, float] - Number of pixels in X direction
        - "RasterYSize": Union[int, float] - Number of pixels in Y direction

    Raises
    ------
    TypeError
        If inputs are None or of invalid type
    ValueError
        If geotransform or bbox are invalid
        If pixel dimensions are zero or near-zero

    Examples
    --------
    >>> geotransform = [0, 1, 0, 10, 0, -1]
    >>> bbox = [2, 4, 4, 8]
    >>> _get_sub_geotransform(geotransform, bbox)
    {'Transform': [2.0, 1.0, 0.0, 8.0, 0.0, -1.0], 'RasterXSize': 2, 'RasterYSize': 4}
    """
    # Input validation
    if geotransform is None or bbox_ogr is None:
        raise TypeError("geotransform and bbox_ogr cannot be None")

    if not _check_is_valid_geotransform(geotransform):
        raise ValueError(f"Invalid geotransform format: {geotransform}")

    if not _check_is_valid_bbox(bbox_ogr):
        raise ValueError(f"Invalid bbox format: {bbox_ogr}")

    # Extract values
    x_min, x_max, y_min, y_max = map(float, bbox_ogr)
    pixel_width = float(geotransform[1])
    pixel_height = float(geotransform[5])

    # Check for zero or near-zero pixel dimensions
    epsilon = 1e-10
    if abs(pixel_width) < epsilon or abs(pixel_height) < epsilon:
        raise ValueError("Pixel width and height cannot be zero or near-zero")

    try:
        # Calculate raster dimensions
        raster_x_size = int(round((x_max - x_min) / pixel_width))
        raster_y_size = int(round((y_max - y_min) / pixel_height))

        # Create new geotransform
        new_transform = [
            x_min,
            pixel_width,
            0.0,
            y_max,
            0.0,
            utils_base._ensure_negative(pixel_height)
        ]

        # Ensure all transform values are float
        new_transform = [float(val) for val in new_transform]

    except (OverflowError, ValueError) as e:
        raise ValueError(f"Error calculating raster dimensions: {str(e)}") from e

    return {
        "Transform": new_transform,
        "RasterXSize": abs(raster_x_size),
        "RasterYSize": abs(raster_y_size),
    }
