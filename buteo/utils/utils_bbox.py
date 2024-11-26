"""### Bounding box utility functions. ###

Various utility functions to work with bounding boxes and gdal.

There are two different formats for bounding boxes used by GDAL:</br>
OGR:  `[x_min, x_max, y_min, y_max]`</br>
WARP: `[x_min, y_min, x_max, y_max]`</br>

_If nothing else is stated, the OGR format is used._

The GDAL geotransform is a list of six parameters:</br>
`x_min, pixel_width, row_skew, y_max, column_skew, pixel_height (negative for north-up)`
"""

# Standard library
from typing import List, Union, Dict, Any, Optional
from uuid import uuid4

# External
import numpy as np
from osgeo import ogr, osr, gdal

# Internal
from buteo.utils import utils_base, utils_projection, utils_path, utils_gdal



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


def _get_bbox_from_raster(
    raster_dataframe: gdal.Dataset,
) -> List[Union[int, float]]:
    """Gets an OGR bounding box from a GDAL raster dataframe.
    The bounding box is in the same projection as the raster.

    Parameters
    ----------
    raster_dataframe : gdal.Dataset
        A GDAL raster dataframe.

    Returns
    -------
    List[Union[int, float]]
        An OGR formatted bbox. `[x_min, x_max, y_min, y_max]`

    Raises
    ------
    TypeError
        If raster_dataframe is not a gdal.Dataset
    ValueError
        If raster_dataframe has invalid dimensions or geotransform

    Examples
    --------
    >>> ds = gdal.Open('example.tif')
    >>> _get_bbox_from_raster(ds)
    [0.0, 100.0, 0.0, 100.0]
    >>> _get_bbox_from_raster(None)
    Raises TypeError
    """
    # Input validation
    if raster_dataframe is None:
        raise TypeError("raster_dataframe cannot be None")

    if not isinstance(raster_dataframe, gdal.Dataset):
        raise TypeError(f"raster_dataframe must be gdal.Dataset, got {type(raster_dataframe)}")

    # Get dimensions
    x_size = raster_dataframe.RasterXSize
    y_size = raster_dataframe.RasterYSize

    if x_size <= 0 or y_size <= 0:
        raise ValueError(f"Invalid raster dimensions: {x_size}x{y_size}")

    # Get geotransform
    geotransform = raster_dataframe.GetGeoTransform()
    if geotransform is None:
        raise ValueError("Could not get geotransform from raster")

    try:
        bbox = _get_bbox_from_geotransform(geotransform, x_size, y_size)
    except (ValueError, AssertionError) as e:
        raise ValueError(f"Failed to compute bbox from geotransform: {str(e)}") from e

    # Ensure return type consistency
    return [float(val) for val in bbox]


def _get_bbox_from_vector(
    vector_dataframe: ogr.DataSource,
) -> List[Union[int, float]]:
    """Gets an OGR bounding box from an OGR dataframe.
    The bounding box is in the same projection as the vector.

    Parameters
    ----------
    vector_dataframe : ogr.DataSource
        An OGR vector dataframe.

    Returns
    -------
    List[Union[int, float]]
        An OGR formatted bbox. `[x_min, x_max, y_min, y_max]`

    Raises
    ------
    TypeError
        If vector_dataframe is not a valid ogr.DataSource
    ValueError
        If vector_dataframe is None or contains no layers
        If any layer extent cannot be computed

    Examples
    --------
    >>> ds = ogr.Open("vector.shp")
    >>> _get_bbox_from_vector(ds)
    [0.0, 100.0, 0.0, 100.0]
    >>> _get_bbox_from_vector(None)
    Raises TypeError
    """
    # Input validation
    if vector_dataframe is None:
        raise TypeError("vector_dataframe cannot be None")

    if not isinstance(vector_dataframe, ogr.DataSource):
        raise TypeError(f"vector_dataframe must be ogr.DataSource, got {type(vector_dataframe)}")

    layer_count = vector_dataframe.GetLayerCount()
    if layer_count == 0:
        raise ValueError("vector_dataframe contains no layers")

    x_min = float('inf')
    x_max = float('-inf')
    y_min = float('inf')
    y_max = float('-inf')

    # Process each layer
    for layer_index in range(layer_count):
        layer = vector_dataframe.GetLayerByIndex(layer_index)
        if layer is None:
            raise ValueError(f"Could not access layer at index {layer_index}")

        try:
            layer_x_min, layer_x_max, layer_y_min, layer_y_max = layer.GetExtent()
        except Exception as e:
            raise ValueError(f"Could not compute extent for layer {layer_index}: {str(e)}") from e

        # Update bounds
        x_min = min(x_min, float(layer_x_min))
        x_max = max(x_max, float(layer_x_max))
        y_min = min(y_min, float(layer_y_min))
        y_max = max(y_max, float(layer_y_max))

    # Check if we found valid bounds
    if any(np.isinf([x_min, x_max, y_min, y_max])):
        raise ValueError("Could not compute valid bounds from vector layers")

    # Ensure consistent return type (float)
    return [float(x_min), float(x_max), float(y_min), float(y_max)]


def _get_bbox_from_vector_layer(
    vector_layer: ogr.Layer,
) -> List[Union[int, float]]:
    """Gets an OGR bounding box from an OGR layer.
    The bounding box is in the same projection as the layer.

    Parameters
    ----------
    vector_layer : ogr.Layer
        An OGR vector layer.

    Returns
    -------
    List[Union[int, float]]
        An OGR formatted bbox. `[x_min, x_max, y_min, y_max]`

    Raises
    ------
    TypeError
        If vector_layer is not an ogr.Layer
    ValueError
        If vector_layer is None or empty
        If layer extent cannot be computed

    Examples
    --------
    >>> layer = ogr.GetLayer()
    >>> _get_bbox_from_vector_layer(layer)
    [0.0, 100.0, 0.0, 100.0]
    >>> _get_bbox_from_vector_layer(None)
    Raises TypeError
    """
    # Input validation
    if vector_layer is None:
        raise TypeError("vector_layer cannot be None")

    if not isinstance(vector_layer, ogr.Layer):
        raise TypeError(f"vector_layer must be ogr.Layer, got {type(vector_layer)}")

    if vector_layer.GetFeatureCount() == 0:
        raise ValueError("vector_layer contains no features")

    # Get extent with error handling
    try:
        x_min, x_max, y_min, y_max = vector_layer.GetExtent()
    except Exception as e:
        raise ValueError(f"Could not compute extent for layer: {str(e)}") from e

    # Check for invalid values
    if any(np.isnan([x_min, x_max, y_min, y_max])):
        raise ValueError("Layer extent contains NaN values")

    if any(np.isinf([x_min, x_max, y_min, y_max])):
        raise ValueError("Layer extent contains infinite values")

    # Ensure consistent float return type
    return [float(x_min), float(x_max), float(y_min), float(y_max)]


def get_bbox_from_dataset(
    dataset: Union[str, gdal.Dataset, ogr.DataSource],
) -> List[Union[int, float]]:
    """Get the bbox from a dataset.
    The bounding box is in the same projection as the dataset.
    If you want the Bounding Box in WGS84, use `get_bbox_from_dataset_wgs84`.

    Parameters
    ----------
    dataset : str or gdal.Dataset or ogr.DataSource
        A dataset or dataset path.

    Returns
    -------
    List[float]
        The bounding box in ogr format: `[x_min, x_max, y_min, y_max]`.

    Raises
    ------
    TypeError
        If dataset is not a string, gdal.Dataset, or ogr.DataSource.
    ValueError
        If dataset is None or empty string.
    RuntimeError
        If dataset cannot be opened or bbox cannot be computed.

    Examples
    --------
    >>> ds = gdal.Open('example.tif')
    >>> get_bbox_from_dataset(ds)
    [0.0, 100.0, 0.0, 100.0]
    >>> get_bbox_from_dataset('example.shp')
    [0.0, 100.0, 0.0, 100.0]
    """
    # Input validation
    if dataset is None:
        raise ValueError("Dataset cannot be None")

    if not isinstance(dataset, (str, gdal.Dataset, ogr.DataSource)):
        raise TypeError(f"Dataset must be a string, gdal.Dataset, or ogr.DataSource. Got: {type(dataset)}")

    if isinstance(dataset, str) and not dataset.strip():
        raise ValueError("Dataset path cannot be empty")

    # Handle already opened datasets
    if isinstance(dataset, (gdal.Dataset, ogr.DataSource)):
        try:
            return (_get_bbox_from_raster(dataset) if isinstance(dataset, gdal.Dataset)
                   else _get_bbox_from_vector(dataset))
        except Exception as e:
            raise RuntimeError(f"Failed to get bbox from opened dataset: {str(e)}") from e

    # Try to open as raster first, then vector
    gdal.PushErrorHandler("CPLQuietErrorHandler")
    try:
        raster_ds = gdal.Open(dataset, gdal.GA_ReadOnly)
        if raster_ds is not None:
            bbox = _get_bbox_from_raster(raster_ds)
            raster_ds = None  # Close dataset
            gdal.PopErrorHandler()
            return bbox

        vector_ds = ogr.Open(dataset, gdal.GA_ReadOnly)
        if vector_ds is not None:
            bbox = _get_bbox_from_vector(vector_ds)
            vector_ds = None  # Close dataset
            gdal.PopErrorHandler()
            return bbox

    except Exception as e:
        gdal.PopErrorHandler()
        raise RuntimeError(f"Error processing dataset {dataset}: {str(e)}") from e

    gdal.PopErrorHandler()
    raise RuntimeError(f"Could not open dataset as either raster or vector: {dataset}")


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


def _get_geom_from_bbox(
    bbox_ogr: List[Union[int, float]],
) -> ogr.Geometry:
    """Convert an OGR bounding box to ogr.Geometry.
    `[x_min, x_max, y_min, y_max] -> ogr.Geometry`

    Parameters
    ----------
    bbox_ogr : List[Union[int, float]]
        An OGR formatted bbox.
        `[x_min, x_max, y_min, y_max]`

    Returns
    -------
    ogr.Geometry
        An OGR geometry representing the bounding box as a polygon.

    Raises
    ------
    TypeError
        If bbox_ogr is None or not a list/tuple.
    ValueError
        If bbox_ogr is not a valid bounding box format.

    Examples
    --------
    >>> bbox = [0, 1, 0, 1]
    >>> geom = _get_geom_from_bbox(bbox)
    >>> isinstance(geom, ogr.Geometry)
    True
    >>> bbox = None
    >>> _get_geom_from_bbox(bbox)
    Raises TypeError
    """
    # Input validation
    if bbox_ogr is None:
        raise TypeError("bbox_ogr cannot be None")

    if not isinstance(bbox_ogr, (list, tuple)):
        raise TypeError(f"bbox_ogr must be a list or tuple, got {type(bbox_ogr)}")

    if not _check_is_valid_bbox(bbox_ogr):
        raise ValueError(f"Invalid bbox format: {bbox_ogr}")

    try:
        x_min, x_max, y_min, y_max = map(float, bbox_ogr)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Could not convert bbox values to float: {str(e)}") from e

    # Create geometry
    try:
        ring = ogr.Geometry(ogr.wkbLinearRing)
        ring.AddPoint(x_min, y_min)
        ring.AddPoint(x_max, y_min)
        ring.AddPoint(x_max, y_max)
        ring.AddPoint(x_min, y_max)
        ring.AddPoint(x_min, y_min)  # Close the ring

        geom = ogr.Geometry(ogr.wkbPolygon)
        if geom.AddGeometry(ring) != 0:
            raise ValueError("Failed to add ring to polygon")

        if not geom.IsValid():
            raise ValueError("Created geometry is not valid")

        return geom

    except (RuntimeError, ValueError) as e:
        raise ValueError(f"Failed to create geometry: {str(e)}") from e


def _get_bbox_from_geom(geom: ogr.Geometry) -> List[float]:
    """Convert an ogr.Geometry to an OGR bounding box.
    `ogr.Geometry -> [x_min, x_max, y_min, y_max]`

    Parameters
    ----------
    geom : ogr.Geometry
        An OGR geometry.

    Returns
    -------
    List[float]
        An OGR formatted bbox. `[x_min, x_max, y_min, y_max]`

    Raises
    ------
    TypeError
        If geom is None or not an ogr.Geometry
    ValueError
        If geometry envelope cannot be computed
        If envelope contains invalid values
    """
    # Input validation
    if geom is None:
        raise TypeError("geom cannot be None")

    if not isinstance(geom, ogr.Geometry):
        raise TypeError(f"geom must be ogr.Geometry, got {type(geom)}")

    try:
        # GetEnvelope returns (minX, maxX, minY, maxY)
        envelope = geom.GetEnvelope()
    except Exception as e:
        raise ValueError(f"Failed to compute geometry envelope: {str(e)}") from e

    # Check for invalid values
    if any(np.isnan(val) for val in envelope):
        raise ValueError("Geometry envelope contains NaN values")

    # Convert to float and reorder to OGR format [x_min, x_max, y_min, y_max]
    try:
        bbox_ogr = [float(val) for val in envelope]
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid values in geometry envelope: {str(e)}") from e

    # Verify bbox is valid
    if not _check_is_valid_bbox(bbox_ogr):
        raise ValueError(f"Generated bbox is invalid: {bbox_ogr}")

    return bbox_ogr


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


def _get_wkt_from_bbox(
    bbox_ogr: List[Union[int, float]],
) -> str:
    """Converts an OGR formatted bbox to a WKT string.
    `[x_min, x_max, y_min, y_max] -> WKT`

    Parameters
    ----------
    bbox_ogr : List[Union[int, float]]
        An OGR formatted bbox.
        `[x_min, x_max, y_min, y_max]`

    Returns
    -------
    str
        A WKT string representing a polygon. `POLYGON ((...))`

    Raises
    ------
    TypeError
        If bbox_ogr is None or not a list/tuple
    ValueError
        If bbox_ogr is not a valid bbox format
        If bbox coordinates cannot be converted to float

    Examples
    --------
    >>> _get_wkt_from_bbox([0, 1, 0, 1])
    'POLYGON ((0.0 0.0, 1.0 0.0, 1.0 1.0, 0.0 1.0, 0.0 0.0))'
    >>> _get_wkt_from_bbox(None)
    Raises TypeError
    """
    # Input validation
    if bbox_ogr is None:
        raise TypeError("bbox_ogr cannot be None")

    if not isinstance(bbox_ogr, (list, tuple)):
        raise TypeError(f"bbox_ogr must be a list or tuple, got {type(bbox_ogr)}")

    if not _check_is_valid_bbox(bbox_ogr):
        raise ValueError(f"Invalid bbox format: {bbox_ogr}")

    try:
        # Convert all values to float for consistent representation
        x_min = float(bbox_ogr[0])
        x_max = float(bbox_ogr[1])
        y_min = float(bbox_ogr[2])
        y_max = float(bbox_ogr[3])
    except (TypeError, ValueError) as e:
        raise ValueError(f"Could not convert bbox coordinates to float: {str(e)}") from e

    # Format WKT string with consistent decimal places
    wkt = (
        f"POLYGON (("
        f"{x_min:.6f} {y_min:.6f}, "
        f"{x_max:.6f} {y_min:.6f}, "
        f"{x_max:.6f} {y_max:.6f}, "
        f"{x_min:.6f} {y_max:.6f}, "
        f"{x_min:.6f} {y_min:.6f}))"
    )

    return wkt


def _get_geojson_from_bbox(
    bbox_ogr: List[Union[int, float]],
) -> Dict[str, Union[str, List[List[List[Union[int, float]]]]]]:
    """Converts an OGR formatted bbox to a GeoJson dictionary.
    `[x_min, x_max, y_min, y_max] -> GeoJson`

    Parameters
    ----------
    bbox_ogr : List[Union[int, float]]
        An OGR formatted bbox.
        `[x_min, x_max, y_min, y_max]`

    Returns
    -------
    Dict[str, Union[str, List[List[List[Union[int, float]]]]]]
        A GeoJson dictionary with the following structure:
        `{ "type": "Polygon", "coordinates": [[[float, float], ...]] }`

    Raises
    ------
    TypeError
        If bbox_ogr is None or not a list/tuple
    ValueError
        If bbox_ogr is not a valid bbox format
        If bbox coordinates cannot be converted to float

    Examples
    --------
    >>> _get_geojson_from_bbox([0, 1, 0, 1])
    {'type': 'Polygon', 'coordinates': [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]]}
    >>> _get_geojson_from_bbox(None)
    Raises TypeError
    >>> _get_geojson_from_bbox([0, None, 0, 1])
    Raises ValueError
    """
    # Input validation
    if bbox_ogr is None:
        raise TypeError("bbox_ogr cannot be None")

    if not isinstance(bbox_ogr, (list, tuple)):
        raise TypeError(f"bbox_ogr must be a list or tuple, got {type(bbox_ogr)}")

    if not _check_is_valid_bbox(bbox_ogr):
        raise ValueError(f"Invalid bbox format: {bbox_ogr}")

    try:
        # Convert all values to float for consistent return type
        x_min = float(bbox_ogr[0])
        x_max = float(bbox_ogr[1])
        y_min = float(bbox_ogr[2])
        y_max = float(bbox_ogr[3])
    except (TypeError, ValueError) as e:
        raise ValueError(f"Could not convert bbox coordinates to float: {str(e)}") from e

    # Create GeoJSON structure with explicit float values
    geojson: Dict[str, Union[str, List[List[List[float]]]]] = {
        "type": "Polygon",
        "coordinates": [
            [
                [x_min, y_min],
                [x_max, y_min],
                [x_max, y_max],
                [x_min, y_max],
                [x_min, y_min],
            ]
        ],
    }

    return geojson


def _get_vector_from_bbox(
    bbox_ogr: List[Union[int, float]],
    projection_osr: Optional[osr.SpatialReference] = None,
) -> str:
    """Converts an OGR formatted bbox to a vector file in /vsimem/.
    The vector is stored as a GeoPackage (.gpkg) file.

    Parameters
    ----------
    bbox_ogr : List[Union[int, float]]
        An OGR formatted bbox.
        `[x_min, x_max, y_min, y_max]`
    projection_osr : Optional[osr.SpatialReference], optional
        The projection of the vector. If None, uses WGS84.

    Returns
    -------
    str
        Path to created vector file in /vsimem/.

    Raises
    ------
    TypeError
        If bbox_ogr is None or not a list/tuple
        If projection_osr is not None and not an osr.SpatialReference
    ValueError
        If bbox_ogr is not a valid bbox format
        If vector creation fails

    Examples
    --------
    >>> proj = osr.SpatialReference()
    >>> proj.ImportFromEPSG(4326)
    >>> path = _get_vector_from_bbox([0, 1, 0, 1], proj)
    >>> isinstance(path, str)
    True
    >>> path.startswith('/vsimem/')
    True
    """
    # Input validation
    if bbox_ogr is None:
        raise TypeError("bbox_ogr cannot be None")

    if not isinstance(bbox_ogr, (list, tuple)):
        raise TypeError(f"bbox_ogr must be a list or tuple, got {type(bbox_ogr)}")

    if not _check_is_valid_bbox(bbox_ogr):
        raise ValueError(f"Invalid bbox format: {bbox_ogr}")

    if projection_osr is not None and not isinstance(projection_osr, osr.SpatialReference):
        raise TypeError(f"projection_osr must be osr.SpatialReference, got {type(projection_osr)}")

    # Use WGS84 as default projection if none provided
    if projection_osr is None:
        projection_osr = utils_projection._get_default_projection_osr()

    try:
        # Create geometry from bbox
        geom = _get_geom_from_bbox(bbox_ogr)
        if geom is None:
            raise ValueError("Failed to create geometry from bbox")

        # Create unique filename in /vsimem/
        extent_name = f"/vsimem/{uuid4().hex}_extent.gpkg"

        # Create vector dataset
        driver = ogr.GetDriverByName("GPKG")
        if driver is None:
            raise RuntimeError("GPKG driver not available")

        extent_ds = driver.CreateDataSource(extent_name)
        if extent_ds is None:
            raise RuntimeError(f"Failed to create vector dataset at {extent_name}")

        # Create layer
        layer = extent_ds.CreateLayer("extent_ogr", projection_osr, ogr.wkbPolygon)
        if layer is None:
            raise RuntimeError("Failed to create layer")

        # Create and set feature
        feature = ogr.Feature(layer.GetLayerDefn())
        if feature is None:
            raise RuntimeError("Failed to create feature")

        if feature.SetGeometry(geom) != 0:
            raise RuntimeError("Failed to set geometry")

        if layer.CreateFeature(feature) != 0:
            raise RuntimeError("Failed to create feature in layer")

        # Cleanup
        feature = None
        layer.SyncToDisk()
        extent_ds = None

        return extent_name

    except Exception as e:
        # Cleanup in case of error
        if 'extent_ds' in locals():
            extent_ds = None
        if 'extent_name' in locals():
            driver = ogr.GetDriverByName("GPKG")
            if driver:
                driver.DeleteDataSource(extent_name)
        raise RuntimeError(f"Failed to create vector from bbox: {str(e)}") from e


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


def _transform_point(point: List[float], transformer: osr.CoordinateTransformation) -> List[Union[int, float]]:
    """Transform a point using the provided transformer.

    Parameters
    ----------
    point : List[float]
        A point to transform [x, y].
    transformer : osr.CoordinateTransformation
        The coordinate transformer.

    Returns
    -------
    List[float]
        The transformed point [y, x].

    Raises
    ------
    ValueError
        If the point transformation fails.
    """
    try:
        transformed = transformer.TransformPoint(point[0], point[1])
        if transformed is None:
            raise ValueError("TransformPoint returned None")
        return [float(transformed[1]), float(transformed[0])]
    except (RuntimeError, ValueError) as e:
        raise ValueError(f"Failed to transform point {point}: {str(e)}") from e


def _transform_bbox_coordinates(
    bbox_ogr: List[Union[int, float]],
    transformer: osr.CoordinateTransformation,
) -> List[List[float]]:
    """Transform bbox corners using coordinate transformation.

    Parameters
    ----------
    bbox_ogr : List[Union[int, float]]
        An OGR formatted bbox. [x_min, x_max, y_min, y_max]
    transformer : osr.CoordinateTransformation
        The coordinate transformer to use.

    Returns
    -------
    List[List[float]]
        List of transformed coordinates [[x1, y1], [x2, y2], [x3, y3], [x4, y4]].

    Raises
    ------
    ValueError
        If coordinate transformation fails.
    """
    x_min, x_max, y_min, y_max = map(float, bbox_ogr)
    corners = [
        [x_min, y_min],  # lower left
        [x_max, y_min],  # lower right
        [x_max, y_max],  # upper right
        [x_min, y_max],  # upper left
    ]

    transformed_points = []
    gdal.PushErrorHandler("CPLQuietErrorHandler")
    try:
        transformed_points = [_transform_point(point, transformer) for point in corners]
        if len(transformed_points) != 4:
            raise ValueError("Failed to transform all corners")
        return transformed_points
    finally:
        gdal.PopErrorHandler()


def _create_polygon_from_points(points: List[List[float]]) -> ogr.Geometry:
    """Create a polygon geometry from a list of points.

    Parameters
    ----------
    points : List[List[float]]
        List of coordinates [[x1, y1], [x2, y2], ...].

    Returns
    -------
    ogr.Geometry
        A valid polygon geometry.

    Raises
    ------
    ValueError
        If geometry creation fails.
    """
    ring = ogr.Geometry(ogr.wkbLinearRing)
    for point in points:
        ring.AddPoint(point[0], point[1])
    ring.AddPoint(points[0][0], points[0][1])  # Close the ring

    polygon = ogr.Geometry(ogr.wkbPolygon)
    if polygon.AddGeometry(ring) != 0:
        raise ValueError("Failed to create polygon from points")

    if not polygon.IsValid():
        raise ValueError("Created geometry is invalid")

    return polygon


def _get_bounds_from_bbox_as_geom(
    bbox_ogr: List[Union[int, float]],
    projection_osr: osr.SpatialReference,
) -> ogr.Geometry:
    """Convert a bounding box from one projection to WGS84 and return as OGR Geometry.

    Parameters
    ----------
    bbox_ogr : List[Union[int, float]]
        An OGR formatted bbox. [x_min, x_max, y_min, y_max]
    projection_osr : osr.SpatialReference
        The projection of the input bbox.

    Returns
    -------
    ogr.Geometry
        OGR Geometry object of the transformed geometry.

    Raises
    ------
    TypeError
        If inputs are None or of invalid type.
    ValueError
        If bbox is invalid or transformation fails.
    """
    if bbox_ogr is None or projection_osr is None:
        raise TypeError("bbox_ogr and projection_osr cannot be None")

    if not isinstance(bbox_ogr, (list, tuple)):
        raise TypeError(f"bbox_ogr must be a list or tuple, got {type(bbox_ogr)}")

    if not isinstance(projection_osr, osr.SpatialReference):
        raise TypeError(f"projection_osr must be osr.SpatialReference, got {type(projection_osr)}")

    if not _check_is_valid_bbox(bbox_ogr):
        raise ValueError(f"Invalid bbox format: {bbox_ogr}")

    default_projection = utils_projection._get_default_projection_osr()

    # Return untransformed geometry if already in WGS84
    if utils_projection._check_projections_match(projection_osr, default_projection):
        return _get_geom_from_bbox(bbox_ogr)

    # Create transformer with area of interest
    try:
        options = osr.CoordinateTransformationOptions()
        options.SetAreaOfInterest(-180.0, -90.0, 180.0, 90.0)
        transformer = osr.CoordinateTransformation(projection_osr, default_projection, options)
    except Exception as e:
        raise ValueError(f"Failed to create coordinate transformation: {str(e)}") from e

    # Transform coordinates and create geometry
    try:
        transformed_points = _transform_bbox_coordinates(bbox_ogr, transformer)
        return _create_polygon_from_points(transformed_points)
    except Exception as e:
        raise ValueError(f"Failed to create transformed geometry: {str(e)}") from e


def _get_bounds_from_bbox_as_wkt(
    bbox_ogr: List[Union[int, float]],
    projection_osr: osr.SpatialReference,
) -> str:
    """Convert a bounding box from one projection to WGS84 and return as WKT string.

    Parameters
    ----------
    bbox_ogr : List[Union[int, float]]
        An OGR formatted bbox.
        `[x_min, x_max, y_min, y_max]`
    projection_osr : osr.SpatialReference
        The projection of the input bbox.

    Returns
    -------
    str
        WKT string representation of the transformed geometry.

    Raises
    ------
    TypeError
        If bbox_ogr is None or not a list/tuple.
        If projection_osr is None or not an osr.SpatialReference.
    ValueError
        If bbox_ogr is not a valid bbox.
        If coordinate transformation fails.
        If geometry creation fails.
    """
    geom = _get_bounds_from_bbox_as_geom(bbox_ogr, projection_osr)
    return geom.ExportToWkt()


def _get_utm_zone_from_bbox(
    bbox_ogr_latlng: List[Union[int, float]],
) -> str:
    """Get the UTM zone from a WGS84 (lat/long) OGR formatted bbox.

    Parameters
    ----------
    bbox_ogr_latlng : List[Union[int, float]]
        An OGR formatted bbox in WGS84 coordinates.
        `[x_min, x_max, y_min, y_max]`

    Returns
    -------
    str
        UTM zone identifier (e.g., "32N" or "32S")

    Raises
    ------
    TypeError
        If bbox_ogr_latlng is None or not a list/tuple
    ValueError
        If bbox_ogr_latlng is not a valid lat/long bbox
        If bbox_ogr_latlng contains invalid coordinates

    Examples
    --------
    >>> _get_utm_zone_from_bbox([8, 9, 50, 51])
    '32N'
    >>> _get_utm_zone_from_bbox([-180, 180, -90, 90])  # World bbox
    '30N'  # Returns central UTM zone
    >>> _get_utm_zone_from_bbox(None)
    Raises TypeError
    """
    # Input validation
    if bbox_ogr_latlng is None:
        raise TypeError("bbox_ogr_latlng cannot be None")

    if not isinstance(bbox_ogr_latlng, (list, tuple)):
        raise TypeError(f"bbox_ogr_latlng must be a list or tuple, got {type(bbox_ogr_latlng)}")

    # Validate bbox format
    if not _check_is_valid_bbox(bbox_ogr_latlng):
        raise ValueError(f"Invalid bbox format: {bbox_ogr_latlng}")

    if not _check_is_valid_bbox_latlng(bbox_ogr_latlng):
        raise ValueError(f"Bbox is not in valid lat/long format: {bbox_ogr_latlng}")

    try:
        # Convert to float and calculate midpoint
        lng_min, lng_max, lat_min, lat_max = map(float, bbox_ogr_latlng)

        # Handle coordinate wrapping for longitude
        if lng_min > lng_max:  # Crosses 180/-180 meridian
            lng_min, lng_max = lng_max, lng_min

        # Calculate midpoint
        mid_lng = ((lng_min + lng_max) / 2) % 360  # Normalize to 0-360
        if mid_lng > 180:
            mid_lng -= 360  # Convert back to -180 to 180 range

        mid_lat = (lat_min + lat_max) / 2

        # Check for valid coordinates
        if not (-180 <= mid_lng <= 180) or not -90 <= mid_lat <= 90:
            raise ValueError(f"Invalid midpoint coordinates: lat={mid_lat}, lng={mid_lng}")

        epsg = utils_projection._get_utm_epsg_from_latlng([mid_lat, mid_lng])
        hemisphere = "N" if mid_lat >= 0 else "S"
        zone = epsg[-2:]

        return f"{zone}{hemisphere}"

    except (TypeError, ValueError) as e:
        raise ValueError(f"Error calculating UTM zone: {str(e)}") from e


def _get_utm_zone_from_dataset(
    dataset: Union[str, gdal.Dataset, ogr.DataSource],
) -> str:
    """Get the UTM zone from a GDAL dataset.

    Parameters
    ----------
    dataset : str or gdal.Dataset or ogr.DataSource
        A dataset or path to a dataset to get the UTM zone from.
        The dataset must have valid spatial reference information.

    Returns
    -------
    str
        The UTM zone identifier (e.g., "32N" or "32S")

    Raises
    ------
    TypeError
        If dataset is None or not a string, gdal.Dataset, or ogr.DataSource
    ValueError
        If dataset cannot be opened or processed
        If dataset has invalid spatial reference information
        If UTM zone cannot be determined

    Examples
    --------
    >>> ds = gdal.Open('example.tif')
    >>> _get_utm_zone_from_dataset(ds)
    '32N'
    >>> _get_utm_zone_from_dataset('path/to/vector.shp')
    '33S'
    >>> _get_utm_zone_from_dataset(None)
    Raises TypeError
    """
    # Input validation
    if dataset is None:
        raise TypeError("dataset cannot be None")

    if not isinstance(dataset, (str, gdal.Dataset, ogr.DataSource)):
        raise TypeError(f"dataset must be a string, gdal.Dataset, or ogr.DataSource, got {type(dataset)}")

    try:
        # Get bbox and source projection
        bbox = get_bbox_from_dataset(dataset)
        source_projection = utils_projection._get_projection_from_dataset(dataset)

        if source_projection is None:
            raise ValueError("Could not determine dataset projection")

        # Get target projection (WGS84)
        target_projection = utils_projection._get_default_projection_osr()

        # Check if source and target projections are the same
        if not utils_projection._check_projections_match(source_projection, target_projection):
            # Reproject bbox to WGS84
            bbox = utils_projection.reproject_bbox(bbox, source_projection, target_projection)

        # Get UTM zone
        utm_zone = _get_utm_zone_from_bbox(bbox)

        if not isinstance(utm_zone, str):
            raise ValueError("Failed to get valid UTM zone string")

        return utm_zone

    except Exception as e:
        raise ValueError(f"Failed to determine UTM zone for dataset: {str(e)}") from e


def _get_utm_zone_from_dataset_list(
    datasets: List[Union[str, gdal.Dataset, ogr.DataSource]],
) -> str:
    """Get the UTM zone from a list of GDAL datasets.

    Parameters
    ----------
    datasets : List[Union[str, gdal.Dataset, ogr.DataSource]]
        A list of datasets or paths to datasets to determine the UTM zone from.
        The datasets must have valid spatial reference information.

    Returns
    -------
    str
        The UTM zone identifier (e.g., "32N" or "32S")

    Raises
    ------
    TypeError
        If datasets is None or not a list
        If any dataset in the list is not a string, gdal.Dataset, or ogr.DataSource
    ValueError
        If datasets is empty
        If any dataset cannot be opened or processed
        If datasets have incompatible projections
        If UTM zone cannot be determined

    Examples
    --------
    >>> ds1 = gdal.Open('example1.tif')
    >>> ds2 = gdal.Open('example2.tif')
    >>> _get_utm_zone_from_dataset_list([ds1, ds2])
    '32N'
    >>> _get_utm_zone_from_dataset_list(['path1.shp', 'path2.tif'])
    '33N'
    >>> _get_utm_zone_from_dataset_list([])
    Raises ValueError: Empty dataset list
    """
    # Input validation
    if datasets is None:
        raise TypeError("datasets cannot be None")

    if not isinstance(datasets, (list, tuple, np.ndarray)):
        raise TypeError(f"datasets must be a list, tuple, or numpy array, got {type(datasets)}")

    if len(datasets) == 0:
        raise ValueError("Dataset list cannot be empty")

    # Validate each dataset
    for dataset in datasets:
        if not isinstance(dataset, (str, gdal.Dataset, ogr.DataSource)):
            raise TypeError(
                f"Each dataset must be a string, gdal.Dataset, or ogr.DataSource, got {type(dataset)}"
            )

    try:
        # Get WGS84 projection once
        latlng_proj = utils_projection._get_default_projection_osr()
        latlng_bboxes = []

        # Process each dataset
        for dataset in datasets:
            try:
                # Get bbox and projection for current dataset
                bbox = get_bbox_from_dataset(dataset)
                projection = utils_projection._get_projection_from_dataset(dataset)

                if projection is None:
                    raise ValueError(f"Could not determine projection for dataset: {dataset}")

                # Reproject bbox to WGS84
                latlng_bbox = utils_projection.reproject_bbox(bbox, projection, latlng_proj)
                latlng_bboxes.append(latlng_bbox)

            except Exception as e:
                raise ValueError(f"Failed to process dataset: {str(e)}") from e

        # Calculate union of all bboxes
        try:
            union_bbox = latlng_bboxes[0]
            for bbox in latlng_bboxes[1:]:
                union_bbox = _get_union_bboxes(union_bbox, bbox)
        except Exception as e:
            raise ValueError(f"Failed to compute union of bounding boxes: {str(e)}") from e

        # Get UTM zone from union bbox
        utm_zone = _get_utm_zone_from_bbox(union_bbox)

        if not isinstance(utm_zone, str):
            raise ValueError("Failed to get valid UTM zone string")

        return utm_zone

    except Exception as e:
        raise ValueError(f"Failed to determine UTM zone for dataset list: {str(e)}") from e


def _additional_bboxes(
    bbox_ogr: List[Union[int, float]],
    projection_osr: osr.SpatialReference,
) -> Dict[str, Any]:
    """This is an internal utility function for metadata generation. It takes a standard
    OGR bounding box and returns a dictionary of variations of bounding boxes and related geometries.

    Parameters
    ----------
    bbox_ogr : List[Union[int, float]]
        An OGR formatted bbox.
        `[x_min, x_max, y_min, y_max]`
    projection_osr : osr.SpatialReference
        The projection of the input bbox.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing various representations of the bbox:
        - bbox_latlng: List[float] - The bbox in lat/long coordinates
        - bbox_wkt: str - The bbox in WKT format
        - bbox_wkt_latlng: str - The bbox in WKT format in lat/long coordinates
        - bbox_geom: ogr.Geometry - The bbox as OGR geometry
        - bbox_geom_latlng: ogr.Geometry - The bbox as OGR geometry in lat/long coordinates
        - bbox_gdal: List[float] - The bbox in GDAL format [minx, miny, maxx, maxy]
        - bbox_gdal_latlng: List[float] - The bbox in GDAL format in lat/long coordinates
        - bbox_dict: Dict[str, float] - The bbox as a dictionary
        - bbox_dict_latlng: Dict[str, float] - The bbox as a dictionary in lat/long coordinates
        - bbox_geojson: Dict[str, Any] - The bbox as a GeoJSON object
        - area_latlng: float - The area in lat/long coordinates
        - area: float - The area in the original projection
        - geom: ogr.Geometry - The original geometry
        - geom_latlng: Union[str, ogr.Geometry] - The geometry in lat/long coordinates

    Raises
    ------
    TypeError
        If inputs are None or of invalid type
    ValueError
        If bbox is invalid or projection is invalid
        If coordinate transformation fails

    Examples
    --------
    >>> proj = osr.SpatialReference()
    >>> proj.ImportFromEPSG(4326)
    >>> result = _additional_bboxes([0, 1, 0, 1], proj)
    >>> isinstance(result, dict)
    True
    >>> all(key in result for key in ['bbox_latlng', 'bbox_wkt', 'area'])
    True
    """
    # Input validation
    if bbox_ogr is None:
        raise TypeError("bbox_ogr cannot be None")

    if projection_osr is None:
        raise TypeError("projection_osr cannot be None")

    if not isinstance(bbox_ogr, (list, tuple)):
        raise TypeError(f"bbox_ogr must be a list or tuple, got {type(bbox_ogr)}")

    if not isinstance(projection_osr, osr.SpatialReference):
        raise TypeError(f"projection_osr must be osr.SpatialReference, got {type(projection_osr)}")

    if not _check_is_valid_bbox(bbox_ogr):
        raise ValueError(f"Invalid bbox format: {bbox_ogr}")

    try:
        # Create copy of input projection to avoid modifying the original
        original_projection = osr.SpatialReference()
        original_projection.ImportFromWkt(projection_osr.ExportToWkt())

        # Get WGS84 projection and reproject bbox
        latlng_projection = utils_projection._get_default_projection_osr()
        bbox_ogr_latlng = utils_projection.reproject_bbox(bbox_ogr, original_projection, latlng_projection)

        # Handle infinite values in lat/long bbox
        world = False
        if any(np.isinf(val) for val in bbox_ogr_latlng):
            world = True
            bbox_ogr_latlng = [
                -179.999999 if np.isinf(bbox_ogr_latlng[0]) else bbox_ogr_latlng[0],
                180.0 if np.isinf(bbox_ogr_latlng[1]) else bbox_ogr_latlng[1],
                -89.999999 if np.isinf(bbox_ogr_latlng[2]) else bbox_ogr_latlng[2],
                90.0 if np.isinf(bbox_ogr_latlng[3]) else bbox_ogr_latlng[3]
            ]

        # Convert coordinates to float for consistency
        x_min, x_max, y_min, y_max = map(float, bbox_ogr)
        latlng_x_min, latlng_x_max, latlng_y_min, latlng_y_max = map(float, bbox_ogr_latlng)

        # Create geometries
        bbox_geom = _get_geom_from_bbox(bbox_ogr)
        bbox_geom_latlng = _get_geom_from_bbox(bbox_ogr_latlng)
        geom = bbox_geom

        # Get transformed geometry
        if world:
            geom_latlng = bbox_geom_latlng
        else:
            geom_latlng = _get_bounds_from_bbox_as_geom(bbox_ogr, original_projection)

        # Create WKT strings
        bbox_wkt = _get_wkt_from_bbox(bbox_ogr)
        bbox_wkt_latlng = _get_wkt_from_bbox(bbox_ogr_latlng)

        # Calculate areas
        try:
            geom_latlng_geom = (ogr.CreateGeometryFromWkt(geom_latlng)
                               if isinstance(geom_latlng, str) else geom_latlng)
            area_latlng = float(geom_latlng_geom.GetArea())
            area = float(geom.GetArea())
        except Exception as e:
            raise ValueError(f"Failed to calculate areas: {str(e)}") from e

        return {
            "bbox_latlng": [float(x) for x in bbox_ogr_latlng],
            "bbox_wkt": bbox_wkt,
            "bbox_wkt_latlng": bbox_wkt_latlng,
            "bbox_geom": bbox_geom,
            "bbox_geom_latlng": bbox_geom_latlng,
            "bbox_gdal": _get_gdal_bbox_from_ogr_bbox(bbox_ogr),
            "bbox_gdal_latlng": _get_gdal_bbox_from_ogr_bbox(bbox_ogr_latlng),
            "bbox_dict": {
                "x_min": float(x_min),
                "x_max": float(x_max),
                "y_min": float(y_min),
                "y_max": float(y_max)
            },
            "bbox_dict_latlng": {
                "x_min": float(latlng_x_min),
                "x_max": float(latlng_x_max),
                "y_min": float(latlng_y_min),
                "y_max": float(latlng_y_max)
            },
            "bbox_geojson": _get_geojson_from_bbox(bbox_ogr_latlng),
            "area_latlng": area_latlng,
            "area": area,
            "geom": geom,
            "geom_latlng": geom_latlng,
        }

    except Exception as e:
        raise ValueError(f"Failed to generate additional bboxes: {str(e)}") from e


def _get_vector_from_geom(
    geom: ogr.Geometry,
    out_path: Optional[str] = None,
    name: Optional[str] = "converted_geom",
    prefix: Optional[str] = "",
    suffix: Optional[str] = "",
    add_uuid: Optional[bool] = False,
    add_timestamp: Optional[bool] = False,
    projection_osr: Optional[osr.SpatialReference] = None,
) -> str:
    """Converts a geometry to a vector file.

    Parameters
    ----------
    geom : ogr.Geometry
        The geometry to convert.
    out_path : Optional[str], optional
        Output path for the vector file. If None, creates a temporary file.
    name : Optional[str], optional
        Name for the layer. Defaults to "converted_geom".
    prefix : Optional[str], optional
        Prefix for the output filename. Defaults to "".
    suffix : Optional[str], optional
        Suffix for the output filename. Defaults to "".
    add_uuid : Optional[bool], optional
        Whether to add a UUID to the filename. Defaults to False.
    add_timestamp : Optional[bool], optional
        Whether to add a timestamp to the filename. Defaults to False.
    projection_osr : Optional[osr.SpatialReference], optional
        Projection for the vector. If None, uses WGS84.

    Returns
    -------
    str
        Path to the created vector file.

    Raises
    ------
    TypeError
        If input types are invalid.
    ValueError
        If geometry is invalid or vector creation fails.
    RuntimeError
        If GDAL operations fail.

    Examples
    --------
    >>> geom = ogr.CreateGeometryFromWkt('POINT (0 0)')
    >>> path = _get_vector_from_geom(geom)
    >>> isinstance(path, str)
    True
    >>> _get_vector_from_geom(None)
    Raises TypeError
    """
    # Input validation
    if geom is None:
        raise TypeError("geom cannot be None")

    if not isinstance(geom, ogr.Geometry):
        raise TypeError(f"geom must be ogr.Geometry, got {type(geom)}")

    if not geom.IsValid():
        raise ValueError("Input geometry is not valid")

    # Validate string parameters
    name = str(name) if name is not None else "converted_geom"
    prefix = str(prefix) if prefix is not None else ""
    suffix = str(suffix) if suffix is not None else ""

    # Handle output path
    try:
        if out_path is None:
            path = utils_path._get_temp_filepath(
                f"{name}.gpkg",
                prefix=prefix,
                suffix=suffix,
                add_uuid=bool(add_uuid),
                add_timestamp=bool(add_timestamp),
            )
        else:
            if not isinstance(out_path, str):
                raise TypeError(f"out_path must be a string, got {type(out_path)}")

            if not utils_path._check_is_valid_output_filepath(out_path):
                raise ValueError(f"Invalid output path: {out_path}")

            path = out_path

    except Exception as e:
        raise ValueError(f"Failed to create output path: {str(e)}") from e

    # Create vector
    try:
        # Get driver
        driver_name = utils_gdal._get_vector_driver_name_from_path(path)
        driver = ogr.GetDriverByName(driver_name)
        if driver is None:
            raise RuntimeError(f"Could not get driver for: {driver_name}")

        # Delete if exists
        if utils_path._check_file_exists(path):
            driver.DeleteDataSource(path)

        # Create datasource
        vector_ds = driver.CreateDataSource(path)
        if vector_ds is None:
            raise RuntimeError(f"Could not create vector at: {path}")

        # Set projection
        proj = (projection_osr if projection_osr is not None
               else utils_projection._get_default_projection_osr())
        if not isinstance(proj, osr.SpatialReference):
            raise TypeError("Invalid projection type")

        # Create layer
        layer = vector_ds.CreateLayer(name, proj, geom.GetGeometryType())
        if layer is None:
            raise RuntimeError("Could not create layer")

        # Create feature
        feature_defn = layer.GetLayerDefn()
        if feature_defn is None:
            raise RuntimeError("Could not get layer definition")

        feature = ogr.Feature(feature_defn)
        if feature is None:
            raise RuntimeError("Could not create feature")

        if feature.SetGeometry(geom) != 0:
            raise RuntimeError("Could not set geometry")

        if layer.CreateFeature(feature) != 0:
            raise RuntimeError("Could not create feature in layer")

    except Exception as e:
        # Cleanup in case of error
        if 'vector_ds' in locals():
            vector_ds = None
        if utils_path._check_file_exists(path):
            driver.DeleteDataSource(path)
        raise RuntimeError(f"Failed to create vector: {str(e)}") from e

    finally:
        # Cleanup
        if 'feature' in locals():
            feature = None
        if 'vector_ds' in locals():
            vector_ds = None

    # Verify file was created
    if not utils_path._check_file_exists(path):
        raise RuntimeError(f"Vector file was not created at: {path}")

    return path
