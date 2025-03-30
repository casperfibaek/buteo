"""### Bounding box source functions. ###

Functions to get bounding boxes from various sources.

There are two different formats for bounding boxes used by GDAL:</br>
OGR:  `[x_min, x_max, y_min, y_max]`</br>
WARP: `[x_min, y_min, x_max, y_max]`</br>

_If nothing else is stated, the OGR format is used._
"""

# Standard library
from typing import List, Union, Optional

# External
import numpy as np
from osgeo import ogr, osr, gdal

# Internal
from buteo.utils.bbox_validation import _check_is_valid_bbox, _check_is_valid_bbox_latlng
from buteo.utils import utils_projection


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
        # Import here to avoid circular imports
        from buteo.utils.bbox_operations import _get_bbox_from_geotransform
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
            from buteo.utils.bbox_operations import _get_union_bboxes
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
