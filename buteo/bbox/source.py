"""Functions for extracting bounding boxes from various geospatial data sources."""

# Standard library
from typing import List, Union, Optional, Sequence, Tuple # Removed unused Type

# External
import numpy as np
from osgeo import ogr, osr, gdal

from beartype import beartype

# Internal
# Import locally used operations functions at top level
from buteo.bbox.operations import _get_bbox_from_geotransform, _get_union_bboxes
from buteo.bbox.validation import _check_is_valid_bbox, _check_is_valid_bbox_latlng
from buteo.utils import utils_projection

# Type Aliases
BboxType = Sequence[Union[int, float]]
DatasetType = Union[str, gdal.Dataset, ogr.DataSource]
DatasetListType = Sequence[DatasetType]


def _get_bbox_from_raster(
    raster_dataset: gdal.Dataset,
) -> List[float]:
    """Extracts the OGR bounding box from an opened GDAL raster dataset.

    The bounding box is returned in the dataset's native projection.

    Parameters
    ----------
    raster_dataset : gdal.Dataset
        An opened GDAL raster dataset object.

    Returns
    -------
    List[float]
        An OGR formatted bounding box: `[x_min, x_max, y_min, y_max]`.

    Raises
    ------
    TypeError
        If `raster_dataset` is not a `gdal.Dataset`.
    ValueError
        If the dataset has invalid dimensions (<= 0) or is missing
        a valid geotransform, or if bbox calculation fails.
    """

    # Input validation
    # Added check for None which was missing in the previous refactor attempt
    if raster_dataset is None:
        raise TypeError("Input raster_dataset cannot be None.")
    if not isinstance(raster_dataset, gdal.Dataset):
        raise TypeError(f"Input must be a gdal.Dataset, got {type(raster_dataset)}")

    # Get dimensions
    x_size: int = raster_dataset.RasterXSize
    y_size: int = raster_dataset.RasterYSize
    if x_size <= 0 or y_size <= 0:
        raise ValueError(f"Invalid raster dimensions: {x_size}x{y_size}. Must be > 0.")

    # Get geotransform
    # Added type hint for geotransform variable
    geotransform: Optional[Tuple[float, ...]] = raster_dataset.GetGeoTransform()
    if geotransform is None:
        raise ValueError("Could not retrieve geotransform from raster dataset.")

    try:
        # _get_bbox_from_geotransform handles geotransform validation
        bbox = _get_bbox_from_geotransform(list(geotransform), x_size, y_size)
    except (ValueError, TypeError) as e:
        # Catch errors from validation or calculation
        raise ValueError(f"Failed to compute bbox from geotransform: {e!s}") from e

    # _get_bbox_from_geotransform already returns List[float]
    return bbox


def _get_bbox_from_vector(
    vector_datasource: ogr.DataSource,
) -> List[float]:
    """Extracts the combined OGR bounding box from all layers in an OGR DataSource.

    The bounding box is returned in the datasource's native projection(s).
    Assumes all layers share the same projection or that the union across
    projections is meaningful.

    Parameters
    ----------
    vector_datasource : ogr.DataSource
        An opened OGR vector datasource object.

    Returns
    -------
    List[float]
        An OGR formatted bounding box encompassing all layers:
        `[x_min, x_max, y_min, y_max]`.

    Raises
    ------
    TypeError
        If `vector_datasource` is not an `ogr.DataSource`.
    ValueError
        If the datasource contains no layers, or if the extent cannot be
        computed for any layer, or if the combined extent results in
        infinite or NaN values.
    """
    # Input validation
    # Added check for None which was missing in the previous refactor attempt
    if vector_datasource is None:
        raise TypeError("Input vector_datasource cannot be None.")
    if not isinstance(vector_datasource, ogr.DataSource):
        raise TypeError(f"Input must be an ogr.DataSource, got {type(vector_datasource)}")

    layer_count: int = vector_datasource.GetLayerCount()
    if layer_count == 0:
        raise ValueError("Input vector datasource contains no layers.")

    # Initialize overall bounds
    overall_x_min = float('inf')
    overall_x_max = float('-inf')
    overall_y_min = float('inf')
    overall_y_max = float('-inf')

    # Iterate through layers and combine extents
    for i in range(layer_count):
        layer: Optional[ogr.Layer] = vector_datasource.GetLayerByIndex(i)
        if layer is None:
            # Should generally not happen if layer_count > 0, but check anyway
            raise ValueError(f"Could not access layer at index {i}.")

        try:
            # GetExtent() returns (minX, maxX, minY, maxY)
            layer_extent: Tuple[float, float, float, float] = layer.GetExtent()
        except RuntimeError as e: # More specific exception
            # Catch potential GDAL errors during GetExtent
            raise ValueError(f"Could not compute extent for layer {i} ('{layer.GetName()}'): {e!s}") from e

        # Check for invalid extent values from the layer
        if any(np.isnan(v) for v in layer_extent):
            raise ValueError(f"Layer {i} ('{layer.GetName()}') extent contains NaN values: {layer_extent}")

        # Update overall bounds
        overall_x_min = min(overall_x_min, layer_extent[0])
        overall_x_max = max(overall_x_max, layer_extent[1])
        overall_y_min = min(overall_y_min, layer_extent[2])
        overall_y_max = max(overall_y_max, layer_extent[3])

    # Check if the combined bounds are valid (finite)
    combined_bbox = [overall_x_min, overall_x_max, overall_y_min, overall_y_max]
    if any(np.isinf(v) for v in combined_bbox):
        # This might happen if a layer extent was infinite or calculation failed
        raise ValueError(f"Could not compute valid finite bounds from vector layers. Combined: {combined_bbox}")

    # Final validation of the combined bbox
    if not _check_is_valid_bbox(combined_bbox):
        raise ValueError(f"Combined bounding box from vector layers is invalid: {combined_bbox}")

    return combined_bbox


def _get_bbox_from_vector_layer(
    vector_layer: ogr.Layer,
) -> List[float]:
    """Extracts the OGR bounding box from a single OGR Layer.

    The bounding box is returned in the layer's native projection.

    Parameters
    ----------
    vector_layer : ogr.Layer
        An opened OGR vector layer object.

    Returns
    -------
    List[float]
        An OGR formatted bounding box: `[x_min, x_max, y_min, y_max]`.

    Raises
    ------
    TypeError
        If `vector_layer` is not an `ogr.Layer`.
    ValueError
        If the layer contains no features, or if its extent cannot be
        computed or results in invalid (NaN, infinite) values.
    """
    # Input validation
    if not isinstance(vector_layer, ogr.Layer):
        raise TypeError(f"Input must be an ogr.Layer, got {type(vector_layer)}")

    # Check if layer has features
    if vector_layer.GetFeatureCount() == 0:
        raise ValueError(f"Input vector layer '{vector_layer.GetName()}' contains no features.")

    # Get extent
    try:
        # GetExtent() returns (minX, maxX, minY, maxY)
        layer_extent: Tuple[float, float, float, float] = vector_layer.GetExtent()
    except Exception as e:
        raise ValueError(f"Could not compute extent for layer '{vector_layer.GetName()}': {str(e)}") from e

    # Check for invalid values
    if any(np.isnan(v) for v in layer_extent):
        raise ValueError(f"Layer '{vector_layer.GetName()}' extent contains NaN values: {layer_extent}")
    if any(np.isinf(v) for v in layer_extent):
        raise ValueError(f"Layer '{vector_layer.GetName()}' extent contains infinite values: {layer_extent}")

    # OGR format is [x_min, x_max, y_min, y_max]
    bbox_ogr = [layer_extent[0], layer_extent[1], layer_extent[2], layer_extent[3]]

    # Final validation of the bbox
    if not _check_is_valid_bbox(bbox_ogr):
        raise ValueError(f"Bounding box from layer '{vector_layer.GetName()}' is invalid: {bbox_ogr}")

    return bbox_ogr


# Helper for get_bbox_from_dataset
def _get_bbox_from_opened_dataset(
    ds_object: Union[gdal.Dataset, ogr.DataSource],
) -> List[float]:
    """Extracts bbox from an already opened GDAL/OGR dataset."""
    if isinstance(ds_object, gdal.Dataset):
        assert isinstance(ds_object, gdal.Dataset) # Assert for type checker
        return _get_bbox_from_raster(ds_object)

    assert isinstance(ds_object, ogr.DataSource) # Assert for type checker
    return _get_bbox_from_vector(ds_object)


# Helper for get_bbox_from_dataset
def _get_bbox_from_path(
    dataset_path: str,
) -> List[float]:
    """Opens a dataset from path and extracts bbox."""
    if not dataset_path.strip():
        raise ValueError("Dataset path string cannot be empty.")

    opened_ds: Optional[Union[gdal.Dataset, ogr.DataSource]] = None # Track opened dataset for finally block
    error_handler_pushed = False

    try:
        # Suppress GDAL errors during open attempts
        gdal.PushErrorHandler("CPLQuietErrorHandler")
        error_handler_pushed = True
        raster_ds: Optional[gdal.Dataset] = None
        vector_ds: Optional[ogr.DataSource] = None

        try:
            # Try opening as raster
            raster_ds = gdal.Open(dataset_path, gdal.GA_ReadOnly)
        except RuntimeError: # Catch if gdal.Open raises error for non-raster
            raster_ds = None
        finally:
            # Pop handler regardless of gdal.Open success/failure
            if error_handler_pushed:
                gdal.PopErrorHandler()
                error_handler_pushed = False # Reset flag

        if raster_ds is not None:
            opened_ds = raster_ds # Assign for finally block cleanup
            return _get_bbox_from_raster(raster_ds) # Pass specific type

        # If gdal.Open failed or returned None, try ogr.Open
        vector_ds = ogr.Open(dataset_path, gdal.GA_ReadOnly)
        if vector_ds is not None:
            opened_ds = vector_ds # Assign for finally block cleanup
            return _get_bbox_from_vector(vector_ds) # Pass specific type

        # If both failed
        raise RuntimeError(f"Could not open dataset as either raster or vector: {dataset_path}")

    finally:
        # Ensure locally opened datasets are closed
        if opened_ds is not None:
            # GDAL Datasets and OGR DataSources are closed by dereferencing in Python's GC
            opened_ds = None
        # Ensure error handler is popped if it was pushed and not popped earlier
        # This check might be redundant now as it's handled in the inner finally, but keep for safety
        if error_handler_pushed:
            gdal.PopErrorHandler()


@beartype
def get_bbox_from_dataset(
    dataset: DatasetType,
) -> List[float]:
    """Extracts the OGR bounding box from a raster or vector dataset.

    The bounding box is returned in the dataset's native projection.

    Refactored to use helper functions for clarity and reduced complexity.

    Parameters
    ----------
    dataset : DatasetType (str | gdal.Dataset | ogr.DataSource)
        A path to a dataset (raster or vector) or an opened GDAL/OGR
        dataset object.

    Returns
    -------
    List[float]
        The bounding box in OGR format: `[x_min, x_max, y_min, y_max]`.

    Raises
    ------
    TypeError
        If `dataset` is not a string path, `gdal.Dataset`, or `ogr.DataSource`.
    ValueError
        If `dataset` path is an empty string, or if internal bbox extraction fails.
    RuntimeError
        If the dataset cannot be opened (if path is provided), or if
        bbox computation fails unexpectedly.

    Examples
    --------
    >>> # Assuming 'raster.tif' and 'vector.shp' exist and are valid
    >>> # bbox_raster = get_bbox_from_dataset('raster.tif')
    >>> # bbox_vector = get_bbox_from_dataset('vector.shp')
    >>> # With opened datasets:
    >>> # ds_raster = gdal.Open('raster.tif')
    >>> # bbox_opened_raster = get_bbox_from_dataset(ds_raster)
    >>> # ds_vector = ogr.Open('vector.shp')
    >>> # bbox_opened_vector = get_bbox_from_dataset(ds_vector)
    >>> get_bbox_from_dataset(None)
    Raises TypeError: Dataset must be a string, gdal.Dataset, or ogr.DataSource. Got: <class 'NoneType'>
    """
    # Input validation
    if not isinstance(dataset, (str, gdal.Dataset, ogr.DataSource)):
        raise TypeError(f"Dataset must be a string, gdal.Dataset, or ogr.DataSource. Got: {type(dataset)}")

    # Input validation (handled by beartype)

    try:
        if isinstance(dataset, (gdal.Dataset, ogr.DataSource)):
            return _get_bbox_from_opened_dataset(dataset)

        if isinstance(dataset, str):
            return _get_bbox_from_path(dataset)

        # Should be unreachable due to beartype validation, but included for safety
        raise TypeError(f"Unexpected dataset type: {type(dataset)}")

    except (ValueError, TypeError, RuntimeError) as e:
        # Re-raise exceptions from internal functions or open attempts
        raise type(e)(f"Error processing dataset: {e!s}") from e
    except Exception as e:
        # Catch unexpected errors
        raise RuntimeError(f"An unexpected error occurred processing dataset: {e!s}") from e


def _get_utm_zone_from_bbox(
    bbox_ogr_latlng: BboxType,
) -> str:
    """Determines the UTM zone containing the center of a WGS84 bounding box.

    Parameters
    ----------
    bbox_ogr_latlng : BboxType
        An OGR formatted bounding box in WGS84 coordinates (longitude, latitude):
        `[lng_min, lng_max, lat_min, lat_max]`.

    Returns
    -------
    str
        The UTM zone identifier string (e.g., "32N", "10S").

    Raises
    ------
    ValueError
        If `bbox_ogr_latlng` is not a valid WGS84 bbox, or if the
        midpoint calculation results in invalid coordinates.

    Notes
    -----
    Calculates the midpoint longitude and latitude and uses helper functions
    to determine the corresponding UTM zone EPSG code, then formats the zone string.
    Handles longitude wrapping across the 180/-180 meridian.

    Examples
    --------
    >>> _get_utm_zone_from_bbox([8.0, 9.0, 50.0, 51.0]) # Central Europe
    '32N'
    >>> _get_utm_zone_from_bbox([-75.0, -74.0, 40.0, 41.0]) # US East Coast
    '18N'
    >>> _get_utm_zone_from_bbox([179.0, -179.0, -10.0, -9.0]) # Crosses dateline
    '60S' # Midpoint is near 180/-180
    >>> _get_utm_zone_from_bbox(None)
    Raises ValueError: Invalid bbox format: None
    """
    # Input validation (checks format and lat/lng bounds)
    if not _check_is_valid_bbox_latlng(bbox_ogr_latlng):
        # Reuse validation error message if applicable
        # Check if it's a valid bbox first
        if not _check_is_valid_bbox(bbox_ogr_latlng):
            raise ValueError(f"Invalid bbox format: {bbox_ogr_latlng}")
        # If it's a valid bbox but not lat/lng
        raise ValueError(f"Bbox is not in valid WGS84 lat/long format: {bbox_ogr_latlng}")

    try:
        # Convert to float for calculation
        lng_min, lng_max, lat_min, lat_max = map(float, bbox_ogr_latlng)

        # Calculate midpoint longitude, handling dateline crossing
        # If min > max, it crosses the dateline.
        if lng_min > lng_max:
            # Center longitude calculation needs care across dateline
            # Example: 179 to -179. Width = 360 + (-179) - 179 = 2. Midpoint = (179 + 2/2) % 360 = 180.
            # Convert 180 to -180 for zone calculation consistency? Or handle zone 60/1 specially.
            # Let's calculate midpoint relative to 0-360 first
            mid_lng_360 = (lng_min + (360.0 + lng_max - lng_min) / 2.0) % 360.0
            # Convert to -180 to 180
            mid_lng = mid_lng_360 if mid_lng_360 <= 180.0 else mid_lng_360 - 360.0
        else:
            # Standard midpoint calculation
            mid_lng = (lng_min + lng_max) / 2.0

        # Calculate midpoint latitude
        mid_lat = (lat_min + lat_max) / 2.0

        # Validate calculated midpoint (should be within bounds if input was valid)
        if not (-180.0 <= mid_lng <= 180.0 and -90.0 <= mid_lat <= 90.0):
            # This indicates a potential logic error if input validation passed
            raise ValueError(f"Internal error: Calculated invalid midpoint ({mid_lng}, {mid_lat}) from valid bbox.")

        # Calculate UTM zone number from the midpoint longitude
        # Formula: zone = floor((longitude + 180) / 6) + 1
        # Handle the 180th meridian edge case: it belongs to zone 60.
        if np.isclose(mid_lng, 180.0):
            zone_num = 60
        else:
            zone_num = int(np.floor((mid_lng + 180.0) / 6.0) + 1)

        # Clamp zone number to be within 1-60 range just in case of floating point issues near edges
        zone_num = max(1, min(60, zone_num))

        # Determine hemisphere based on midpoint latitude
        hemisphere = "N" if mid_lat >= 0 else "S"

        return f"{zone_num}{hemisphere}"

    except (ValueError, TypeError) as e:
        # Catch errors from mapping, calculation, or utility function
        raise ValueError(f"Error calculating UTM zone from bbox: {e!s}") from e


def _get_utm_zone_from_dataset(
    dataset: DatasetType,
) -> str:
    """Determines the UTM zone containing the center of a dataset's bounding box.

    Reprojects the dataset's bounding box to WGS84 if necessary, then
    calculates the UTM zone of the midpoint.

    Parameters
    ----------
    dataset : DatasetType (str | gdal.Dataset | ogr.DataSource)
        A path to a dataset or an opened GDAL/OGR dataset object.
        Must have valid spatial reference information.

    Returns
    -------
    str
        The UTM zone identifier string (e.g., "32N", "10S").

    Raises
    ------
    TypeError
        If `dataset` is not a valid type.
    ValueError
        If the dataset cannot be opened, lacks projection information,
        or if UTM zone calculation fails.
    RuntimeError
        For unexpected errors during dataset processing or projection.

    Examples
    --------
    >>> # Assuming 'raster_utm32n.tif' exists in UTM 32N
    >>> # _get_utm_zone_from_dataset('raster_utm32n.tif')
    >>> # '32N'
    >>> # Assuming 'vector_wgs84.shp' exists in WGS84
    >>> # _get_utm_zone_from_dataset('vector_wgs84.shp')
    >>> # '...' # Depends on the vector's location
    >>> _get_utm_zone_from_dataset(None)
    Raises TypeError: Dataset must be a string, gdal.Dataset, or ogr.DataSource. Got: <class 'NoneType'>
    """
    # dataset type validation happens in get_bbox_from_dataset and _get_projection_from_dataset

    try:
        # Get native bbox and projection
        native_bbox: BboxType = get_bbox_from_dataset(dataset)
        source_proj: Optional[osr.SpatialReference] = utils_projection._get_projection_from_dataset(dataset)

        if source_proj is None:
            raise ValueError("Could not determine spatial reference system for the dataset.")

        # Get target projection (WGS84)
        target_proj: osr.SpatialReference = utils_projection._get_default_projection_osr()

        # Reproject bbox to WGS84 if necessary
        if utils_projection._check_projections_match(source_proj, target_proj):
            bbox_latlng = native_bbox
        else:
            # reproject_bbox handles transformation - ensure input is List
            bbox_latlng = utils_projection.reproject_bbox(list(native_bbox), source_proj, target_proj)

        # Get UTM zone from the WGS84 bbox
        utm_zone = _get_utm_zone_from_bbox(bbox_latlng)

        return utm_zone

    except (ValueError, TypeError, RuntimeError) as e:
        # Catch errors from helper functions
        raise ValueError(f"Failed to determine UTM zone for dataset: {e!s}") from e
    except Exception as e:
        # Catch unexpected errors
        raise RuntimeError(f"An unexpected error occurred getting UTM zone from dataset: {e!s}") from e


def _get_utm_zone_from_dataset_list(
    datasets: DatasetListType,
) -> str:
    """Determines the single UTM zone containing the center of the combined
    bounding box of multiple datasets.

    Calculates the union of all dataset bounding boxes (reprojected to WGS84),
    finds the midpoint of the union, and determines its UTM zone.

    Parameters
    ----------
    datasets : DatasetListType (Sequence[str | gdal.Dataset | ogr.DataSource])
        A sequence (list, tuple) of dataset paths or opened GDAL/OGR dataset objects.
        All datasets must have valid spatial reference information.

    Returns
    -------
    str
        The single UTM zone identifier string (e.g., "32N", "10S") for the
        center of the combined extent.

    Raises
    ------
    TypeError
        If `datasets` is not a sequence, or if any item within is not a
        valid dataset type.
    ValueError
        If `datasets` is empty, if any dataset cannot be processed (opened,
        no projection, etc.), or if the combined UTM zone cannot be determined.
    RuntimeError
        For unexpected errors during processing.

    Examples
    --------
    >>> # Assuming datasets covering areas in UTM 32N exist
    >>> # _get_utm_zone_from_dataset_list(['raster1_utm32.tif', 'vector2_utm32.shp'])
    >>> # '32N'
    >>> # Assuming datasets span across UTM 32N and 33N, centered in 32N
    >>> # _get_utm_zone_from_dataset_list(['raster_utm32.tif', 'vector_utm33.shp'])
    >>> # '32N' # Depends on the exact center of the union
    >>> _get_utm_zone_from_dataset_list([])
    Raises ValueError: Dataset list cannot be empty.
    """
    # Input validation
    if not isinstance(datasets, (list, tuple)):
        raise TypeError(f"Input 'datasets' must be a list or tuple, got {type(datasets)}")
    if not datasets:
        raise ValueError("Input 'datasets' list cannot be empty.")
    if not all(isinstance(d, (str, gdal.Dataset, ogr.DataSource)) for d in datasets):
        raise TypeError("All items in 'datasets' must be a string, gdal.Dataset, or ogr.DataSource.")

    latlng_bboxes: List[BboxType] = []
    target_proj: osr.SpatialReference = utils_projection._get_default_projection_osr()

    # Process each dataset to get its WGS84 bbox
    for i, dataset in enumerate(datasets):
        ds_repr = dataset if isinstance(dataset, str) else f"dataset at index {i}"
        try:
            native_bbox = get_bbox_from_dataset(dataset)
            source_proj = utils_projection._get_projection_from_dataset(dataset)

            if source_proj is None:
                # Attempt to represent dataset for error message
                raise ValueError(f"Could not determine projection for {ds_repr}.")

            # Reproject to WGS84 if necessary
            if utils_projection._check_projections_match(source_proj, target_proj):
                latlng_bboxes.append(native_bbox)
            else:
                # Ensure input is List for reproject_bbox
                latlng_bboxes.append(utils_projection.reproject_bbox(list(native_bbox), source_proj, target_proj))

        except (ValueError, TypeError, RuntimeError) as e:
            raise ValueError(f"Failed to process {ds_repr}: {e!s}") from e
        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred processing {ds_repr}: {e!s}") from e

    # Calculate the union of all WGS84 bounding boxes
    if not latlng_bboxes: # Should be caught earlier, but safety check
        raise ValueError("No valid WGS84 bounding boxes were obtained from the datasets.")

    try:
        # Use imported _get_union_bboxes
        union_bbox = latlng_bboxes[0]
        for next_bbox in latlng_bboxes[1:]:
            union_bbox = _get_union_bboxes(union_bbox, next_bbox)
    except ValueError as e:
        raise ValueError(f"Failed to compute union of WGS84 bounding boxes: {e!s}") from e

    # Get UTM zone from the center of the union bbox
    try:
        utm_zone = _get_utm_zone_from_bbox(union_bbox)
        return utm_zone
    except ValueError as e:
        raise ValueError(f"Failed to determine UTM zone from combined dataset extent: {e!s}") from e
