"""Internal functions for converting bounding boxes between different formats
(OGR bbox, OGR Geometry, WKT, GeoJSON, temporary vector file) and projections.
"""

# Standard library
from typing import List, Union, Dict, Optional, Sequence, Tuple
from uuid import uuid4
from warnings import warn

# External
import numpy as np
from osgeo import ogr, osr, gdal

# Internal
from buteo.bbox.validation import _check_is_valid_bbox
from buteo.utils import utils_projection, utils_path, utils_gdal

# Type Aliases
BboxType = Sequence[Union[int, float]]
PointType = Sequence[Union[int, float]]


def _get_geom_from_bbox(
    bbox_ogr: BboxType,
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

    Warns
    -----
    UserWarning
        If the input bbox is all zeros `[0, 0, 0, 0]`, a very small
        polygon is created instead to avoid potential issues.

    Examples
    --------
    >>> bbox = [0.0, 1.0, 0.0, 1.0]
    >>> geom = _get_geom_from_bbox(bbox)
    >>> isinstance(geom, ogr.Geometry)
    True
    >>> geom.GetGeometryName()
    'POLYGON'
    >>> _get_geom_from_bbox([0, 0, 0, 0]) # Warns and creates small polygon
    <osgeo.ogr.Geometry; proxy of <Swig Object of type 'OGRGeometryShadow *' at ...>>
    >>> _get_geom_from_bbox([0, 1, np.nan, 1])
    Raises ValueError: Bounding box contains NaN values
    >>> _get_geom_from_bbox(None)
    Raises ValueError: Invalid bbox format: None
    """
    # Input validation (raises ValueError if invalid)
    if not _check_is_valid_bbox(bbox_ogr):
        raise ValueError(f"Invalid bbox format: {bbox_ogr}")

    # Convert to float, check for NaN
    try:
        x_min, x_max, y_min, y_max = map(float, bbox_ogr)
        if any(np.isnan(v) for v in [x_min, x_max, y_min, y_max]):
            raise ValueError("Bounding box contains NaN values")
    except (ValueError, TypeError) as e:
        raise ValueError(f"Could not convert bbox values to float: {e!s}") from e

    # Create geometry
    try:
        # Handle zero bbox case by creating a tiny polygon
        if x_min == 0.0 and x_max == 0.0 and y_min == 0.0 and y_max == 0.0:
            warn("Bounding box is zero ([0, 0, 0, 0]), creating a very small polygon instead.", UserWarning)
            ring = ogr.Geometry(ogr.wkbLinearRing)
            ring.AddPoint(0.0, 0.0)
            ring.AddPoint(1e-9, 0.0)  # Tiny offset
            ring.AddPoint(1e-9, 1e-9)
            ring.AddPoint(0.0, 1e-9)
            ring.AddPoint(0.0, 0.0)  # Close ring
        else:
            ring = ogr.Geometry(ogr.wkbLinearRing)
            ring.AddPoint(x_min, y_min)
            ring.AddPoint(x_max, y_min)
            ring.AddPoint(x_max, y_max)
            ring.AddPoint(x_min, y_max)
            ring.AddPoint(x_min, y_min)  # Close ring explicitly

        geom = ogr.Geometry(ogr.wkbPolygon)
        if geom.AddGeometry(ring) != ogr.OGRERR_NONE:
            raise ValueError("Failed to add ring to polygon geometry.")

        # Final validation check
        if not geom.IsValid():
            # Attempt to fix minor issues
            geom_fixed = geom.MakeValid()
            if geom_fixed is None or not geom_fixed.IsValid():
                raise ValueError("Created geometry is invalid, even after attempting MakeValid().")
            geom = geom_fixed  # Use the fixed geometry

        return geom

    except (RuntimeError, ValueError) as e:
        raise ValueError(f"Failed to create geometry from bbox: {e!s}") from e


def _get_bbox_from_geom(geom: ogr.Geometry) -> List[float]:
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
    # Input validation
    if not isinstance(geom, ogr.Geometry):
        raise TypeError(f"Input must be an ogr.Geometry object, got {type(geom)}")

    try:
        # GetEnvelope() returns (minX, maxX, minY, maxY)
        envelope: Tuple[float, float, float, float] = geom.GetEnvelope()
    except RuntimeError as e:  # More specific than Exception
        raise ValueError(f"Failed to compute geometry envelope: {e!s}") from e

    # Check for NaN values in envelope
    if any(np.isnan(v) for v in envelope):
        raise ValueError(f"Geometry envelope contains NaN values: {envelope}")

    # OGR format is [x_min, x_max, y_min, y_max]
    bbox_ogr = [envelope[0], envelope[1], envelope[2], envelope[3]]

    # Final validation of the generated bbox
    if not _check_is_valid_bbox(bbox_ogr):
        raise ValueError(f"Generated bounding box from geometry is invalid: {bbox_ogr}")

    return bbox_ogr


def _get_wkt_from_bbox(
    bbox_ogr: BboxType,
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
    >>> _get_wkt_from_bbox([0.0, 1.0, 2.0, 3.0])
    'POLYGON ((0.000000 2.000000, 1.000000 2.000000, 1.000000 3.000000, 0.000000 3.000000, 0.000000 2.000000))'
    >>> _get_wkt_from_bbox(None)
    Raises ValueError: Invalid bbox format: None
    """
    # Input validation (raises ValueError if invalid)
    if not _check_is_valid_bbox(bbox_ogr):
        raise ValueError(f"Invalid bbox format: {bbox_ogr}")

    # Convert to float, check for NaN
    try:
        x_min, x_max, y_min, y_max = map(float, bbox_ogr)
        if any(np.isnan(v) for v in [x_min, x_max, y_min, y_max]):
            raise ValueError("Bounding box contains NaN values")
    except (ValueError, TypeError) as e:
        raise ValueError(f"Could not convert bbox values to float: {e!s}") from e

    # Format WKT string with fixed precision (e.g., 6 decimal places)
    # Ensure closing coordinate is the same as the starting coordinate.
    wkt = (
        f"POLYGON (("
        f"{x_min:.6f} {y_min:.6f}, "
        f"{x_max:.6f} {y_min:.6f}, "
        f"{x_max:.6f} {y_max:.6f}, "
        f"{x_min:.6f} {y_max:.6f}, "
        f"{x_min:.6f} {y_min:.6f}"  # Closing point
        f"))"
    )

    return wkt


def _get_geojson_from_bbox(
    bbox_ogr: BboxType,
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
    >>> _get_geojson_from_bbox([0.0, 1.0, 2.0, 3.0])
    {'type': 'Polygon', 'coordinates': [[[0.0, 2.0], [1.0, 2.0], [1.0, 3.0], [0.0, 3.0], [0.0, 2.0]]]}
    >>> _get_geojson_from_bbox(None)
    Raises ValueError: Invalid bbox format: None
    """
    # Input validation (raises ValueError if invalid)
    if not _check_is_valid_bbox(bbox_ogr):
        raise ValueError(f"Invalid bbox format: {bbox_ogr}")

    # Convert to float, check for NaN
    try:
        x_min, x_max, y_min, y_max = map(float, bbox_ogr)
        if any(np.isnan(v) for v in [x_min, x_max, y_min, y_max]):
            raise ValueError("Bounding box contains NaN values")
    except (ValueError, TypeError) as e:
        raise ValueError(f"Could not convert bbox values to float: {e!s}") from e

    # Create GeoJSON coordinates list (must close the ring)
    coordinates = [
        [x_min, y_min],
        [x_max, y_min],
        [x_max, y_max],
        [x_min, y_max],
        [x_min, y_min]  # Closing coordinate
    ]

    # Construct GeoJSON dictionary
    geojson: Dict[str, Union[str, List[List[List[float]]]]] = {
        "type": "Polygon",
        "coordinates": [coordinates],  # GeoJSON Polygons require an outer list
    }

    return geojson


# Helper function for _get_vector_from_geom
def _create_vector_datasource(
    path: str,
    driver_name: str,
) -> Tuple[ogr.DataSource, ogr.Driver]:
    """Gets GDAL driver and creates an OGR DataSource."""
    driver = ogr.GetDriverByName(driver_name)
    if driver is None:
        raise RuntimeError(f"Could not get GDAL driver for format: {driver_name}")

    # Delete existing file if it exists (driver handles this)
    if utils_path._check_file_exists(path):
        if driver.DeleteDataSource(path) != ogr.OGRERR_NONE:
            # Warn instead of error if deletion fails, creation might still work
            warn(f"Could not delete existing file at {path}. Attempting to overwrite.", UserWarning)

    # Create datasource
    vector_ds = driver.CreateDataSource(path)
    if vector_ds is None:
        raise RuntimeError(f"Could not create vector datasource at: {path}")

    return vector_ds, driver


# Helper function for _get_vector_from_geom
def _write_geom_to_layer(
    vector_ds: ogr.DataSource,
    geom: ogr.Geometry,
    layer_name: str,
    projection_osr: osr.SpatialReference,
) -> None:
    """Creates a layer and writes a geometry feature to it."""
    try:
        # Create layer
        layer = vector_ds.CreateLayer(layer_name, projection_osr, geom.GetGeometryType())
        if layer is None:
            raise RuntimeError("Could not create layer in datasource.")

        # Create feature
        feature_defn = layer.GetLayerDefn()
        feature = ogr.Feature(feature_defn)
        if feature.SetGeometry(geom) != ogr.OGRERR_NONE:
            raise RuntimeError("Could not set geometry on feature.")

        if layer.CreateFeature(feature) != ogr.OGRERR_NONE:
            raise RuntimeError("Could not create feature in layer.")

        # Cleanup GDAL objects
        feature.Destroy()
        layer = None  # Dereference layer
        vector_ds.FlushCache()

    except RuntimeError as e: # Catch specific GDAL/OGR runtime errors
        # Let the caller handle cleanup of the datasource
        raise RuntimeError(f"Failed during layer/feature creation: {e!s}") from e


def _get_vector_from_bbox(
    bbox_ogr: BboxType,
    projection_osr: Optional[osr.SpatialReference] = None,
) -> str:
    """Converts an OGR formatted bounding box to a temporary vector file
    (GeoPackage) stored in GDAL's virtual memory filesystem (`/vsimem/`).

    Parameters
    ----------
    bbox_ogr : BboxType
        An OGR formatted bounding box: `[x_min, x_max, y_min, y_max]`.
    projection_osr : osr.SpatialReference, optional
        The spatial reference system (projection) for the output vector.
        If None, defaults to WGS84 (EPSG:4326). Default: None.

    Returns
    -------
    str
        The path to the created temporary vector file in `/vsimem/`.
        Example: `/vsimem/xxxxxxxx_extent.gpkg`

    Raises
    ------
    ValueError
        If `bbox_ogr` is invalid, or if vector creation fails (e.g.,
        GPKG driver unavailable, geometry creation error).
    TypeError
        If `projection_osr` is provided but is not a valid
        `osr.SpatialReference` object.
    RuntimeError
        If GDAL driver operations fail unexpectedly.

    Examples
    --------
    >>> proj = osr.SpatialReference()
    >>> _ = proj.ImportFromEPSG(4326) # Use _ to suppress output in doctest
    >>> path = _get_vector_from_bbox([0.0, 1.0, 0.0, 1.0], proj)
    >>> isinstance(path, str) and path.startswith('/vsimem/') and path.endswith('_extent.gpkg')
    True
    >>> # Example with default projection (WGS84)
    >>> path_default = _get_vector_from_bbox([-10, 10, -10, 10])
    >>> isinstance(path_default, str) and path_default.startswith('/vsimem/')
    True
    >>> _get_vector_from_bbox(None)
    Raises ValueError: Invalid bbox format: None
    """
    # Input validation (bbox checked by _get_geom_from_bbox)
    if projection_osr is not None and not isinstance(projection_osr, osr.SpatialReference):
        raise TypeError(f"projection_osr must be osr.SpatialReference or None, got {type(projection_osr)}")

    # Use WGS84 as default projection if none provided
    proj: osr.SpatialReference = projection_osr or utils_projection._get_default_projection_osr()

    extent_name: Optional[str] = None  # Define for potential cleanup
    extent_ds: Optional[ogr.DataSource] = None  # Define for potential cleanup
    driver: Optional[ogr.Driver] = None  # Define for potential cleanup

    try:
        # Create geometry from bbox (raises ValueError if bbox is invalid)
        geom = _get_geom_from_bbox(bbox_ogr)

        # Create unique filename in /vsimem/
        extent_name = f"/vsimem/{uuid4().hex}_extent.gpkg"

        # Get GeoPackage driver
        driver = ogr.GetDriverByName("GPKG")
        if driver is None:
            raise RuntimeError("GPKG driver is not available.")

        # Create vector dataset in virtual memory
        # ogr.CreateDataSource can return None on failure
        extent_ds = driver.CreateDataSource(extent_name)
        if extent_ds is None:
            raise RuntimeError(f"Failed to create vector dataset at {extent_name}")

        # Create layer
        # CreateLayer can return None on failure
        layer = extent_ds.CreateLayer("extent_ogr", proj, ogr.wkbPolygon)
        if layer is None:
            raise RuntimeError("Failed to create layer in vector dataset.")

        # Create feature and set geometry
        feature = ogr.Feature(layer.GetLayerDefn())
        if feature.SetGeometry(geom) != ogr.OGRERR_NONE:
            raise RuntimeError("Failed to set geometry on feature.")

        if layer.CreateFeature(feature) != ogr.OGRERR_NONE:
            raise RuntimeError("Failed to create feature in layer.")

        # Explicitly destroy feature and sync layer (best practice)
        feature.Destroy()
        layer.SyncToDisk()
        extent_ds.FlushCache()

        # Close dataset by dereferencing (Python garbage collection)
        extent_ds = None

        return extent_name

    except (RuntimeError, ValueError, TypeError) as e:
        # Cleanup: Attempt to delete the virtual file if creation failed partially
        if extent_name is not None:
            try:
                # Ensure dataset is closed before deleting
                if extent_ds is not None:
                    extent_ds = None
                if driver is None:
                    driver = ogr.GetDriverByName("GPKG")
                if driver is not None:
                    driver.DeleteDataSource(extent_name)
            except RuntimeError:  # More specific exception for GDAL/OGR errors
                # Ignore cleanup errors, prioritize original error
                pass
        # Re-raise the original error
        raise RuntimeError(f"Failed to create vector from bbox: {e!s}") from e


def _transform_point(
    point: PointType,
    transformer: osr.CoordinateTransformation
) -> List[float]:
    """Transforms a single coordinate pair using an OSR transformer.

    Parameters
    ----------
    point : PointType
        A coordinate pair to transform: `[x, y]` or `(x, y)`.
    transformer : osr.CoordinateTransformation
        The initialized OSR coordinate transformation object.

    Returns
    -------
    List[float]
        The transformed coordinate pair: `[x_transformed, y_transformed]`.

    Raises
    ------
    ValueError
        If the input `point` is not a sequence of 2 numbers, or if
        the transformation fails (e.g., `TransformPoint` returns None).
    TypeError
        If `transformer` is not a valid `osr.CoordinateTransformation`.
    """
    # Input validation
    if not isinstance(point, (list, tuple)) or len(point) != 2:
        raise ValueError("Input 'point' must be a sequence of 2 numbers (x, y).")
    if not all(isinstance(coord, (int, float)) for coord in point):
        raise ValueError("Coordinates in 'point' must be numeric.")
    if not isinstance(transformer, osr.CoordinateTransformation):
        raise TypeError("Input 'transformer' must be an osr.CoordinateTransformation object.")

    # Perform transformation
    try:
        # TransformPoint returns (x, y, z) tuple
        transformed: Optional[Tuple[float, float, float]] = transformer.TransformPoint(float(point[0]), float(point[1]))
        if transformed is None:
            # This can happen for invalid coordinates or transformer issues
            raise ValueError(f"Coordinate transformation failed for point {point}.")
        # Return only x, y as a list of floats
        return [transformed[0], transformed[1]]
    except (RuntimeError, ValueError, TypeError) as e:
        # Catch potential GDAL runtime errors or ValueErrors raised above
        raise ValueError(f"Failed to transform point {point}: {e!s}") from e


def _transform_bbox_coordinates(
    bbox_ogr: BboxType,
    transformer: osr.CoordinateTransformation,
) -> List[List[float]]:
    """Transforms the four corner coordinates of a bounding box using an
    OSR coordinate transformer.

    Parameters
    ----------
    bbox_ogr : BboxType
        An OGR formatted bounding box: `[x_min, x_max, y_min, y_max]`.
    transformer : osr.CoordinateTransformation
        The initialized OSR coordinate transformation object.

    Returns
    -------
    List[List[float]]
        A list containing the four transformed corner coordinate pairs:
        `[[x_ll, y_ll], [x_lr, y_lr], [x_ur, y_ur], [x_ul, y_ul]]`
        (ll=lower-left, lr=lower-right, ur=upper-right, ul=upper-left).

    Raises
    ------
    ValueError
        If `bbox_ogr` is invalid, or if transformation fails for any corner.
    TypeError
        If `transformer` is not a valid `osr.CoordinateTransformation`.
    """
    # Input validation (bbox validity checked within loop by _transform_point)
    if not isinstance(transformer, osr.CoordinateTransformation):
        raise TypeError("Input 'transformer' must be an osr.CoordinateTransformation object.")
    if not _check_is_valid_bbox(bbox_ogr):
        raise ValueError(f"Invalid OGR bounding box provided: {bbox_ogr}")

    # Define corners in standard order (e.g., lower-left, lower-right, upper-right, upper-left)
    x_min, x_max, y_min, y_max = map(float, bbox_ogr)
    corners: List[PointType] = [
        [x_min, y_min],  # lower-left
        [x_max, y_min],  # lower-right
        [x_max, y_max],  # upper-right
        [x_min, y_max],  # upper-left
    ]

    transformed_points: List[List[float]] = []
    # Use GDAL's quiet error handler during transformation attempts
    gdal.PushErrorHandler("CPLQuietErrorHandler")
    try:
        for corner in corners:
            transformed_points.append(_transform_point(corner, transformer))

        # Basic check if all points were transformed
        if len(transformed_points) != 4:
            raise ValueError("Failed to transform all four corner points.")

        return transformed_points
    except ValueError as e:
        # Re-raise value errors from _transform_point or the length check
        raise ValueError(f"Error transforming bbox coordinates: {e!s}") from e
    finally:
        # Ensure error handler is always popped
        gdal.PopErrorHandler()


def _create_polygon_from_points(
    points: List[List[float]],
    projection_osr: Optional[osr.SpatialReference] = None, # Added optional SRS
) -> ogr.Geometry:
    """Creates an OGR Polygon Geometry from a list of coordinate pairs.

    Assumes the points define the outer ring of the polygon. The ring
    will be explicitly closed by adding the first point to the end if needed.

    Parameters
    ----------
    points : List[List[float]]
        A list of coordinate pairs defining the polygon vertices:
        `[[x1, y1], [x2, y2], ..., [xn, yn]]`.
    projection_osr : osr.SpatialReference, optional
        The spatial reference system to assign to the created geometry.
        Default: None.

    Returns
    -------
    ogr.Geometry
        A valid OGR Polygon geometry.

    Raises
    ------
    ValueError
        If `points` is empty, contains invalid coordinates, or if
        geometry creation fails (e.g., adding ring, validation).
    TypeError
        If `projection_osr` is not an `osr.SpatialReference` or None.
    """
    # Input validation
    if not points:
        raise ValueError("Input 'points' list cannot be empty.")
    if not all(isinstance(p, (list, tuple)) and len(p) == 2 and
               all(isinstance(coord, (int, float)) for coord in p) for p in points):
        raise ValueError("Input 'points' must be a list of coordinate pairs (e.g., [[x1, y1], ...]).")
    if projection_osr is not None and not isinstance(projection_osr, osr.SpatialReference):
        raise TypeError("Input 'projection_osr' must be an osr.SpatialReference or None.")

    # Create linear ring
    ring = ogr.Geometry(ogr.wkbLinearRing)
    try:
        for point in points:
            ring.AddPoint(float(point[0]), float(point[1]))

        # Ensure the ring is closed
        first_point = points[0]
        last_point = points[-1]
        if first_point[0] != last_point[0] or first_point[1] != last_point[1]:
            ring.AddPoint(float(first_point[0]), float(first_point[1]))

    except (TypeError, ValueError) as e:
        raise ValueError(f"Error adding points to ring: {e!s}") from e

    # Create polygon from ring
    polygon = ogr.Geometry(ogr.wkbPolygon)
    if polygon.AddGeometry(ring) != ogr.OGRERR_NONE:
        raise ValueError("Failed to add ring to polygon geometry.")

    # Assign SRS if provided
    if projection_osr is not None:
        polygon.AssignSpatialReference(projection_osr)

    # Validate the created polygon
    if not polygon.IsValid():
        # Attempt to fix minor issues
        polygon_fixed = polygon.MakeValid()
        if polygon_fixed is None or not polygon_fixed.IsValid():
            raise ValueError("Created polygon geometry is invalid, even after attempting MakeValid().")
        polygon = polygon_fixed  # Use the fixed geometry

    return polygon


def _get_bounds_from_bbox_as_geom(
    bbox_ogr: BboxType,
    projection_osr: osr.SpatialReference,
) -> ogr.Geometry:
    """Transforms an OGR bounding box to WGS84 (EPSG:4326) and returns
    the potentially non-rectangular bounds as an OGR Polygon Geometry.

    Parameters
    ----------
    bbox_ogr : BboxType
        An OGR formatted bounding box: `[x_min, x_max, y_min, y_max]`
        in the source projection.
    projection_osr : osr.SpatialReference
        The spatial reference system (projection) of the input `bbox_ogr`.

    Returns
    -------
    ogr.Geometry
        An OGR Polygon geometry representing the transformed bounds in WGS84.
        This may not be rectangular if the transformation involves rotation or warping.

    Raises
    ------
    ValueError
        If `bbox_ogr` is invalid, `projection_osr` is invalid, or if
        the coordinate transformation or geometry creation fails.
    TypeError
        If `projection_osr` is not an `osr.SpatialReference` object.
    """
    # Input validation
    if not isinstance(projection_osr, osr.SpatialReference):
        raise TypeError(f"projection_osr must be an osr.SpatialReference object, got {type(projection_osr)}")
    # bbox_ogr validity checked by _transform_bbox_coordinates and _get_geom_from_bbox

    # Get target projection (WGS84)
    target_projection = utils_projection._get_default_projection_osr()

    # If already in WGS84, just convert bbox to geometry directly
    if utils_projection._check_projections_match(projection_osr, target_projection):
        try:
            # Create geometry and assign the WGS84 SRS
            geom = _get_geom_from_bbox(bbox_ogr)
            geom.AssignSpatialReference(target_projection)
            return geom
        except ValueError as e:
            raise ValueError(f"Failed to create geometry from WGS84 bbox: {e!s}") from e

    # Create coordinate transformer
    try:
        # Using AreaOfInterest might improve accuracy near poles/dateline
        options = osr.CoordinateTransformationOptions()
        options.SetAreaOfInterest(-180.0, -90.0, 180.0, 90.0)  # WGS84 bounds
        transformer = osr.CoordinateTransformation(projection_osr, target_projection, options)
    except RuntimeError as e: # More specific exception
        # Catch potential errors during transformer creation
        raise ValueError(f"Failed to create coordinate transformation: {e!s}") from e

    # Transform corner coordinates
    try:
        transformed_points = _transform_bbox_coordinates(bbox_ogr, transformer)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Failed to transform bbox coordinates: {e!s}") from e

    # Create polygon geometry from transformed points, assigning target SRS
    try:
        return _create_polygon_from_points(transformed_points, target_projection)
    except ValueError as e:
        raise ValueError(f"Failed to create polygon from transformed points: {e!s}") from e


def _get_bounds_from_bbox_as_wkt(
    bbox_ogr: BboxType,
    projection_osr: osr.SpatialReference,
) -> str:
    """Transforms an OGR bounding box to WGS84 (EPSG:4326) and returns
    the potentially non-rectangular bounds as a WKT Polygon string.

    Parameters
    ----------
    bbox_ogr : BboxType
        An OGR formatted bounding box: `[x_min, x_max, y_min, y_max]`
        in the source projection.
    projection_osr : osr.SpatialReference
        The spatial reference system (projection) of the input `bbox_ogr`.

    Returns
    -------
    str
        A WKT Polygon string representing the transformed bounds in WGS84.

    Raises
    ------
    ValueError
        If inputs are invalid or transformation/geometry creation fails.
    TypeError
        If `projection_osr` is not an `osr.SpatialReference` object.
    """
    # Get the transformed geometry
    try:
        geom = _get_bounds_from_bbox_as_geom(bbox_ogr, projection_osr)
        wkt = geom.ExportToWkt()
        if not isinstance(wkt, str):
            raise ValueError("ExportToWkt did not return a string.")
        return wkt
    except (ValueError, TypeError) as e:
        # Catch errors from _get_bounds_from_bbox_as_geom or ExportToWkt
        raise ValueError(f"Failed to get WKT from transformed bbox: {e!s}") from e


def _get_vector_from_geom(
    geom: ogr.Geometry,
    out_path: Optional[str] = None,
    name: str = "converted_geom",
    prefix: str = "",
    suffix: str = "",
    add_uuid: bool = False,
    add_timestamp: bool = False,
    projection_osr: Optional[osr.SpatialReference] = None,
) -> str:
    """Saves an OGR Geometry object to a vector file (default: temporary GeoPackage).

    Refactored to use helper functions for clarity and reduced complexity.

    Parameters
    ----------
    geom : ogr.Geometry
        The OGR geometry object to save.
    out_path : str, optional
        The desired output path for the vector file. If None, a temporary
        GeoPackage file is created in GDAL's virtual memory (`/vsimem/`).
        Default: None.
    name : str, optional
        The name for the layer within the vector file. Default: "converted_geom".
    prefix : str, optional
        A prefix to add to the temporary filename if `out_path` is None. Default: "".
    suffix : str, optional
        A suffix to add to the temporary filename if `out_path` is None. Default: "".
    add_uuid : bool, optional
        Whether to add a UUID to the temporary filename if `out_path` is None.
        Default: False.
    add_timestamp : bool, optional
        Whether to add a timestamp to the temporary filename if `out_path` is None.
        Default: False.
    projection_osr : osr.SpatialReference, optional
        The spatial reference system to assign to the output vector layer.
        If None, defaults to WGS84 (EPSG:4326). Default: None.

    Returns
    -------
    str
        The path to the created vector file (either `out_path` or the temporary path).

    Raises
    ------
    TypeError
        If `geom` is not an `ogr.Geometry`, `out_path` is not a string (if provided),
        or `projection_osr` is not an `osr.SpatialReference` (if provided).
    ValueError
        If `geom` is invalid, `out_path` is invalid (if provided), or if temporary
        path generation fails.
    RuntimeError
        If GDAL driver operations fail.

    Examples
    --------
    >>> point_geom = ogr.CreateGeometryFromWkt('POINT (0 0)')
    >>> temp_path = _get_vector_from_geom(point_geom)
    >>> isinstance(temp_path, str) and temp_path.startswith('/vsimem/')
    True
    >>> # Example saving to a specific file (requires write access)
    >>> # path = _get_vector_from_geom(point_geom, out_path='./my_point.gpkg')
    >>> _get_vector_from_geom(None)
    Raises TypeError: Input 'geom' must be an ogr.Geometry object.
    """
    # --- Input Validation ---
    if not isinstance(geom, ogr.Geometry):
        raise TypeError("Input 'geom' must be an ogr.Geometry object.")
    if not geom.IsValid():
        geom_fixed = geom.MakeValid()
        if geom_fixed is None or not geom_fixed.IsValid():
            raise ValueError("Input geometry is invalid and could not be fixed.")
        geom = geom_fixed
    if projection_osr is not None and not isinstance(projection_osr, osr.SpatialReference):
        raise TypeError("Input 'projection_osr' must be an osr.SpatialReference or None.")

    # --- Determine Path and Driver ---
    path: str
    driver_name: str
    try:
        if out_path is None:
            path = utils_path._get_temp_filepath(
                f"{name}.gpkg", prefix=prefix, suffix=suffix,
                add_uuid=add_uuid, add_timestamp=add_timestamp
            )
            driver_name = "GPKG"
        else:
            if not isinstance(out_path, str):
                raise TypeError(f"out_path must be a string, got {type(out_path)}")
            if not utils_path._check_is_valid_output_filepath(out_path):
                raise ValueError(f"Invalid output path provided: {out_path}")
            path = out_path
            driver_name = utils_gdal._get_vector_driver_name_from_path(path)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Error determining output path or driver: {e!s}") from e

    # --- Create Datasource and Write Geometry ---
    vector_ds: Optional[ogr.DataSource] = None
    driver: Optional[ogr.Driver] = None
    try:
        vector_ds, driver = _create_vector_datasource(path, driver_name)
        proj: osr.SpatialReference = projection_osr or utils_projection._get_default_projection_osr()
        _write_geom_to_layer(vector_ds, geom, name, proj)

    except (RuntimeError, ValueError) as e:
        # Attempt to delete partially created file on error
        if path is not None and driver is not None and utils_path._check_file_exists(path):
            try:
                driver.DeleteDataSource(path)
            except RuntimeError: # If deletion fails, ignore
                pass
        # Re-raise the original error
        raise RuntimeError(f"Failed to create vector from geometry: {e!s}") from e
    finally:
        # Ensure datasource is closed/dereferenced
        if vector_ds is not None:
            vector_ds = None

    # --- Final Check ---
    if not utils_path._check_file_exists(path):
        raise RuntimeError(f"Vector file was not successfully created at: {path}")

    return path
