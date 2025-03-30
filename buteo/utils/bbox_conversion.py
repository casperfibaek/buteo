"""### Bounding box conversion functions. ###

Functions to convert bounding boxes to and from various formats.

There are two different formats for bounding boxes used by GDAL:</br>
OGR:  `[x_min, x_max, y_min, y_max]`</br>
WARP: `[x_min, y_min, x_max, y_max]`</br>

_If nothing else is stated, the OGR format is used._
"""

# Standard library
from typing import List, Union, Dict, Optional
from uuid import uuid4
from warnings import warn

# External
import numpy as np
from osgeo import ogr, osr, gdal

# Internal
from buteo.utils.bbox_validation import _check_is_valid_bbox
from buteo.utils import utils_projection, utils_path, utils_gdal


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

    if any(np.isnan(val) for val in [x_min, x_max, y_min, y_max]):
        raise ValueError("Bounding box contains NaN values")

    # Create geometry
    try:
        # if all are zero
        if x_min == 0 and x_max == 0 and y_min == 0 and y_max == 0:
            wkt = """POLYGON Z ((0 0 0,
                        0.000001 0.0 0,
                        0.000001 0.000001 0,
                        0.0 0.000001 0,
                        0 0 0))"""
            geom = ogr.CreateGeometryFromWkt(wkt)

            warn("Bounding box is zero ([0, 0, 0, 0]), creating a very small polygon instead.")

        else:
            ring = ogr.Geometry(ogr.wkbLinearRing)
            ring.AddPoint(x_min, y_min)
            ring.AddPoint(x_max, y_min)
            ring.AddPoint(x_max, y_max)
            ring.AddPoint(x_min, y_max)
            ring.CloseRings()  # Ensure the ring is closed properly

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

        vector_ds.FlushCache()
        del vector_ds

    except Exception as e:
        if 'vector_ds' in locals():
            del vector_ds
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
