"""Bounding Box class implementation with utility functions.

This module provides a BBox class for consistent handling of bounding boxes
across different coordinate formats, along with utility functions for common
bounding box operations.
"""

# Standard library
from typing import List, Union, Dict, Tuple, Sequence, Optional, Any, TypeVar, cast, overload
import copy

# External
import numpy as np
from beartype import beartype
from osgeo import ogr, osr

# Internal
from buteo.bbox.validation import (_check_is_valid_bbox, _check_is_valid_bbox_latlng,
                                  _check_bboxes_intersect, _check_bboxes_within)
from buteo.bbox.operations import _get_union_bboxes, _get_intersection_bboxes

# Type aliases
NumType = Union[int, float]
CoordPair = Tuple[NumType, NumType]
BboxType = Sequence[NumType]  # Any sequence of 4 numbers
OGRBboxType = List[float]     # Specifically [x_min, x_max, y_min, y_max]
GDALBboxType = List[float]    # Specifically [x_min, y_min, x_max, y_max]
GeoJSONBboxType = List[float] # Specifically [minx, miny, maxx, maxy]
PointsType = Sequence[Sequence[NumType]]  # Sequence of points [[x1,y1], [x2,y2], ...]
T = TypeVar('T', bound='BBox')  # Type variable for class methods returning self


class BBox:
    """A class representing a bounding box with consistent coordinate handling.
    
    The BBox class provides a uniform interface for working with bounding boxes
    in different coordinate formats, along with utility methods for common operations.
    
    Internally, coordinates are stored in OGR format [x_min, x_max, y_min, y_max],
    but the class provides properties and methods to convert to/from other formats.
    
    Attributes
    ----------
    x_min : float
        Minimum x-coordinate (left)
    x_max : float
        Maximum x-coordinate (right)
    y_min : float
        Minimum y-coordinate (bottom)
    y_max : float
        Maximum y-coordinate (top)
    
    Examples
    --------
    >>> # Create from OGR format [x_min, x_max, y_min, y_max]
    >>> bbox = BBox.from_ogr([0, 10, 5, 15])
    >>> bbox.as_ogr()
    [0.0, 10.0, 5.0, 15.0]
    
    >>> # Create from GDAL format [x_min, y_min, x_max, y_max]
    >>> bbox = BBox.from_gdal([0, 5, 10, 15])
    >>> bbox.as_gdal()
    [0.0, 5.0, 10.0, 15.0]
    
    >>> # Create from GeoJSON format [minx, miny, maxx, maxy]
    >>> bbox = BBox.from_geojson([0, 5, 10, 15])
    >>> bbox.as_geojson()
    [0.0, 5.0, 10.0, 15.0]
    
    >>> # Create from corner points
    >>> bbox = BBox.from_points([[0, 5], [3, 7], [10, 15]])
    >>> bbox.as_ogr()
    [0.0, 10.0, 5.0, 15.0]
    
    >>> # Common operations
    >>> bbox.center
    (5.0, 10.0)
    >>> bbox.width
    10.0
    >>> bbox.height
    10.0
    >>> bbox.area
    100.0
    >>> bbox.aspect_ratio
    1.0
    
    >>> # Buffer operations
    >>> buffered = bbox.buffer(2)
    >>> buffered.as_ogr()
    [-2.0, 12.0, 3.0, 17.0]
    """
    
    def __init__(self, x_min: float, x_max: float, y_min: float, y_max: float):
        """Initialize a BBox with explicit coordinates.
        
        Parameters
        ----------
        x_min : float
            Minimum x-coordinate (left)
        x_max : float
            Maximum x-coordinate (right)
        y_min : float
            Minimum y-coordinate (bottom)
        y_max : float
            Maximum y-coordinate (top)
            
        Raises
        ------
        ValueError
            If y_min > y_max or coordinates contain NaN values
        """
        # Convert to float and validate
        try:
            self.x_min = float(x_min)
            self.x_max = float(x_max)
            self.y_min = float(y_min)
            self.y_max = float(y_max)
        except (ValueError, TypeError) as e:
            raise ValueError(f"All bbox coordinates must be numeric: {e}") from e
            
        # Check for NaN values
        if any(np.isnan(v) for v in [self.x_min, self.x_max, self.y_min, self.y_max]):
            raise ValueError("Bounding box cannot contain NaN values")
            
        # Check y_min <= y_max (we allow x_min > x_max for dateline crossing)
        if self.y_min > self.y_max:
            raise ValueError(f"y_min ({y_min}) must be less than or equal to y_max ({y_max})")
    
    @classmethod
    @beartype
    def from_ogr(cls, bbox: BboxType) -> 'BBox':
        """Create a BBox from OGR format [x_min, x_max, y_min, y_max].
        
        Parameters
        ----------
        bbox : BboxType
            A sequence of 4 values in OGR format: [x_min, x_max, y_min, y_max]
            
        Returns
        -------
        BBox
            A new BBox instance
            
        Raises
        ------
        ValueError
            If the input is not a valid OGR formatted bbox
        """
        if not _check_is_valid_bbox(bbox):
            raise ValueError(f"Invalid OGR bbox format: {bbox}")
            
        return cls(bbox[0], bbox[1], bbox[2], bbox[3])

    @classmethod
    @beartype
    def from_gdal(cls, bbox: BboxType) -> 'BBox':
        """Create a BBox from GDAL format [x_min, y_min, x_max, y_max].
        
        Parameters
        ----------
        bbox : BboxType
            A sequence of 4 values in GDAL format: [x_min, y_min, x_max, y_max]
            
        Returns
        -------
        BBox
            A new BBox instance
            
        Raises
        ------
        ValueError
            If the input does not contain 4 numeric values or if the bbox is invalid
        """
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            raise ValueError("GDAL bbox must be a sequence of 4 numbers")
            
        try:
            # GDAL order: x_min, y_min, x_max, y_max -> OGR order: x_min, x_max, y_min, y_max
            return cls(bbox[0], bbox[2], bbox[1], bbox[3])
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid GDAL bbox format: {e}") from e

    @classmethod
    @beartype
    def from_geojson(cls, bbox: BboxType) -> 'BBox':
        """Create a BBox from GeoJSON format [minx, miny, maxx, maxy].
        
        Parameters
        ----------
        bbox : BboxType
            A sequence of 4 values in GeoJSON format: [minx, miny, maxx, maxy]
            (Note: This is identical to GDAL format)
            
        Returns
        -------
        BBox
            A new BBox instance
            
        Raises
        ------
        ValueError
            If the input does not contain 4 numeric values or if the bbox is invalid
        """
        # GeoJSON bbox format is the same as GDAL format
        return cls.from_gdal(bbox)
        
    @classmethod
    @beartype
    def from_points(cls, points: PointsType) -> 'BBox':
        """Create a BBox encompassing a collection of points.
        
        Parameters
        ----------
        points : PointsType
            A sequence of points, where each point is a sequence of at least 2 values [x, y, ...]
            
        Returns
        -------
        BBox
            A new BBox instance encompassing all input points
            
        Raises
        ------
        ValueError
            If points is empty or contains invalid entries
        """
        if not points:
            raise ValueError("Points sequence cannot be empty")
            
        try:
            # Extract x and y values
            x_values = [float(p[0]) for p in points]
            y_values = [float(p[1]) for p in points]
            
            # Get min/max
            x_min = min(x_values)
            x_max = max(x_values)
            y_min = min(y_values)
            y_max = max(y_values)
            
            return cls(x_min, x_max, y_min, y_max)
        except (ValueError, TypeError, IndexError) as e:
            raise ValueError(f"Invalid points format. Each point must contain at least x, y values: {e}") from e
            
    @classmethod
    @beartype
    def from_geom(cls, geom: ogr.Geometry) -> 'BBox':
        """Create a BBox from an OGR Geometry's envelope.
        
        Parameters
        ----------
        geom : ogr.Geometry
            An OGR geometry object
            
        Returns
        -------
        BBox
            A new BBox instance representing the geometry's envelope
            
        Raises
        ------
        TypeError
            If geom is not an ogr.Geometry object
        ValueError
            If the envelope cannot be computed
        """
        if not isinstance(geom, ogr.Geometry):
            raise TypeError(f"Input must be an ogr.Geometry object, got {type(geom)}")
            
        try:
            # GetEnvelope() returns (minX, maxX, minY, maxY)
            envelope = geom.GetEnvelope()
            return cls(envelope[0], envelope[1], envelope[2], envelope[3])
        except Exception as e:
            raise ValueError(f"Failed to compute geometry envelope: {e}") from e
    
    @beartype
    def as_ogr(self) -> OGRBboxType:
        """Return coordinates in OGR format.
        
        Returns
        -------
        OGRBboxType
            The bounding box in OGR format: [x_min, x_max, y_min, y_max]
        """
        return [self.x_min, self.x_max, self.y_min, self.y_max]
        
    @beartype
    def as_gdal(self) -> GDALBboxType:
        """Return coordinates in GDAL format.
        
        Returns
        -------
        GDALBboxType
            The bounding box in GDAL format: [x_min, y_min, x_max, y_max]
        """
        return [self.x_min, self.y_min, self.x_max, self.y_max]
        
    @beartype
    def as_geojson(self) -> GeoJSONBboxType:
        """Return coordinates in GeoJSON bbox format.
        
        Returns
        -------
        GeoJSONBboxType
            The bounding box in GeoJSON format: [minx, miny, maxx, maxy]
            (Note: This is identical to GDAL format)
        """
        return self.as_gdal()
        
    @beartype
    def as_corners(self) -> List[List[float]]:
        """Return the four corner points of the bounding box.
        
        Returns
        -------
        List[List[float]]
            A list of corner coordinates in [x, y] format, in the order:
            [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
        """
        return [
            [self.x_min, self.y_min],  # bottom-left
            [self.x_max, self.y_min],  # bottom-right
            [self.x_max, self.y_max],  # top-right
            [self.x_min, self.y_max],  # top-left
        ]
        
    @beartype
    def to_geom(self) -> ogr.Geometry:
        """Convert to an OGR Polygon Geometry.
        
        Returns
        -------
        ogr.Geometry
            An OGR Polygon geometry representing the bounding box
            
        Raises
        ------
        ValueError
            If geometry creation fails
        """
        try:
            ring = ogr.Geometry(ogr.wkbLinearRing)
            for point in self.as_corners():
                ring.AddPoint(point[0], point[1])
            # Close the ring
            ring.AddPoint(self.x_min, self.y_min)
            
            polygon = ogr.Geometry(ogr.wkbPolygon)
            if polygon.AddGeometry(ring) != ogr.OGRERR_NONE:
                raise ValueError("Failed to add ring to polygon geometry")
                
            return polygon
        except Exception as e:
            raise ValueError(f"Failed to create geometry: {e}") from e
            
    @beartype
    def to_wkt(self) -> str:
        """Convert to a WKT Polygon string.
        
        Returns
        -------
        str
            A WKT string representing the bounding box as a Polygon
            
        Raises
        ------
        ValueError
            If WKT conversion fails
        """
        try:
            geom = self.to_geom()
            wkt = geom.ExportToWkt()
            if not isinstance(wkt, str):
                raise ValueError("ExportToWkt did not return a string")
            return wkt
        except Exception as e:
            raise ValueError(f"Failed to convert to WKT: {e}") from e
            
    @beartype
    def to_geojson_dict(self) -> Dict[str, Any]:
        """Convert to a GeoJSON Polygon dictionary.
        
        Returns
        -------
        Dict[str, Any]
            A GeoJSON dictionary representing the bounding box as a Polygon:
            {"type": "Polygon", "coordinates": [[[x1, y1], [x2, y2], ...]]}
        """
        corners = self.as_corners()
        # Close the polygon by adding the first point at the end
        corners.append([self.x_min, self.y_min])
        
        return {
            "type": "Polygon",
            "coordinates": [corners]
        }
        
    @property
    def width(self) -> float:
        """Get the width of the bounding box.
        
        Returns
        -------
        float
            The width (x_max - x_min)
        """
        return abs(self.x_max - self.x_min)
        
    @property
    def height(self) -> float:
        """Get the height of the bounding box.
        
        Returns
        -------
        float
            The height (y_max - y_min)
        """
        return abs(self.y_max - self.y_min)
        
    @property
    def area(self) -> float:
        """Get the area of the bounding box.
        
        Returns
        -------
        float
            The area (width * height)
        """
        return self.width * self.height
        
    @property
    def aspect_ratio(self) -> float:
        """Get the aspect ratio (width/height) of the bounding box.
        
        Returns
        -------
        float
            The aspect ratio (width/height)
            
        Raises
        ------
        ValueError
            If height is 0, resulting in division by zero
        """
        if self.height == 0:
            raise ValueError("Cannot calculate aspect ratio: height is 0")
        return self.width / self.height
        
    @property
    def center(self) -> Tuple[float, float]:
        """Get the center point of the bounding box.
        
        Returns
        -------
        Tuple[float, float]
            The center coordinates (x, y)
        """
        return ((self.x_min + self.x_max) / 2, (self.y_min + self.y_max) / 2)
        
    @beartype
    def contains_point(self, point: Sequence[NumType]) -> bool:
        """Check if the bounding box contains a point.
        
        Parameters
        ----------
        point : Sequence[NumType]
            The point coordinates [x, y]
            
        Returns
        -------
        bool
            True if the bounding box contains the point, False otherwise
            
        Raises
        ------
        ValueError
            If point is not a valid coordinate pair
        """
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            raise ValueError("Point must be a sequence with at least 2 values [x, y]")
            
        try:
            x, y = float(point[0]), float(point[1])
            # Handle dateline crossing
            if self.x_min > self.x_max:  # Crosses dateline
                return (x >= self.x_min or x <= self.x_max) and y >= self.y_min and y <= self.y_max
            else:
                return x >= self.x_min and x <= self.x_max and y >= self.y_min and y <= self.y_max
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid point values: {e}") from e
            
    @beartype
    def buffer(self, distance: NumType) -> 'BBox':
        """Create a new BBox by buffering this one by a fixed distance.
        
        Parameters
        ----------
        distance : NumType
            The buffer distance to apply in all directions
            
        Returns
        -------
        BBox
            A new BBox instance with coordinates expanded by the buffer distance
        """
        try:
            dist = float(distance)
            return BBox(
                self.x_min - dist,
                self.x_max + dist,
                self.y_min - dist,
                self.y_max + dist
            )
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid buffer distance: {e}") from e
            
    @beartype
    def buffer_percent(self, percent: NumType) -> 'BBox':
        """Create a new BBox by buffering this one by a percentage of its size.
        
        Parameters
        ----------
        percent : NumType
            The buffer percentage (e.g., 10 for 10% buffer)
            
        Returns
        -------
        BBox
            A new BBox instance with coordinates expanded by the percentage
            
        Raises
        ------
        ValueError
            If percent is negative
        """
        try:
            pct = float(percent)
            if pct < 0:
                raise ValueError("Percent cannot be negative")
                
            # Calculate buffer distances based on dimensions
            x_buffer = self.width * (pct / 100)
            y_buffer = self.height * (pct / 100)
            
            return BBox(
                self.x_min - x_buffer,
                self.x_max + x_buffer,
                self.y_min - y_buffer,
                self.y_max + y_buffer
            )
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid buffer percentage: {e}") from e
            
    @beartype
    def intersects(self, other: Union['BBox', BboxType]) -> bool:
        """Check if this bounding box intersects with another.
        
        Parameters
        ----------
        other : Union['BBox', BboxType]
            Another BBox instance or a bbox in OGR format
            
        Returns
        -------
        bool
            True if the bounding boxes intersect, False otherwise
        """
        # Get other bbox in OGR format
        other_ogr = other.as_ogr() if isinstance(other, BBox) else other
        
        try:
            return _check_bboxes_intersect(self.as_ogr(), other_ogr)
        except ValueError as e:
            raise ValueError(f"Failed to check intersection: {e}") from e
            
    @beartype
    def intersection(self, other: Union['BBox', BboxType]) -> 'BBox':
        """Calculate the intersection with another bounding box.
        
        Parameters
        ----------
        other : Union['BBox', BboxType]
            Another BBox instance or a bbox in OGR format
            
        Returns
        -------
        BBox
            A new BBox instance representing the intersection
            
        Raises
        ------
        ValueError
            If the bounding boxes do not intersect
        """
        # Get other bbox in OGR format
        other_ogr = other.as_ogr() if isinstance(other, BBox) else other
        
        try:
            intersection_ogr = _get_intersection_bboxes(self.as_ogr(), other_ogr)
            return BBox.from_ogr(intersection_ogr)
        except ValueError as e:
            raise ValueError(f"Failed to calculate intersection: {e}") from e
            
    @beartype
    def union(self, other: Union['BBox', BboxType]) -> 'BBox':
        """Calculate the union with another bounding box.
        
        Parameters
        ----------
        other : Union['BBox', BboxType]
            Another BBox instance or a bbox in OGR format
            
        Returns
        -------
        BBox
            A new BBox instance representing the union
        """
        # Get other bbox in OGR format
        other_ogr = other.as_ogr() if isinstance(other, BBox) else other
        
        try:
            union_ogr = _get_union_bboxes(self.as_ogr(), other_ogr)
            return BBox.from_ogr(union_ogr)
        except ValueError as e:
            raise ValueError(f"Failed to calculate union: {e}") from e
            
    @beartype
    def contains(self, other: Union['BBox', BboxType]) -> bool:
        """Check if this bounding box completely contains another.
        
        Parameters
        ----------
        other : Union['BBox', BboxType]
            Another BBox instance or a bbox in OGR format
            
        Returns
        -------
        bool
            True if this bbox completely contains the other, False otherwise
        """
        # Get other bbox in OGR format
        other_ogr = other.as_ogr() if isinstance(other, BBox) else other
        
        try:
            return _check_bboxes_within(other_ogr, self.as_ogr())
        except ValueError as e:
            raise ValueError(f"Failed to check containment: {e}") from e
            
    def __repr__(self) -> str:
        """Return a string representation of the BBox.
        
        Returns
        -------
        str
            A string representation including the class name and coordinates
        """
        return f"BBox(x_min={self.x_min}, x_max={self.x_max}, y_min={self.y_min}, y_max={self.y_max})"
        
    def __eq__(self, other: object) -> bool:
        """Check if this BBox is equal to another.
        
        Parameters
        ----------
        other : object
            Another object to compare with
            
        Returns
        -------
        bool
            True if the bounding boxes have equal coordinates, False otherwise
        """
        if not isinstance(other, BBox):
            return False
            
        return (self.x_min == other.x_min and
                self.x_max == other.x_max and
                self.y_min == other.y_min and
                self.y_max == other.y_max)


# Utility functions

@beartype
def create_bbox_from_points(points: PointsType) -> OGRBboxType:
    """Create an OGR formatted bbox from a collection of points.
    
    Parameters
    ----------
    points : PointsType
        A sequence of points, where each point is a sequence of at least 2 values [x, y, ...]
        
    Returns
    -------
    OGRBboxType
        An OGR formatted bbox [x_min, x_max, y_min, y_max]
        
    Raises
    ------
    ValueError
        If points is empty or contains invalid entries
    """
    return BBox.from_points(points).as_ogr()

@beartype
def convert_bbox_ogr_to_gdal(bbox_ogr: BboxType) -> GDALBboxType:
    """Convert a bbox from OGR format to GDAL format.
    
    Parameters
    ----------
    bbox_ogr : BboxType
        A bbox in OGR format [x_min, x_max, y_min, y_max]
        
    Returns
    -------
    GDALBboxType
        A bbox in GDAL format [x_min, y_min, x_max, y_max]
        
    Raises
    ------
    ValueError
        If the input is not a valid OGR bbox
    """
    return BBox.from_ogr(bbox_ogr).as_gdal()

@beartype
def convert_bbox_gdal_to_ogr(bbox_gdal: BboxType) -> OGRBboxType:
    """Convert a bbox from GDAL format to OGR format.
    
    Parameters
    ----------
    bbox_gdal : BboxType
        A bbox in GDAL format [x_min, y_min, x_max, y_max]
        
    Returns
    -------
    OGRBboxType
        A bbox in OGR format [x_min, x_max, y_min, y_max]
        
    Raises
    ------
    ValueError
        If the input is not valid
    """
    return BBox.from_gdal(bbox_gdal).as_ogr()

@beartype
def get_bbox_center(bbox_ogr: BboxType) -> Tuple[float, float]:
    """Get the center point of an OGR formatted bbox.
    
    Parameters
    ----------
    bbox_ogr : BboxType
        A bbox in OGR format [x_min, x_max, y_min, y_max]
        
    Returns
    -------
    Tuple[float, float]
        The center coordinates (x, y)
        
    Raises
    ------
    ValueError
        If the input is not a valid OGR bbox
    """
    return BBox.from_ogr(bbox_ogr).center

@beartype
def buffer_bbox(bbox_ogr: BboxType, distance: NumType) -> OGRBboxType:
    """Buffer an OGR formatted bbox by a fixed distance.
    
    Parameters
    ----------
    bbox_ogr : BboxType
        A bbox in OGR format [x_min, x_max, y_min, y_max]
    distance : NumType
        The buffer distance to apply in all directions
        
    Returns
    -------
    OGRBboxType
        A new OGR formatted bbox with coordinates expanded by the buffer distance
        
    Raises
    ------
    ValueError
        If the input is not a valid OGR bbox or the distance is invalid
    """
    return BBox.from_ogr(bbox_ogr).buffer(distance).as_ogr()

@beartype
def get_bbox_aspect_ratio(bbox_ogr: BboxType) -> float:
    """Get the aspect ratio (width/height) of an OGR formatted bbox.
    
    Parameters
    ----------
    bbox_ogr : BboxType
        A bbox in OGR format [x_min, x_max, y_min, y_max]
        
    Returns
    -------
    float
        The aspect ratio (width/height)
        
    Raises
    ------
    ValueError
        If the input is not a valid OGR bbox or if height is 0
    """
    return BBox.from_ogr(bbox_ogr).aspect_ratio

@beartype
def bbox_contains_point(bbox_ogr: BboxType, point: Sequence[NumType]) -> bool:
    """Check if an OGR formatted bbox contains a point.
    
    Parameters
    ----------
    bbox_ogr : BboxType
        A bbox in OGR format [x_min, x_max, y_min, y_max]
    point : Sequence[NumType]
        The point coordinates [x, y]
        
    Returns
    -------
    bool
        True if the bbox contains the point, False otherwise
        
    Raises
    ------
    ValueError
        If the input is not a valid OGR bbox or the point is invalid
    """
    return BBox.from_ogr(bbox_ogr).contains_point(point)
