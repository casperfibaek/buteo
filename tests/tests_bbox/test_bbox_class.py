"""Unit tests for the BBox class and utility functions in buteo.bbox module."""

# Standard library
import math
from typing import Any

# External
import pytest
import numpy as np
from osgeo import ogr
from beartype.roar import BeartypeCallHintParamViolation # Import beartype exception

# Internal
from buteo.bbox import (
    BBox,
    create_bbox_from_points,
    convert_bbox_ogr_to_gdal,
    convert_bbox_gdal_to_ogr,
    get_bbox_center,
    buffer_bbox,
    get_bbox_aspect_ratio,
    bbox_contains_point
)


class TestBBoxClass:
    """Tests for the BBox class."""

    def test_init(self):
        """Test BBox initialization with explicit coordinates."""
        bbox = BBox(0.0, 10.0, 5.0, 15.0)
        assert bbox.x_min == 0.0
        assert bbox.x_max == 10.0
        assert bbox.y_min == 5.0
        assert bbox.y_max == 15.0

    def test_init_validation(self):
        """Test BBox initialization validation."""
        # y_min > y_max should raise ValueError
        with pytest.raises(ValueError):
            BBox(0.0, 10.0, 15.0, 5.0)

        # NaN values should raise ValueError
        with pytest.raises(ValueError):
            BBox(0.0, 10.0, np.nan, 15.0)

        # Non-numeric values should raise ValueError
        with pytest.raises(ValueError):
            BBox("invalid", 10.0, 5.0, 15.0)

    def test_from_ogr(self):
        """Test BBox.from_ogr factory method."""
        bbox = BBox.from_ogr([0.0, 10.0, 5.0, 15.0])
        assert bbox.x_min == 0.0
        assert bbox.x_max == 10.0
        assert bbox.y_min == 5.0
        assert bbox.y_max == 15.0

    def test_from_gdal(self):
        """Test BBox.from_gdal factory method."""
        bbox = BBox.from_gdal([0.0, 5.0, 10.0, 15.0])
        assert bbox.x_min == 0.0
        assert bbox.x_max == 10.0
        assert bbox.y_min == 5.0
        assert bbox.y_max == 15.0

    def test_from_geojson(self):
        """Test BBox.from_geojson factory method."""
        bbox = BBox.from_geojson([0.0, 5.0, 10.0, 15.0])
        assert bbox.x_min == 0.0
        assert bbox.x_max == 10.0
        assert bbox.y_min == 5.0
        assert bbox.y_max == 15.0

    def test_from_points(self):
        """Test BBox.from_points factory method."""
        points = [[0.0, 5.0], [3.0, 7.0], [10.0, 15.0]]
        bbox = BBox.from_points(points)
        assert bbox.x_min == 0.0
        assert bbox.x_max == 10.0
        assert bbox.y_min == 5.0
        assert bbox.y_max == 15.0

        # Empty points list should raise ValueError
        with pytest.raises(ValueError):
            BBox.from_points([])

        # Invalid points should raise ValueError
        with pytest.raises(ValueError):
            BBox.from_points([[0.0], [3.0, 7.0]])

    def test_from_geom(self):
        """Test BBox.from_geom factory method."""
        wkt = "POLYGON ((0 5, 10 5, 10 15, 0 15, 0 5))"
        geom = ogr.CreateGeometryFromWkt(wkt)
        bbox = BBox.from_geom(geom)
        
        # GetEnvelope returns (minX, maxX, minY, maxY)
        assert bbox.x_min == 0.0
        assert bbox.x_max == 10.0
        assert bbox.y_min == 5.0
        assert bbox.y_max == 15.0

        # Invalid geometry should raise BeartypeCallHintParamViolation
        with pytest.raises(BeartypeCallHintParamViolation):
            BBox.from_geom("not a geometry") # type: ignore

    def test_as_ogr(self):
        """Test as_ogr method."""
        bbox = BBox(0.0, 10.0, 5.0, 15.0)
        ogr_bbox = bbox.as_ogr()
        assert ogr_bbox == [0.0, 10.0, 5.0, 15.0]

    def test_as_gdal(self):
        """Test as_gdal method."""
        bbox = BBox(0.0, 10.0, 5.0, 15.0)
        gdal_bbox = bbox.as_gdal()
        assert gdal_bbox == [0.0, 5.0, 10.0, 15.0]

    def test_as_geojson(self):
        """Test as_geojson method."""
        bbox = BBox(0.0, 10.0, 5.0, 15.0)
        geojson_bbox = bbox.as_geojson()
        assert geojson_bbox == [0.0, 5.0, 10.0, 15.0]

    def test_as_corners(self):
        """Test as_corners method."""
        bbox = BBox(0.0, 10.0, 5.0, 15.0)
        corners = bbox.as_corners()
        assert corners == [
            [0.0, 5.0],   # bottom-left
            [10.0, 5.0],  # bottom-right
            [10.0, 15.0], # top-right
            [0.0, 15.0]   # top-left
        ]

    def test_to_geom(self):
        """Test to_geom method."""
        bbox = BBox(0.0, 10.0, 5.0, 15.0)
        geom = bbox.to_geom()
        
        assert isinstance(geom, ogr.Geometry)
        assert geom.GetGeometryName() == "POLYGON"
        
        # Check if the geometry bounds match bbox
        envelope = geom.GetEnvelope()  # (minX, maxX, minY, maxY)
        assert envelope == (0.0, 10.0, 5.0, 15.0)

    def test_to_wkt(self):
        """Test to_wkt method."""
        bbox = BBox(0.0, 10.0, 5.0, 15.0)
        wkt = bbox.to_wkt()
        
        assert isinstance(wkt, str)
        assert wkt.startswith("POLYGON ((")
        
        # Parse the WKT back to geometry and check bounds
        geom = ogr.CreateGeometryFromWkt(wkt)
        envelope = geom.GetEnvelope()
        assert envelope == (0.0, 10.0, 5.0, 15.0)

    def test_to_geojson_dict(self):
        """Test to_geojson_dict method."""
        bbox = BBox(0.0, 10.0, 5.0, 15.0)
        geojson = bbox.to_geojson_dict()
        
        assert isinstance(geojson, dict)
        assert geojson["type"] == "Polygon"
        assert len(geojson["coordinates"][0]) == 5  # 5 points with closing point
        
        # Check the coordinates
        coords = geojson["coordinates"][0]
        assert coords[0] == [0.0, 5.0]   # bottom-left
        assert coords[1] == [10.0, 5.0]  # bottom-right
        assert coords[2] == [10.0, 15.0] # top-right
        assert coords[3] == [0.0, 15.0]  # top-left
        assert coords[4] == coords[0]    # closing point

    def test_width_height_area(self):
        """Test width, height, and area properties."""
        bbox = BBox(0.0, 10.0, 5.0, 15.0)
        
        assert bbox.width == 10.0
        assert bbox.height == 10.0
        assert bbox.area == 100.0

    def test_aspect_ratio(self):
        """Test aspect_ratio property."""
        # Square bbox
        bbox1 = BBox(0.0, 10.0, 0.0, 10.0)
        assert bbox1.aspect_ratio == 1.0
        
        # Rectangle bbox
        bbox2 = BBox(0.0, 20.0, 0.0, 10.0)
        assert bbox2.aspect_ratio == 2.0
        
        # Zero height should raise ValueError
        bbox3 = BBox(0.0, 10.0, 5.0, 5.0)
        with pytest.raises(ValueError):
            _ = bbox3.aspect_ratio

    def test_center(self):
        """Test center property."""
        bbox = BBox(0.0, 10.0, 5.0, 15.0)
        assert bbox.center == (5.0, 10.0)

    def test_contains_point(self):
        """Test contains_point method."""
        bbox = BBox(0.0, 10.0, 5.0, 15.0)
        
        # Points inside
        assert bbox.contains_point([5.0, 10.0])  # center
        assert bbox.contains_point([0.0, 5.0])   # bottom-left corner
        assert bbox.contains_point([10.0, 15.0]) # top-right corner
        
        # Points outside
        assert not bbox.contains_point([-1.0, 10.0])  # left
        assert not bbox.contains_point([5.0, 4.0])    # below
        assert not bbox.contains_point([11.0, 10.0])  # right
        assert not bbox.contains_point([5.0, 16.0])   # above
        
        # Test dateline crossing
        dateline_bbox = BBox(170.0, -170.0, -10.0, 10.0)
        assert dateline_bbox.contains_point([175.0, 0.0])  # east side
        assert dateline_bbox.contains_point([-175.0, 0.0]) # west side
        assert not dateline_bbox.contains_point([0.0, 0.0]) # outside
        
        # Invalid point should raise ValueError
        with pytest.raises(ValueError):
            bbox.contains_point([])
        # Invalid type should raise BeartypeCallHintParamViolation
        with pytest.raises(BeartypeCallHintParamViolation):
            bbox.contains_point("invalid") # type: ignore

    def test_buffer(self):
        """Test buffer method."""
        bbox = BBox(0.0, 10.0, 5.0, 15.0)
        
        # Buffer by 2.0
        buffered = bbox.buffer(2.0)
        assert buffered.x_min == -2.0
        assert buffered.x_max == 12.0
        assert buffered.y_min == 3.0
        assert buffered.y_max == 17.0
        
        # Original bbox should be unchanged
        assert bbox.x_min == 0.0
        assert bbox.x_max == 10.0
        assert bbox.y_min == 5.0
        assert bbox.y_max == 15.0
        
        # Invalid buffer distance type should raise BeartypeCallHintParamViolation
        with pytest.raises(BeartypeCallHintParamViolation):
            bbox.buffer("invalid") # type: ignore

    def test_buffer_percent(self):
        """Test buffer_percent method."""
        bbox = BBox(0.0, 10.0, 0.0, 10.0)
        
        # Buffer by 10%
        buffered = bbox.buffer_percent(10)
        assert buffered.x_min == -1.0
        assert buffered.x_max == 11.0
        assert buffered.y_min == -1.0
        assert buffered.y_max == 11.0
        
        # Original bbox should be unchanged
        assert bbox.x_min == 0.0
        assert bbox.x_max == 10.0
        assert bbox.y_min == 0.0
        assert bbox.y_max == 10.0
        
        # Test rectangle
        bbox2 = BBox(0.0, 20.0, 0.0, 10.0)
        buffered2 = bbox2.buffer_percent(10)
        assert buffered2.x_min == -2.0
        assert buffered2.x_max == 22.0
        assert buffered2.y_min == -1.0
        assert buffered2.y_max == 11.0
        
        # Negative percent should raise ValueError
        with pytest.raises(ValueError):
            bbox.buffer_percent(-10)
        
        # Invalid percent type should raise BeartypeCallHintParamViolation
        with pytest.raises(BeartypeCallHintParamViolation):
            bbox.buffer_percent("invalid") # type: ignore

    def test_intersects(self):
        """Test intersects method."""
        bbox1 = BBox(0.0, 10.0, 0.0, 10.0)
        
        # Overlapping bbox
        bbox2 = BBox(5.0, 15.0, 5.0, 15.0)
        assert bbox1.intersects(bbox2)
        assert bbox2.intersects(bbox1)
        
        # Touching corner
        bbox3 = BBox(10.0, 20.0, 10.0, 20.0)
        assert bbox1.intersects(bbox3)
        assert bbox3.intersects(bbox1)
        
        # Non-intersecting
        bbox4 = BBox(20.0, 30.0, 20.0, 30.0)
        assert not bbox1.intersects(bbox4)
        assert not bbox4.intersects(bbox1)
        
        # Using OGR bbox format
        assert bbox1.intersects([5.0, 15.0, 5.0, 15.0])
        assert not bbox1.intersects([20.0, 30.0, 20.0, 30.0])
        
        # Invalid bbox type should raise BeartypeCallHintParamViolation
        with pytest.raises(BeartypeCallHintParamViolation):
            bbox1.intersects("invalid") # type: ignore

    def test_intersection(self):
        """Test intersection method."""
        bbox1 = BBox(0.0, 10.0, 0.0, 10.0)
        bbox2 = BBox(5.0, 15.0, 5.0, 15.0)
        
        # Calculate intersection
        intersection = bbox1.intersection(bbox2)
        assert intersection.x_min == 5.0
        assert intersection.x_max == 10.0
        assert intersection.y_min == 5.0
        assert intersection.y_max == 10.0
        
        # Using OGR bbox format
        intersection2 = bbox1.intersection([5.0, 15.0, 5.0, 15.0])
        assert intersection2.x_min == 5.0
        assert intersection2.x_max == 10.0
        assert intersection2.y_min == 5.0
        assert intersection2.y_max == 10.0
        
        # Non-intersecting should raise ValueError
        bbox3 = BBox(20.0, 30.0, 20.0, 30.0)
        with pytest.raises(ValueError):
            bbox1.intersection(bbox3)
        
        # Invalid bbox type should raise BeartypeCallHintParamViolation
        with pytest.raises(BeartypeCallHintParamViolation):
            bbox1.intersection("invalid") # type: ignore

    def test_union(self):
        """Test union method."""
        bbox1 = BBox(0.0, 10.0, 0.0, 10.0)
        bbox2 = BBox(5.0, 15.0, 5.0, 15.0)
        
        # Calculate union
        union = bbox1.union(bbox2)
        assert union.x_min == 0.0
        assert union.x_max == 15.0
        assert union.y_min == 0.0
        assert union.y_max == 15.0
        
        # Using OGR bbox format
        union2 = bbox1.union([5.0, 15.0, 5.0, 15.0])
        assert union2.x_min == 0.0
        assert union2.x_max == 15.0
        assert union2.y_min == 0.0
        assert union2.y_max == 15.0
        
        # Invalid bbox type should raise BeartypeCallHintParamViolation
        with pytest.raises(BeartypeCallHintParamViolation):
            bbox1.union("invalid") # type: ignore

    def test_contains(self):
        """Test contains method."""
        bbox1 = BBox(0.0, 10.0, 0.0, 10.0)
        
        # Completely contained bbox
        bbox2 = BBox(2.0, 8.0, 2.0, 8.0)
        assert bbox1.contains(bbox2)
        assert not bbox2.contains(bbox1)
        
        # Overlapping but not contained
        bbox3 = BBox(5.0, 15.0, 5.0, 15.0)
        assert not bbox1.contains(bbox3)
        assert not bbox3.contains(bbox1)
        
        # Using OGR bbox format
        assert bbox1.contains([2.0, 8.0, 2.0, 8.0])
        assert not bbox1.contains([5.0, 15.0, 5.0, 15.0])
        
        # Invalid bbox type should raise BeartypeCallHintParamViolation
        with pytest.raises(BeartypeCallHintParamViolation):
            bbox1.contains("invalid") # type: ignore

    def test_repr(self):
        """Test __repr__ method."""
        bbox = BBox(0.0, 10.0, 5.0, 15.0)
        repr_str = repr(bbox)
        
        assert repr_str == "BBox(x_min=0.0, x_max=10.0, y_min=5.0, y_max=15.0)"

    def test_eq(self):
        """Test __eq__ method."""
        bbox1 = BBox(0.0, 10.0, 5.0, 15.0)
        bbox2 = BBox(0.0, 10.0, 5.0, 15.0)
        bbox3 = BBox(1.0, 11.0, 5.0, 15.0)
        
        assert bbox1 == bbox2
        assert bbox1 != bbox3
        assert bbox1 != "not a bbox"


class TestUtilityFunctions:
    """Tests for the utility functions."""

    def test_create_bbox_from_points(self):
        """Test create_bbox_from_points function."""
        points = [[0.0, 5.0], [3.0, 7.0], [10.0, 15.0]]
        bbox = create_bbox_from_points(points)
        
        assert bbox == [0.0, 10.0, 5.0, 15.0]
        
        # Empty points list should raise ValueError
        with pytest.raises(ValueError):
            create_bbox_from_points([])

    def test_convert_bbox_ogr_to_gdal(self):
        """Test convert_bbox_ogr_to_gdal function."""
        ogr_bbox = [0.0, 10.0, 5.0, 15.0]
        gdal_bbox = convert_bbox_ogr_to_gdal(ogr_bbox)
        
        assert gdal_bbox == [0.0, 5.0, 10.0, 15.0]
        
        # Invalid bbox should raise ValueError
        with pytest.raises(ValueError):
            convert_bbox_ogr_to_gdal([0.0, 10.0, 15.0]) # wrong length

    def test_convert_bbox_gdal_to_ogr(self):
        """Test convert_bbox_gdal_to_ogr function."""
        gdal_bbox = [0.0, 5.0, 10.0, 15.0]
        ogr_bbox = convert_bbox_gdal_to_ogr(gdal_bbox)
        
        assert ogr_bbox == [0.0, 10.0, 5.0, 15.0]
        
        # Invalid bbox should raise ValueError
        with pytest.raises(ValueError):
            convert_bbox_gdal_to_ogr([0.0, 5.0, 10.0]) # wrong length

    def test_get_bbox_center(self):
        """Test get_bbox_center function."""
        ogr_bbox = [0.0, 10.0, 5.0, 15.0]
        center = get_bbox_center(ogr_bbox)
        
        assert center == (5.0, 10.0)
        
        # Invalid bbox should raise ValueError
        with pytest.raises(ValueError):
            get_bbox_center([0.0, 10.0, 5.0]) # wrong length

    def test_buffer_bbox(self):
        """Test buffer_bbox function."""
        ogr_bbox = [0.0, 10.0, 5.0, 15.0]
        buffered = buffer_bbox(ogr_bbox, 2.0)
        
        assert buffered == [-2.0, 12.0, 3.0, 17.0]
        
        # Invalid bbox should raise ValueError
        with pytest.raises(ValueError):
            buffer_bbox([0.0, 10.0, 5.0], 2.0) # wrong length
        
        # Invalid distance type should raise BeartypeCallHintParamViolation
        with pytest.raises(BeartypeCallHintParamViolation):
            buffer_bbox(ogr_bbox, "invalid") # type: ignore

    def test_get_bbox_aspect_ratio(self):
        """Test get_bbox_aspect_ratio function."""
        # Square bbox
        ogr_bbox1 = [0.0, 10.0, 0.0, 10.0]
        assert get_bbox_aspect_ratio(ogr_bbox1) == 1.0
        
        # Rectangle bbox
        ogr_bbox2 = [0.0, 20.0, 0.0, 10.0]
        assert get_bbox_aspect_ratio(ogr_bbox2) == 2.0
        
        # Zero height should raise ValueError
        ogr_bbox3 = [0.0, 10.0, 5.0, 5.0]
        with pytest.raises(ValueError):
            get_bbox_aspect_ratio(ogr_bbox3)
        
        # Invalid bbox should raise ValueError
        with pytest.raises(ValueError):
            get_bbox_aspect_ratio([0.0, 10.0, 5.0]) # wrong length

    def test_bbox_contains_point(self):
        """Test bbox_contains_point function."""
        ogr_bbox = [0.0, 10.0, 5.0, 15.0]
        
        # Points inside
        assert bbox_contains_point(ogr_bbox, [5.0, 10.0])  # center
        assert bbox_contains_point(ogr_bbox, [0.0, 5.0])   # bottom-left corner
        assert bbox_contains_point(ogr_bbox, [10.0, 15.0]) # top-right corner
        
        # Points outside
        assert not bbox_contains_point(ogr_bbox, [-1.0, 10.0])  # left
        assert not bbox_contains_point(ogr_bbox, [5.0, 4.0])    # below
        assert not bbox_contains_point(ogr_bbox, [11.0, 10.0])  # right
        assert not bbox_contains_point(ogr_bbox, [5.0, 16.0])   # above
        
        # Test dateline crossing
        dateline_bbox = [170.0, -170.0, -10.0, 10.0]
        assert bbox_contains_point(dateline_bbox, [175.0, 0.0])  # east side
        assert bbox_contains_point(dateline_bbox, [-175.0, 0.0]) # west side
        assert not bbox_contains_point(dateline_bbox, [0.0, 0.0]) # outside
        
        # Invalid bbox should raise ValueError
        with pytest.raises(ValueError):
            bbox_contains_point([0.0, 10.0, 5.0], [5.0, 10.0]) # wrong length
        
        # Invalid point should raise ValueError
        with pytest.raises(ValueError):
            bbox_contains_point(ogr_bbox, [])
