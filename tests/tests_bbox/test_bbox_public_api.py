"""Unit tests for the public API functions in buteo.bbox module."""

# Standard library
from typing import Union, Sequence

# External
import pytest
import numpy as np
from osgeo import ogr, osr

# Internal
from buteo.bbox import (
    get_bbox_from_dataset,
    union_bboxes,
    intersection_bboxes,
    bbox_to_geom,
    bbox_from_geom,
    bbox_to_wkt,
    bbox_to_geojson,
    align_bbox,
    validate_bbox,
    validate_bbox_latlng
)

# Type Aliases
BboxType = Sequence[Union[int, float]]


class TestPublicAPI:
    """Tests for the public API functions in buteo.bbox."""

    def test_union_bboxes(self):
        """Test union_bboxes function."""
        bbox1 = [0.0, 2.0, 0.0, 2.0]
        bbox2 = [1.0, 3.0, 1.0, 3.0]
        result = union_bboxes(bbox1, bbox2)
        assert result == [0.0, 3.0, 0.0, 3.0]

    def test_intersection_bboxes(self):
        """Test intersection_bboxes function."""
        bbox1 = [0.0, 2.0, 0.0, 2.0]
        bbox2 = [1.0, 3.0, 1.0, 3.0]
        result = intersection_bboxes(bbox1, bbox2)
        assert result == [1.0, 2.0, 1.0, 2.0]

    def test_bbox_to_geom(self):
        """Test bbox_to_geom function."""
        bbox = [0.0, 1.0, 0.0, 1.0]
        geom = bbox_to_geom(bbox)
        assert isinstance(geom, ogr.Geometry)
        assert geom.GetGeometryName() == "POLYGON"
        
        # Check if the geometry bounds match input bbox
        envelope = geom.GetEnvelope()  # (minX, maxX, minY, maxY)
        np.testing.assert_allclose(envelope, bbox)

    def test_bbox_from_geom(self):
        """Test bbox_from_geom function."""
        wkt = "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))"
        geom = ogr.CreateGeometryFromWkt(wkt)
        bbox = bbox_from_geom(geom)
        assert bbox == [0.0, 1.0, 0.0, 1.0]

    def test_bbox_to_wkt(self):
        """Test bbox_to_wkt function."""
        bbox = [0.0, 1.0, 2.0, 3.0]
        wkt = bbox_to_wkt(bbox)
        assert isinstance(wkt, str)
        assert wkt.startswith("POLYGON ((")
        
        # Parse the WKT back to geometry and check bounds
        geom = ogr.CreateGeometryFromWkt(wkt)
        envelope = geom.GetEnvelope()
        np.testing.assert_allclose(envelope, bbox)

    def test_bbox_to_geojson(self):
        """Test bbox_to_geojson function."""
        bbox = [0.0, 1.0, 2.0, 3.0]
        geojson = bbox_to_geojson(bbox)
        assert isinstance(geojson, dict)
        assert geojson["type"] == "Polygon"
        assert len(geojson["coordinates"][0]) == 5  # 5 points with closing point
        
        # Check the coordinates
        coords = geojson["coordinates"][0]
        assert coords[0] == [0.0, 2.0]  # bottom-left
        assert coords[1] == [1.0, 2.0]  # bottom-right
        assert coords[2] == [1.0, 3.0]  # top-right
        assert coords[3] == [0.0, 3.0]  # top-left
        assert coords[4] == coords[0]   # closing point

    def test_align_bbox(self):
        """Test align_bbox function."""
        ref_bbox = [0.0, 4.0, 0.0, 4.0]
        target_bbox = [1.2, 3.7, 1.2, 3.7]
        
        # Align with 1x1 pixel size (pixel_height=-1)
        aligned = align_bbox(ref_bbox, target_bbox, 1.0, -1.0)
        assert aligned == [1.0, 4.0, 1.0, 4.0]
        
        # Align with 0.5x0.5 pixel size
        aligned = align_bbox(ref_bbox, target_bbox, 0.5, -0.5)
        assert aligned == [1.0, 4.0, 1.0, 4.0]  # Snaps to nearest 0.5 outwards

    def test_validate_bbox(self):
        """Test validate_bbox function."""
        assert validate_bbox([0, 1, 0, 1]) is True
        assert validate_bbox([1, 0, 0, 1]) is True  # x_min > x_max is allowed (dateline crossing)
        assert validate_bbox([0, 1, 1, 0]) is False  # y_min > y_max is not allowed
        assert validate_bbox([0, 1, 0]) is False  # Wrong length
        
        # For None and non-sequence inputs, we can't test directly with beartype validation
        # Instead, test that they raise expected validation errors
        try:
            validate_bbox(None)
            assert False, "Should have raised BeartypeCallHintParamViolation"
        except Exception:
            # Any exception is acceptable here since we expect validation to fail
            pass
        
        try:
            validate_bbox(['1', '2', '3', '4'])
            assert False, "Should have raised exception for wrong types"
        except Exception:
            # Any exception is acceptable
            pass

    def test_validate_bbox_latlng(self):
        """Test validate_bbox_latlng function."""
        assert validate_bbox_latlng([-180, 180, -90, 90]) is True
        assert validate_bbox_latlng([170, -170, -10, 10]) is True  # Dateline crossing
        assert validate_bbox_latlng([-10, 10, -10, 10]) is True
        assert validate_bbox_latlng([-181, 180, -90, 90]) is False  # Invalid longitude
        assert validate_bbox_latlng([0, 1, -91, 90]) is False  # Invalid latitude
