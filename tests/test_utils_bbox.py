# pylint: skip-file
# type: ignore

# Standard library
import sys; sys.path.append("../")

import pytest
import numpy as np
from osgeo import ogr, osr, gdal
from typing import List, Union

from buteo.utils import utils_bbox

# Fixtures
@pytest.fixture
def sample_bbox_ogr() -> List[float]:
    """Sample OGR bbox: [x_min, x_max, y_min, y_max]"""
    return [0.0, 1.0, 0.0, 1.0]

@pytest.fixture
def sample_bbox_latlng() -> List[float]:
    """Sample lat/long bbox"""
    return [-10.0, 10.0, -10.0, 10.0]

@pytest.fixture
def sample_geotransform() -> List[float]:
    """Sample GDAL geotransform"""
    return [0.0, 1.0, 0.0, 10.0, 0.0, -1.0]

@pytest.fixture
def wgs84_srs() -> osr.SpatialReference:
    """WGS84 spatial reference"""
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    return srs

class TestBboxValidation:
    def test_valid_bbox(self, sample_bbox_ogr):
        assert utils_bbox._check_is_valid_bbox(sample_bbox_ogr) is True

    def test_invalid_bbox_none(self):
        assert utils_bbox._check_is_valid_bbox(None) is False

    def test_invalid_bbox_wrong_length(self):
        assert utils_bbox._check_is_valid_bbox([0, 1, 0]) is False

    def test_invalid_bbox_wrong_order(self):
        assert utils_bbox._check_is_valid_bbox([1, 0, 0, 1]) is False

    def test_invalid_bbox_none_values(self):
        assert utils_bbox._check_is_valid_bbox([None, 1, 0, 1]) is False

    @pytest.mark.parametrize("bbox", [
        [-np.inf, np.inf, -90, 90],
        [-180, 180, -np.inf, np.inf]
    ])
    def test_valid_bbox_infinite(self, bbox):
        assert utils_bbox._check_is_valid_bbox(bbox) is True

class TestBboxLatLngValidation:
    def test_valid_latlng_bbox(self, sample_bbox_latlng):
        assert utils_bbox._check_is_valid_bbox_latlng(sample_bbox_latlng) is True

    def test_invalid_latlng_bbox_out_of_bounds(self):
        bbox = [-181, 180, -90, 90]
        assert utils_bbox._check_is_valid_bbox_latlng(bbox) is False

    @pytest.mark.parametrize("bbox", [
        [-180, 180, -91, 90],
        [-180, 180, -90, 91],
        [-180, 181, -90, 90]
    ])
    def test_invalid_latlng_bbox_values(self, bbox):
        assert utils_bbox._check_is_valid_bbox_latlng(bbox) is False

class TestGeotransformValidation:
    def test_valid_geotransform(self, sample_geotransform):
        assert utils_bbox._check_is_valid_geotransform(sample_geotransform) is True

    def test_invalid_geotransform_none(self):
        assert utils_bbox._check_is_valid_geotransform(None) is False

    def test_invalid_geotransform_length(self):
        assert utils_bbox._check_is_valid_geotransform([0, 1, 0, 0, 0]) is False

    def test_invalid_geotransform_zero_pixel(self):
        assert utils_bbox._check_is_valid_geotransform([0, 0, 0, 0, 0, 0]) is False

class TestPixelOffsets:
    def test_get_pixel_offsets(self, sample_bbox_ogr, sample_geotransform):
        offsets = utils_bbox._get_pixel_offsets(sample_geotransform, sample_bbox_ogr)
        assert len(offsets) == 4
        assert all(isinstance(x, int) for x in offsets)

    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            utils_bbox._get_pixel_offsets(None, [0, 1, 0, 1])
        with pytest.raises(ValueError):
            utils_bbox._get_pixel_offsets([0, 1, 0, 0, 0, -1], None)

    def test_zero_pixel_size(self):
        with pytest.raises(ValueError):
            utils_bbox._get_pixel_offsets([0, 0, 0, 0, 0, -1], [0, 1, 0, 1])

class TestBboxTransformations:
    def test_bbox_to_geom(self, sample_bbox_ogr):
        geom = utils_bbox._get_geom_from_bbox(sample_bbox_ogr)
        assert isinstance(geom, ogr.Geometry)
        assert geom.GetGeometryName() == "POLYGON"

    def test_geom_to_bbox(self):
        ring = ogr.Geometry(ogr.wkbLinearRing)
        for point in [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]:
            ring.AddPoint(*point)
        poly = ogr.Geometry(ogr.wkbPolygon)
        poly.AddGeometry(ring)
        bbox = utils_bbox._get_bbox_from_geom(poly)
        assert bbox == [0.0, 1.0, 0.0, 1.0]

    def test_bbox_to_wkt(self, sample_bbox_ogr):
        wkt = utils_bbox._get_wkt_from_bbox(sample_bbox_ogr)
        assert isinstance(wkt, str)
        assert wkt.startswith("POLYGON")

class TestUtmZone:
    @pytest.mark.parametrize("bbox,expected", [
        ([9.472, 9.474, 55.494, 55.496], "32N"),  # [minx, maxx, miny, maxy] for bbox around Kolding, Denmark
        ([172.636, 172.639, -43.531, -43.529], "59S")  # [minx, maxx, miny, maxy] for bbox in Christchurch, New Zealand
    ])
    def test_get_utm_zone(self, bbox, expected):
        zone = utils_bbox._get_utm_zone_from_bbox(bbox)
        assert zone == expected, f"Expected UTM zone {expected}, but got {zone}"

    def test_invalid_bbox(self):
        with pytest.raises(ValueError):
            utils_bbox._get_utm_zone_from_bbox([-181, 180, -90, 90])

class TestBboxOperations:
    def test_bbox_intersection(self):
        bbox1 = [0, 2, 0, 2]
        bbox2 = [1, 3, 1, 3]
        result = utils_bbox._get_intersection_bboxes(bbox1, bbox2)
        assert result == [1.0, 2.0, 1.0, 2.0]

    def test_bbox_union(self):
        bbox1 = [0, 1, 0, 1]
        bbox2 = [1, 2, 1, 2]
        result = utils_bbox._get_union_bboxes(bbox1, bbox2)
        assert result == [0.0, 2.0, 0.0, 2.0]

    def test_bbox_intersection_check(self):
        assert utils_bbox._check_bboxes_intersect([0, 2, 0, 2], [1, 3, 1, 3]) is True
        assert utils_bbox._check_bboxes_intersect([0, 1, 0, 1], [2, 3, 2, 3]) is False

    def test_bbox_within_check(self):
        assert utils_bbox._check_bboxes_within([1, 2, 1, 2], [0, 3, 0, 3]) is True
        assert utils_bbox._check_bboxes_within([0, 3, 0, 3], [1, 2, 1, 2]) is False


# Test null/invalid input handling
def test_invalid_bbox_none():
    assert utils_bbox._check_is_valid_bbox(None) is False

def test_invalid_bbox_wrong_length():
    assert utils_bbox._check_is_valid_bbox([1, 2, 3]) is False

def test_invalid_bbox_wrong_types():
    assert utils_bbox._check_is_valid_bbox(['1', '2', '3', '4']) is False

# Test bbox coordinate validation
def test_invalid_bbox_min_greater_than_max():
    assert utils_bbox._check_is_valid_bbox([2, 1, 2, 1]) is False
    assert utils_bbox._check_is_valid_bbox([1, 2, 2, 1]) is False

# Test geotransform validation
def test_invalid_geotransform_non_numeric():
    assert utils_bbox._check_is_valid_geotransform(['0', 1, 0, 0, 0, -1]) is False

def test_geotransform_rotation():
    # Test case where rotation terms are non-zero
    assert utils_bbox._check_is_valid_geotransform([0, 1, 0.1, 0, 0, -1]) is True

# Test bbox transformations
def test_bbox_to_geom_invalid():
    with pytest.raises(TypeError):
        utils_bbox._get_geom_from_bbox(None)
        
def test_bbox_to_geom_empty():
    with pytest.raises(ValueError):
        utils_bbox._get_geom_from_bbox([])

# Test pixel offset calculations
def test_pixel_offsets_fractional():
    geotransform = [0, 0.5, 0, 0, 0, -0.5]  # 0.5 degree pixels
    bbox = [0.25, 1.75, 0.25, 1.75]  # Should round to pixel boundaries
    offsets = utils_bbox._get_pixel_offsets(geotransform, bbox)
    assert len(offsets) == 4
    assert all(isinstance(x, int) for x in offsets)
