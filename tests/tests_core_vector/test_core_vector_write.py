# pylint: skip-file
# type: ignore

# Standard library
import sys; sys.path.append("../../")

import pytest
from osgeo import ogr, osr

from buteo.core_vector.core_vector_write import (
    vector_create_copy,
    vector_create_from_bbox,
    vector_create_from_wkt,
    vector_create_from_points,
    vector_create_from_geojson,
    vector_set_crs,
)

@pytest.fixture
def simple_vector(tmp_path):
    """Create a simple vector file with one point feature."""
    vector_path = tmp_path / "simple.gpkg"
    driver = ogr.GetDriverByName('GPKG')
    ds = driver.CreateDataSource(str(vector_path))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    layer = ds.CreateLayer('test', srs, ogr.wkbPoint)

    # Add a field
    field_defn = ogr.FieldDefn('name', ogr.OFTString)
    layer.CreateField(field_defn)

    # Add a feature
    feature = ogr.Feature(layer.GetLayerDefn())
    point = ogr.Geometry(ogr.wkbPoint)
    point.AddPoint(0, 0)
    feature.SetGeometry(point)
    feature.SetField('name', 'test_point')
    layer.CreateFeature(feature)
    
    ds = None
    return str(vector_path)

class TestVectorCreateCopy:
    def test_basic_copy(self, simple_vector, tmp_path):
        """Test basic vector copy functionality."""
        out_path = tmp_path / "copied.gpkg"
        result = vector_create_copy(simple_vector, str(out_path))
        assert isinstance(result, str)
        assert ogr.Open(result) is not None

    def test_copy_with_layer_selection(self, simple_vector, tmp_path):
        """Test copying specific layers."""
        out_path = tmp_path / "copied_layer.gpkg"
        result = vector_create_copy(simple_vector, str(out_path), layer_names_or_ids="test")
        assert isinstance(result, str)
        ds = ogr.Open(result)
        assert ds.GetLayerByName("test") is not None

    def test_copy_with_prefix_suffix(self, simple_vector, tmp_path):
        """Test copying with prefix and suffix."""
        out_path = tmp_path / "base.gpkg"
        result = vector_create_copy(simple_vector, str(out_path), prefix="pre_", suffix="_post")
        assert "pre_" in result
        assert "_post" in result

    def test_copy_multiple_vectors(self, simple_vector, tmp_path):
        """Test copying multiple vectors."""
        out_paths = [tmp_path / "copy1.gpkg", tmp_path / "copy2.gpkg"]
        result = vector_create_copy([simple_vector, simple_vector], [str(p) for p in out_paths])
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(ogr.Open(p) is not None for p in result)

class TestVectorCreateFromBbox:
    def test_basic_bbox(self, tmp_path):
        """Test creating vector from basic bbox."""
        bbox = [0, 1, 0, 1]
        result = vector_create_from_bbox(bbox, out_path=str(tmp_path / "bbox.gpkg"))
        ds = ogr.Open(result)
        assert ds is not None
        layer = ds.GetLayer(0)
        assert layer.GetGeomType() == ogr.wkbPolygon

    def test_bbox_with_projection(self, tmp_path):
        """Test creating bbox with specific projection."""
        bbox = [0, 1, 0, 1]
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(3857)
        result = vector_create_from_bbox(bbox, projection=srs, out_path=str(tmp_path / "bbox_proj.gpkg"))
        ds = ogr.Open(result)
        assert ds.GetLayer(0).GetSpatialRef().GetAuthorityCode(None) == '3857'

    def test_invalid_bbox(self):
        """Test with invalid bbox."""
        with pytest.raises(AssertionError):
            vector_create_from_bbox([0, 1])  # Invalid bbox length

class TestVectorCreateFromWkt:
    def test_basic_wkt(self, tmp_path):
        """Test creating vector from WKT string."""
        wkt = "POINT (0 0)"
        result = vector_create_from_wkt(wkt, out_path=str(tmp_path / "wkt.gpkg"))
        ds = ogr.Open(result)
        assert ds is not None
        assert ds.GetLayer(0).GetGeomType() == ogr.wkbPoint

    def test_complex_wkt(self, tmp_path):
        """Test creating vector from complex WKT."""
        wkt = "POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))"
        result = vector_create_from_wkt(wkt, out_path=str(tmp_path / "wkt_poly.gpkg"))
        ds = ogr.Open(result)
        assert ds.GetLayer(0).GetGeomType() == ogr.wkbPolygon

    def test_invalid_wkt(self):
        """Test with invalid WKT string."""
        with pytest.raises(ValueError):
            vector_create_from_wkt("")

class TestVectorCreateFromPoints:
    def test_basic_points(self, tmp_path):
        """Test creating vector from list of points."""
        points = [[0, 0], [1, 1], [2, 2]]
        result = vector_create_from_points(points, out_path=str(tmp_path / "points.gpkg"))
        ds = ogr.Open(result)
        assert ds is not None
        layer = ds.GetLayer(0)
        assert layer.GetFeatureCount() == 3

    def test_reverse_xy_order(self, tmp_path):
        """Test creating points with reversed xy order."""
        points = [[0, 1]]
        result = vector_create_from_points(
            points,
            out_path=str(tmp_path / "points_rev.gpkg"),
            reverse_xy_order=True
        )
        ds = ogr.Open(result)
        feat = ds.GetLayer(0).GetNextFeature()
        geom = feat.GetGeometryRef()
        assert geom.GetX() == 1 and geom.GetY() == 0

    def test_invalid_points(self):
        """Test with invalid points."""
        with pytest.raises(AssertionError):
            vector_create_from_points([])  # Empty list
        with pytest.raises(AssertionError):
            vector_create_from_points([[0]])  # Invalid point format

class TestVectorCreateFromGeojson:
    def test_basic_geojson(self, tmp_path):
        """Test creating vector from GeoJSON string."""
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [0, 0]
                    },
                    "properties": {
                        "name": "test_point"
                    }
                }
            ]
        }
        result = vector_create_from_geojson(geojson, out_path=str(tmp_path / "geojson.gpkg"))
        ds = ogr.Open(result)
        assert ds is not None
        layer = ds.GetLayer(0)
        assert layer.GetFeatureCount() == 1

    def test_geojson_with_projection(self, tmp_path):
        """Test creating vector from GeoJSON with specific projection."""
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [0, 0]
                    },
                    "properties": {
                        "name": "test_point"
                    }
                }
            ]
        }
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(3857)
        result = vector_create_from_geojson(geojson, projection=srs, out_path=str(tmp_path / "geojson_proj.gpkg"))
        ds = ogr.Open(result)
        assert ds.GetLayer(0).GetSpatialRef().GetAuthorityCode(None) == '3857'

    def test_invalid_geojson(self):
        """Test with invalid GeoJSON."""
        with pytest.raises(ValueError):
            vector_create_from_geojson("invalid_geojson")

class TestVectorSetCrs:
    def test_set_crs(self, simple_vector, tmp_path):
        """Test setting CRS for a vector."""
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(3857)
        result = vector_set_crs(simple_vector, srs, out_path=str(tmp_path / "set_crs.gpkg"))
        ds = ogr.Open(result)
        assert ds.GetLayer(0).GetSpatialRef().GetAuthorityCode(None) == '3857'

    def test_set_crs_overwrite(self, simple_vector):
        """Test setting CRS with overwrite."""
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(3857)
        result = vector_set_crs(simple_vector, srs, overwrite=True)
        ds = ogr.Open(result)
        assert ds.GetLayer(0).GetSpatialRef().GetAuthorityCode(None) == '3857'

    def test_invalid_crs(self, simple_vector):
        """Test setting invalid CRS."""
        with pytest.raises(ValueError):
            vector_set_crs(simple_vector, "invalid_crs")
