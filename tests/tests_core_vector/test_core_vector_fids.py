# pylint: skip-file
# type: ignore

import pytest
from osgeo import ogr, osr

# Standard library
import sys; sys.path.append("../../")


from buteo.core_vector.core_vector_fids import (
    vector_reset_fids,
    vector_create_attribute_from_fid,
)


@pytest.fixture
def test_vector(tmp_path):
    """Create a vector file with multiple features for testing."""
    vector_path = tmp_path / "test.gpkg"
    driver = ogr.GetDriverByName('GPKG')
    ds = driver.CreateDataSource(str(vector_path))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)

    layer = ds.CreateLayer('test', srs, ogr.wkbPolygon)

    # Add fields
    field_defn = ogr.FieldDefn("value", ogr.OFTInteger)
    layer.CreateField(field_defn)

    # Create features with specific FIDs
    for i in range(3):
        wkt = f"POLYGON (({i} {i}, {i} {i+1}, {i+1} {i+1}, {i+1} {i}, {i} {i}))"
        poly = ogr.CreateGeometryFromWkt(wkt)

        feature = ogr.Feature(layer.GetLayerDefn())
        feature.SetField("value", i * 10)
        feature.SetGeometry(poly)
        feature.SetFID(i * 2) # Non-sequential FIDs
        layer.CreateFeature(feature)

    ds = None
    return str(vector_path)


class TestVectorResetFids:
    def test_reset_fids_basic(self, test_vector, tmp_path):
        """Test basic FID reset functionality."""
        out_path = str(tmp_path / "reset.gpkg")
        result = vector_reset_fids(test_vector, out_path=out_path)

        ds = ogr.Open(result)
        layer = ds.GetLayer()

        # Check if FIDs are sequential
        fids = []
        for feature in layer:
            fids.append(feature.GetFID())

        assert fids == list(range(len(fids)))
        ds = None

    def test_reset_fids_inplace(self, test_vector):
        """Test resetting FIDs in place."""
        result = vector_reset_fids(test_vector, inplace=True)

        ds = ogr.Open(result)
        layer = ds.GetLayer()

        fids = []
        for feature in layer:
            fids.append(feature.GetFID())

        assert fids == list(range(len(fids)))
        ds = None

    def test_reset_fids_memory(self, test_vector):
        """Test resetting FIDs with memory output."""
        result = vector_reset_fids(test_vector)
        assert "/vsimem/" in result

        ds = ogr.Open(result)
        assert ds is not None
        ds = None


class TestVectorCreateAttributeFromFID:
    def test_create_attribute_from_fid_basic(self, test_vector, tmp_path):
        """Test creating attribute from FID functionality."""
        out_path = str(tmp_path / "attr_from_fid.gpkg")
        result = vector_create_attribute_from_fid(test_vector, out_path=out_path, attribute_name="custom_id")

        ds = ogr.Open(result)
        layer = ds.GetLayer()
        layer_defn = layer.GetLayerDefn()

        field_names = [layer_defn.GetFieldDefn(i).GetName() for i in range(layer_defn.GetFieldCount())]
        assert "custom_id" in field_names

        # Check that custom_id values match the original FIDs (0, 2, 4)
        expected_ids = [0, 2, 4]
        for i, feature in enumerate(layer):
            assert feature.GetField("custom_id") == expected_ids[i]

        ds = None

    def test_create_attribute_from_fid_inplace(self, test_vector):
        """Test creating attribute from FID in place."""
        result = vector_create_attribute_from_fid(test_vector, inplace=True, attribute_name="id_field")

        ds = ogr.Open(result)
        layer = ds.GetLayer()
        layer_defn = layer.GetLayerDefn()

        field_names = [layer_defn.GetFieldDefn(i).GetName() for i in range(layer_defn.GetFieldCount())]
        assert "id_field" in field_names

        for feature in layer:
            assert feature.GetField("id_field") == feature.GetFID()

        ds = None

    def test_error_handling(self, test_vector, tmp_path):
        """Test error handling for invalid inputs."""
        out_path = str(tmp_path / "exists.gpkg")

        # Create dummy file
        with open(out_path, 'w') as f:
            f.write("dummy")

        # Should raise error without overwrite
        with pytest.raises(FileExistsError):
            vector_create_attribute_from_fid(test_vector, out_path=out_path, overwrite=False)

        # Test invalid layer name
        with pytest.raises(ValueError):
            vector_create_attribute_from_fid(test_vector, layer_name_or_id="invalid_layer")
