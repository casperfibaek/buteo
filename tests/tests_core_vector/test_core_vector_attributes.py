# pylint: skip-file
# type: ignore

import pytest
from osgeo import ogr, osr

import sys; sys.path.append("../../")

from buteo.core_vector.core_vector_attributes import (
    vector_get_attribute_table,
    vector_add_field,
    vector_set_attribute_table,
    vector_delete_fields,
)


@pytest.fixture
def test_vector(tmp_path):
    """Create a test vector with attributes."""
    vector_path = tmp_path / "test.gpkg"
    driver = ogr.GetDriverByName('GPKG')
    ds = driver.CreateDataSource(str(vector_path))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    layer = ds.CreateLayer('test', srs, ogr.wkbPolygon)
    
    # Add fields
    layer.CreateField(ogr.FieldDefn('name', ogr.OFTString))
    layer.CreateField(ogr.FieldDefn('value', ogr.OFTInteger))
    
    # Create features
    poly1_wkt = "POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))"
    poly1 = ogr.CreateGeometryFromWkt(poly1_wkt)
    
    feature = ogr.Feature(layer.GetLayerDefn())
    feature.SetGeometry(poly1)
    feature.SetField('name', 'feature1')
    feature.SetField('value', 1)
    layer.CreateFeature(feature)
    
    poly2_wkt = "POLYGON ((1 1, 1 2, 2 2, 2 1, 1 1))"
    poly2 = ogr.CreateGeometryFromWkt(poly2_wkt)
    
    feature = ogr.Feature(layer.GetLayerDefn())
    feature.SetGeometry(poly2)
    feature.SetField('name', 'feature2')
    feature.SetField('value', 2)
    layer.CreateFeature(feature)
    
    ds = None
    return str(vector_path)

class TestVectorAttributes:
    def test_get_attribute_table_basic(self, test_vector):
        """Test getting attribute table with default parameters."""
        header, table = vector_get_attribute_table(test_vector)
        
        # Check header
        assert 'fid' in header
        assert 'name' in header
        assert 'value' in header
        
        # Check data
        assert len(table) == 2  # Two features
        assert len(table[0]) == len(header)  # Number of columns matches header
        
        # Check values
        name_idx = header.index('name')
        value_idx = header.index('value')
        assert table[0][name_idx] == 'feature1'
        assert table[0][value_idx] == 1
        assert table[1][name_idx] == 'feature2'
        assert table[1][value_idx] == 2

    def test_get_attribute_table_with_geometry(self, test_vector):
        """Test getting attribute table including geometry."""
        header, table = vector_get_attribute_table(
            test_vector,
            include_geometry=True
        )
        
        assert 'geom' in header
        geom_idx = header.index('geom')
        assert 'POLYGON' in table[0][geom_idx]

    def test_add_field(self, test_vector):
        """Test adding a new field."""
        result = vector_add_field(test_vector, 'new_field', 'string')
        
        # Check if field was added
        ds = ogr.Open(result)
        layer = ds.GetLayer()
        layer_defn = layer.GetLayerDefn()
        field_names = [layer_defn.GetFieldDefn(i).GetName() 
                      for i in range(layer_defn.GetFieldCount())]
        
        assert 'new_field' in field_names
        ds = None

    def test_add_field_different_types(self, test_vector):
        """Test adding fields of different types."""
        types = [
            ('int_field', 'int'),
            ('float_field', 'float'),
            ('string_field', 'string'),
            ('date_field', 'date')
        ]
        
        for field_name, field_type in types:
            vector_add_field(test_vector, field_name, field_type)
        
        ds = ogr.Open(test_vector)
        layer = ds.GetLayer()
        layer_defn = layer.GetLayerDefn()
        field_names = [layer_defn.GetFieldDefn(i).GetName() 
                      for i in range(layer_defn.GetFieldCount())]
        
        for field_name, _ in types:
            assert field_name in field_names
        ds = None

    def test_set_attribute_table(self, test_vector):
        """Test setting attribute table values."""
        header = ['fid', 'name', 'value']
        table = [
            [1, 'updated1', 10],
            [2, 'updated2', 20]
        ]
        
        result = vector_set_attribute_table(test_vector, header, table)
        
        # Verify changes
        ds = ogr.Open(result)
        layer = ds.GetLayer()
        feature = layer.GetFeature(1)
        assert feature.GetField('name') == 'updated1'
        assert feature.GetField('value') == 10
        
        feature = layer.GetFeature(2)
        assert feature.GetField('name') == 'updated2'
        assert feature.GetField('value') == 20
        ds = None

    def test_set_attribute_table_new_field(self, test_vector):
        """Test setting attribute table with a new field."""
        header = ['fid', 'name', 'value', 'new_field']
        table = [
            [1, 'feature1', 1, 'new1'],
            [2, 'feature2', 2, 'new2']
        ]
        
        result = vector_set_attribute_table(test_vector, header, table)
        
        # Verify new field was added with values
        ds = ogr.Open(result)
        layer = ds.GetLayer()
        layer_defn = layer.GetLayerDefn()
        field_names = [layer_defn.GetFieldDefn(i).GetName() 
                      for i in range(layer_defn.GetFieldCount())]
        
        assert 'new_field' in field_names
        
        feature = layer.GetFeature(1)
        assert feature.GetField('new_field') == 'new1'
        ds = None

    def test_delete_fields(self, test_vector):
        """Test deleting fields."""
        result = vector_delete_fields(test_vector, ['value'])
        
        # Verify field was deleted
        ds = ogr.Open(result)
        layer = ds.GetLayer()
        layer_defn = layer.GetLayerDefn()
        field_names = [layer_defn.GetFieldDefn(i).GetName() 
                      for i in range(layer_defn.GetFieldCount())]
        
        assert 'value' not in field_names
        assert 'name' in field_names  # Other field should remain
        ds = None

    def test_delete_nonexistent_field(self, test_vector):
        """Test deleting a field that doesn't exist."""
        result = vector_delete_fields(test_vector, ['nonexistent_field'])
        
        # Should not raise error and should not modify existing fields
        ds = ogr.Open(result)
        layer = ds.GetLayer()
        layer_defn = layer.GetLayerDefn()
        field_count = layer_defn.GetFieldCount()
        assert field_count == 2  # Original fields should remain
        ds = None

    def test_error_invalid_field_type(self, test_vector):
        """Test error when adding field with invalid type."""
        with pytest.raises(ValueError):
            vector_add_field(test_vector, 'invalid_field', 'invalid_type')

    def test_error_set_attribute_mismatch(self, test_vector):
        """Test error when setting attributes with mismatched header/data."""
        header = ['fid', 'name']  # Missing 'value' column
        table = [
            [0, 'feature1', 1],  # Has extra column
            [1, 'feature2', 2]
        ]
        
        with pytest.raises(AssertionError):
            vector_set_attribute_table(test_vector, header, table)