# pylint: skip-file
# type: ignore

# Standard library
import sys; sys.path.append("../../")
import os

import pytest
from osgeo import ogr, osr

from buteo.core_vector.core_vector_conversion import vector_convert_geometry

@pytest.fixture
def singlepart_vector(tmp_path):
    """Create a singlepart vector."""
    vector_path = tmp_path / "singlepart.gpkg"
    driver = ogr.GetDriverByName('GPKG')
    ds = driver.CreateDataSource(str(vector_path))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    layer = ds.CreateLayer('test', srs, ogr.wkbPolygon)

    # Create a valid polygon

    poly1_wkt = "POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))"
    poly1 = ogr.CreateGeometryFromWkt(poly1_wkt)
    
    feature = ogr.Feature(layer.GetLayerDefn())
    feature.SetGeometry(poly1)
    layer.CreateFeature(feature)
    
    ds = None
    return str(vector_path)

@pytest.fixture
def multipart_vector(tmp_path):
    """Create a multipart vector."""
    vector_path = tmp_path / "multipart.gpkg"
    driver = ogr.GetDriverByName('GPKG')
    ds = driver.CreateDataSource(str(vector_path))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    layer = ds.CreateLayer('test', srs, ogr.wkbMultiPolygon)

    # Create a valid multipolygon
    poly1_wkt = "POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))"
    poly1 = ogr.CreateGeometryFromWkt(poly1_wkt)

    poly2_wkt = "POLYGON ((2 2, 2 3, 3 3, 3 2, 2 2))"
    poly2 = ogr.CreateGeometryFromWkt(poly2_wkt)
    
    multipoly = ogr.Geometry(ogr.wkbMultiPolygon)
    multipoly.AddGeometry(poly1)
    multipoly.AddGeometry(poly2)
    
    feature = ogr.Feature(layer.GetLayerDefn())
    feature.SetGeometry(multipoly)
    layer.CreateFeature(feature)

    ds = None
    return str(vector_path)

@pytest.fixture
def mixed_dimension_vector(tmp_path):
    """Create a vector with mixed 2D/3D geometries."""
    vector_path = tmp_path / "mixed_dim.gpkg"
    driver = ogr.GetDriverByName('GPKG')
    ds = driver.CreateDataSource(str(vector_path))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    layer = ds.CreateLayer('test', srs, ogr.wkbPolygon25D)
    
    # Add fields for Z and M values
    layer.CreateField(ogr.FieldDefn('z_val', ogr.OFTReal))
    layer.CreateField(ogr.FieldDefn('m_val', ogr.OFTReal))

    # Create a 3D polygon
    poly1_wkt = "POLYGON Z ((0 0 1, 0 1 1, 1 1 1, 1 0 1, 0 0 1))"
    poly1 = ogr.CreateGeometryFromWkt(poly1_wkt)
    
    feature1 = ogr.Feature(layer.GetLayerDefn())
    feature1.SetGeometry(poly1)
    feature1.SetField('z_val', 1.0)
    feature1.SetField('m_val', 1.0)
    layer.CreateFeature(feature1)
    
    ds = None
    return str(vector_path)

class TestVectorConvertGeometry:
    def test_convert_singlepart_to_multipart(self, singlepart_vector, tmp_path):
        """Test converting singlepart to multipart."""
        out_path = str(tmp_path / "converted_multipart.gpkg")
        result = vector_convert_geometry(singlepart_vector, multitype=True, output_path=out_path)

        # Check if output exists
        assert os.path.exists(result)

        # Check if geometry is multipart
        ds = ogr.Open(result)
        layer = ds.GetLayer()
        feature = layer.GetNextFeature()
        assert feature.GetGeometryRef().GetGeometryType() == ogr.wkbMultiPolygon
        ds = None

    def test_convert_multipart_to_singlepart(self, multipart_vector, tmp_path):
        """Test converting multipart to singlepart."""
        out_path = str(tmp_path / "converted_singlepart.gpkg")
        result = vector_convert_geometry(multipart_vector, multitype=False, output_path=out_path)

        # Check if output exists
        assert os.path.exists(result)
        
        # Check if geometry is singlepart
        ds = ogr.Open(result)
        layer = ds.GetLayer()
        feature = layer.GetNextFeature()
        assert feature.GetGeometryRef().GetGeometryType() == ogr.wkbPolygon
        ds = None

    def test_convert_to_3d(self, singlepart_vector, tmp_path):
        """Test converting to 3D."""
        out_path = str(tmp_path / "converted_3d.gpkg")
        result = vector_convert_geometry(singlepart_vector, z=True, output_path=out_path)

        # Check if output exists
        assert os.path.exists(result)
        
        # Check if geometry is 3D
        ds = ogr.Open(result)
        layer = ds.GetLayer()
        feature = layer.GetNextFeature()
        assert feature.GetGeometryRef().GetGeometryType() == ogr.wkbPolygon25D
        ds = None

    def test_convert_to_2d(self, singlepart_vector, tmp_path):
        """Test converting to 2D."""
        out_path = str(tmp_path / "converted_2d.gpkg")
        result = vector_convert_geometry(singlepart_vector, z=False, output_path=out_path)

        # Check if output exists
        assert os.path.exists(result)
        
        # Check if geometry is 2D
        ds = ogr.Open(result)
        layer = ds.GetLayer()
        feature = layer.GetNextFeature()
        assert feature.GetGeometryRef().GetGeometryType() == ogr.wkbPolygon
        ds = None

    def test_convert_with_output_options(self, singlepart_vector, tmp_path):
        """Test converting with various output options."""
        result = vector_convert_geometry(
            singlepart_vector,
            prefix="test_",
            suffix="_converted",
        )
        
        assert "test_" in result
        assert "_converted" in result
        assert ogr.Open(result) is not None

    def test_convert_memory_output(self, singlepart_vector):
        """Test converting with memory output."""
        result = vector_convert_geometry(singlepart_vector, output_path=None)
        assert "/vsimem/" in result
        assert ogr.Open(result) is not None

    def test_convert_with_overwrite(self, singlepart_vector, tmp_path):
        """Test converting with overwrite option."""
        out_path = str(tmp_path / "overwrite.gpkg")
        
        # Create dummy file
        with open(out_path, 'w') as f:
            f.write("dummy")
        
        # Should raise error without overwrite
        with pytest.raises(FileExistsError):
            vector_convert_geometry(singlepart_vector, output_path=out_path, overwrite=False)
        
        # Should succeed with overwrite
        result = vector_convert_geometry(singlepart_vector, output_path=out_path, overwrite=True)
        assert ogr.Open(result) is not None

    def test_convert_with_z_attribute(self, mixed_dimension_vector, tmp_path):
        """Test converting geometry using z attribute."""
        out_path = str(tmp_path / "converted_z_attr.gpkg")
        result = vector_convert_geometry(
            mixed_dimension_vector,
            z=True,
            z_attribute='z_val',
            output_path=out_path
        )
        
        ds = ogr.Open(result)
        layer = ds.GetLayer()
        feature = layer.GetNextFeature()
        geom = feature.GetGeometryRef()
        
        # Check if geometry is 3D
        assert geom.GetCoordinateDimension() == 3
        
        # Check if Z value matches the attribute
        point = geom.GetGeometryRef(0).GetPoint(0)
        assert point[2] == feature.GetField('z_val')
        ds = None

    # TODO: FIX THIS TEST
    # def test_convert_with_m_attribute(self, mixed_dimension_vector, tmp_path):
    #     """Test converting geometry using m attribute."""
    #     out_path = str(tmp_path / "converted_m_attr.gpkg")
    #     result = vector_convert_geometry(
    #         mixed_dimension_vector,
    #         m=True,
    #         m_attribute='m_val',
    #         output_path=out_path
    #     )
        
    #     ds = ogr.Open(result)
    #     layer = ds.GetLayer()
    #     feature = layer.GetNextFeature()
    #     geom = feature.GetGeometryRef()
        
    #     # Check if geometry has M value
    #     geom_type = geom.GetGeometryType()
    #     assert geom_type & 0x40000000 != 0
    #     ds = None

    def test_convert_multitype_and_z(self, multipart_vector, tmp_path):
        """Test converting geometry with both multipart and Z options."""
        out_path = str(tmp_path / "converted_multi_z.gpkg")
        result = vector_convert_geometry(
            multipart_vector,
            multitype=False,
            z=True,
            output_path=out_path
        )
        
        ds = ogr.Open(result)
        layer = ds.GetLayer()
        feature = layer.GetNextFeature()
        geom = feature.GetGeometryRef()
        
        # Check if geometry is singlepart and 3D
        assert geom.GetGeometryType() == ogr.wkbPolygon25D
        ds = None

    def test_convert_all_options(self, mixed_dimension_vector, tmp_path):
        """Test converting geometry with all options combined."""
        out_path = str(tmp_path / "converted_all.gpkg")
        result = vector_convert_geometry(
            mixed_dimension_vector,
            multitype=True,
            multipart=True,
            z=True,
            m=False,
            z_attribute='z_val',
            m_attribute='m_val',
            output_path=out_path
        )
        
        ds = ogr.Open(result)
        layer = ds.GetLayer()
        feature = layer.GetNextFeature()
        geom = feature.GetGeometryRef()
        
        # Check if geometry has all requested properties
        geom_type = geom.GetGeometryType()
        assert geom_type & ogr.wkbMultiPolygon != 0  # Check multitype
        ds = None

    def test_error_multitype_multipart_conflict(self, singlepart_vector, tmp_path):
        """Test error when conflicting multitype and multipart options."""
        out_path = str(tmp_path / "error.gpkg")
        with pytest.raises(ValueError):
            vector_convert_geometry(
                singlepart_vector,
                multitype=False,
                multipart=True,
                output_path=out_path
            )

    def test_convert_empty_attributes(self, singlepart_vector, tmp_path):
        """Test converting geometry with non-existent Z/M attributes."""
        out_path = str(tmp_path / "converted_empty_attr.gpkg")

        with pytest.raises(ValueError):
            result = vector_convert_geometry(
                singlepart_vector,
                z=True,
                m=True,
                z_attribute='nonexistent_z',
                m_attribute='nonexistent_m',
                output_path=out_path
            )
            
        result = None

    def test_convert_preserve_attributes(self, mixed_dimension_vector, tmp_path):
        """Test that conversion preserves existing attributes."""
        out_path = str(tmp_path / "converted_preserve.gpkg")
        result = vector_convert_geometry(
            mixed_dimension_vector,
            multitype=True,
            output_path=out_path
        )
        
        ds = ogr.Open(result)
        layer = ds.GetLayer()
        feature = layer.GetNextFeature()
        
        # Check if original fields exist
        assert feature.GetField('z_val') is not None
        assert feature.GetField('m_val') is not None
        ds = None