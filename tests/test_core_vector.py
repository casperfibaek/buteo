""" Tests for core_raster.py """
# pylint: disable=missing-function-docstring, bare-except


# Standard library
import os
import sys; sys.path.append("../")
import shutil

# External
import pytest
from osgeo import ogr

# Internal
from utils_tests import create_sample_raster, create_sample_vector
from buteo.vector import core_vector

tmpdir = "./tests/tmp/"

def test_vector_open():
    sample_vector = create_sample_vector()

    opened_vector = core_vector.vector_open(sample_vector, writeable=False, allow_raster=False)
    assert isinstance(opened_vector, ogr.DataSource), "Opened vector should be of ogr.DataSource type"
    opened_vector = None

def test_vector_open_multiple_vectors():
    sample_vector1 = create_sample_vector()
    sample_vector2 = create_sample_vector()

    opened_vectors = core_vector.vector_open([sample_vector1, sample_vector2], writeable=False, allow_raster=False)

    assert isinstance(opened_vectors, list), "Output should be a list of opened vectors"
    for opened_vector in opened_vectors:
        assert isinstance(opened_vector, ogr.DataSource), "Opened vector should be of ogr.DataSource type"
    opened_vectors = None

def test_vector_open_allow_raster():
    sample_raster = create_sample_raster()

    opened_vector = core_vector.vector_open(sample_raster, writeable=False, allow_raster=True)
    assert isinstance(opened_vector, ogr.DataSource), "Opened vector should be of ogr.DataSource type"
    opened_vector = None

def test_vector_open_fail_on_raster():
    sample_raster = create_sample_raster()

    with pytest.raises(RuntimeError):
        core_vector.vector_open(sample_raster, writeable=False, allow_raster=False)

def test__vector_open():
    sample_vector = create_sample_vector()

    opened_vector = core_vector._vector_open(sample_vector, writeable=False, allow_raster=False)
    assert isinstance(opened_vector, ogr.DataSource), "Opened vector should be of ogr.DataSource type"
    opened_vector = None

def test_get_basic_metadata_vector():
    sample_vector = create_sample_vector()

    metadata = core_vector._get_basic_metadata_vector(sample_vector)

    assert isinstance(metadata, dict), "Returned metadata should be a dictionary"
    assert metadata["layer_count"] > 0, "Layer count should be greater than 0"
    assert "layers" in metadata, "Metadata should have a layers key"
    assert isinstance(metadata["layers"], list), "Metadata layers should be a list"

    for layer in metadata["layers"]:
        assert "layer_name" in layer, "Layer should have a layer_name key"
        assert "bbox" in layer, "Layer should have a bbox key"
        assert "field_names" in layer, "Layer should have a field_names key"
        assert "field_types" in layer, "Layer should have a field_types key"

def test_get_basic_metadata_vector_with_attributes():
    attribute_data = [
        {"name": "attr1", "type": ogr.OFTInteger, "values": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
        {"name": "attr2", "type": ogr.OFTString, "values": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]},
    ]
    sample_vector = create_sample_vector(attribute_data=attribute_data)

    metadata = core_vector._get_basic_metadata_vector(sample_vector)

    for layer in metadata["layers"]:
        assert set(layer["field_names"]) == set(["attr1", "attr2"]), "Field names should match the attribute_data provided"
        assert len(layer["field_types"]) == len(attribute_data), "Field types length should match the attribute_data length"

def test_get_basic_metadata_vector_in_memory():
    sample_vector = create_sample_vector()

    metadata = core_vector._get_basic_metadata_vector(sample_vector)

    assert metadata["in_memory"] is True, "In-memory flag should be True for in-memory vector"

def test_get_basic_metadata_vector_path():
    sample_vector = create_sample_vector()

    metadata = core_vector._get_basic_metadata_vector(sample_vector)

    assert metadata["path"].startswith("/vsimem/"), "Path should start with /vsimem/ for in-memory vector"
    assert metadata["basename"] == os.path.basename(metadata["path"]), "basename should match the path basename"
    assert metadata["name"] == os.path.splitext(os.path.basename(metadata["path"]))[0], "name should match the path basename without extension"
    assert metadata["folder"] == os.path.dirname(metadata["path"]), "folder should match the path directory"
    assert metadata["ext"] == os.path.splitext(metadata["path"])[1], "ext should match the path extension"


# Create sample vectors
sample_vector1 = create_sample_vector(geom_type="polygon", num_features=10, epsg_code=4326)
sample_vector2 = create_sample_vector(geom_type="point", num_features=5, epsg_code=4326)

# Define filter functions
filter_function1 = lambda attr: True  # All features pass
filter_function2 = lambda attr: False  # No features pass
filter_function3 = lambda attr: bool(attr["id"] % 2 == 0)  # Even id features pass


def test__vector_filter():
    out_path = os.path.join(tmpdir, "filtered_01.gpkg")
    filtered_vector = core_vector._vector_filter(sample_vector1, filter_function1, out_path=out_path, overwrite=True)
    assert os.path.exists(filtered_vector)

    # Check if the filter function was applied correctly
    src_ds = ogr.Open(sample_vector1)
    src_layer = src_ds.GetLayer()
    src_feature_count = src_layer.GetFeatureCount()
    dst_ds = ogr.Open(filtered_vector)
    dst_layer = dst_ds.GetLayer()
    dst_feature_count = dst_layer.GetFeatureCount()
    assert src_feature_count == dst_feature_count

    src_ds = None; dst_ds = None

    try:
        os.remove(out_path)
    except:
        pass


def test_vector_filter_single_vector():
    out_path = os.path.join(tmpdir, "filtered_02.gpkg")
    filtered_vector = core_vector.vector_filter(sample_vector1, filter_function1, out_path=out_path, overwrite=True)
    assert os.path.exists(filtered_vector)

    # Check if the filter function was applied correctly
    src_ds = ogr.Open(sample_vector1)
    src_layer = src_ds.GetLayer()
    src_feature_count = src_layer.GetFeatureCount()
    dst_ds = ogr.Open(filtered_vector)
    dst_layer = dst_ds.GetLayer()
    dst_feature_count = dst_layer.GetFeatureCount()
    assert src_feature_count == dst_feature_count

    src_ds = None; dst_ds = None

    try:
        os.remove(out_path)
    except:
        pass


def test_vector_filter_multiple_vectors():
    out_path1 = os.path.join(tmpdir, "filtered_03.gpkg")
    out_path2 = os.path.join(tmpdir, "filtered_04.gpkg")
    out_paths = [out_path1, out_path2]
    filtered_vectors = core_vector.vector_filter([sample_vector1, sample_vector2], filter_function1, out_paths, overwrite=True)
    for fv in filtered_vectors:
        assert os.path.exists(fv)

    # Check if the filter function was applied correctly for both vectors
    for i in range(2):
        src_ds = ogr.Open([sample_vector1, sample_vector2][i])
        src_layer = src_ds.GetLayer()
        src_feature_count = src_layer.GetFeatureCount()
        dst_ds = ogr.Open(filtered_vectors[i])
        dst_layer = dst_ds.GetLayer()
        dst_feature_count = dst_layer.GetFeatureCount()
        assert src_feature_count == dst_feature_count
        src_ds = None; dst_ds = None

    try:
        os.remove(out_path1)
        os.remove(out_path2)
    except:
        pass


def test_vector_add_index_single_vector():
    sample_vector3 = create_sample_vector(geom_type="polygon", num_features=10, epsg_code=4326)

    result = core_vector.vector_add_index(sample_vector3)
    assert result == sample_vector3

    # Check if the index was added correctly
    ds = ogr.Open(sample_vector3)
    layer = ds.GetLayer()
    layer_name = layer.GetName()
    geom_name = layer.GetGeometryColumn()
    sql = f"SELECT HasSpatialIndex('{layer_name}', '{geom_name}');"
    index_check = ds.ExecuteSQL(sql, dialect="SQLITE")
    index_status = index_check.GetFeature(0).GetField(0)
    assert index_status == 1
    ds = None

    try:
        os.remove(sample_vector3)
    except:
        pass


def test_vector_add_index_multiple_vectors():
    sample_vector4 = create_sample_vector(geom_type="point", num_features=5, epsg_code=4326)
    sample_vector5 = create_sample_vector(geom_type="point", num_features=5, epsg_code=4326)

    result = core_vector.vector_add_index([sample_vector4, sample_vector5])
    assert result == [sample_vector4, sample_vector5]

    # Check if the index was added correctly for both vectors
    for out_path in [sample_vector4, sample_vector5]:
        ds = ogr.Open(out_path)
        layer = ds.GetLayer()
        layer_name = layer.GetName()
        geom_name = layer.GetGeometryColumn()
        sql = f"SELECT HasSpatialIndex('{layer_name}', '{geom_name}');"
        index_check = ds.ExecuteSQL(sql, dialect="SQLITE")
        index_status = index_check.GetFeature(0).GetField(0)
        assert index_status == 1
        ds = None


def test_vector_get_attribute_table_single_vector():
    attribute_table = core_vector.vector_get_attribute_table(sample_vector1, return_header=False)

    # Check if the attribute table is not empty and has the correct number of features
    assert attribute_table
    assert len(attribute_table) == 10

def test_vector_get_attribute_table_single_vector_include_fid():
    attribute_table = core_vector.vector_get_attribute_table(sample_vector1, include_fids=True, return_header=False)

    # Check if the FIDs are included
    assert all(isinstance(row[0], int) for row in attribute_table)

def test_vector_get_attribute_table_single_vector_include_geometry():
    attribute_table = core_vector.vector_get_attribute_table(sample_vector1, include_geometry=True, return_header=False)

    # Check if the geometry is included
    assert all(isinstance(row[-1], str) for row in attribute_table)

def test_vector_get_attribute_table_single_vector_no_attributes():
    attribute_table = core_vector.vector_get_attribute_table(sample_vector1, include_attributes=False, return_header=False)

    # Check if the attribute table has only FID values
    assert all(len(row) == 1 for row in attribute_table)

def test_vector_get_attribute_table_multiple_vectors():
    attribute_tables = core_vector.vector_get_attribute_table([sample_vector1, sample_vector2], return_header=False)

    # Check if the attribute tables are not empty and have the correct number of features
    assert attribute_tables
    assert len(attribute_tables[0]) == 10
    assert len(attribute_tables[1]) == 5

# Create sample multi-layer vectors
sample_vector10 = create_sample_vector(geom_type="point", num_features=10, epsg_code=4326, n_layers=3)

def test_vector_filter_layer_by_index():
    out_path = core_vector.vector_filter_layer(sample_vector10, 1)

    # Check if output vector is created and has only one layer
    out_vector = ogr.Open(out_path)
    assert out_vector.GetLayerCount() == 1
    out_vector = None

def test_vector_filter_layer_by_name():
    in_vector = ogr.Open(sample_vector10)
    layer_name = in_vector.GetLayer(2).GetName()

    out_path = core_vector.vector_filter_layer(sample_vector10, layer_name)

    # Check if output vector is created and has only one layer
    out_vector = ogr.Open(out_path)
    assert out_vector.GetLayerCount() == 1
    in_vector = None; out_vector = None

def test_vector_filter_layer_invalid_layer_name_or_idx():
    with pytest.raises(RuntimeError):
        core_vector.vector_filter_layer(sample_vector10, 3.14)  # Invalid layer_name_or_idx data type

def test_vector_filter_layer_output_path():
    out_path = os.path.join(tmpdir, "filtered_vector.gpkg")
    out_path = core_vector.vector_filter_layer(sample_vector10, 0, out_path=out_path, overwrite=True)

    opened = ogr.Open(out_path)

    # Check if the output file is created
    assert opened is not None
    assert os.path.exists(out_path)

    opened = None
    try:
        os.remove(out_path)
    except:
        pass
