""" Tests for core_raster.py """
# pylint: disable=missing-function-docstring, bare-except


# Standard library
import os
import sys; sys.path.append("../")

# External
from osgeo import ogr
import pytest

# Internal
from utils_tests import create_sample_vector
from buteo.vector.clip import vector_clip
from buteo.vector import core_vector

tmpdir = "./tests/tmp/"

# Create sample vector
sample_vector = create_sample_vector(geom_type="point", num_features=10)
sample_area = core_vector._get_basic_metadata_vector(sample_vector)["area"]

# Create the test base like so:
def create_sample_wkt():
    # Define simple geometry in WKT format
    wkt_point = "POINT (30 10)"
    wkt_polygon = "POLYGON ((30 10, 40 40, 20 40, 10 20, 30 10))"

    # Create a new spatial reference
    srs = ogr.osr.SpatialReference()
    srs.ImportFromEPSG(4326)  # WGS84

    # Parse WKT geometries
    point = ogr.CreateGeometryFromWkt(wkt_point, srs)
    polygon = ogr.CreateGeometryFromWkt(wkt_polygon, srs)

    return point, polygon



# def test_vector_clip_basic():
#     # Create sample clip geometry
#     clip_geom = create_sample_vector(geom_type="polygon", num_features=1)

#     out_path = os.path.join(tmpdir, "clipped_vector.gpkg")

#     # Clip operation
#     clipped = vector_clip(sample_vector, clip_geom, out_path=out_path, overwrite=True)

#     import pdb; pdb.set_trace()

#     # Check if output vector is created
#     out_vector = ogr.Open(clipped)
#     assert out_vector is not None

#     # Check if the clip operation is performed correctly
#     # Here we assume that the clip operation reduces the feature count
#     features_after = core_vector._get_basic_metadata_vector(out_vector)["feature_count"]
#     assert features_after < core_vector._get_basic_metadata_vector(sample_vector)["feature_count"]

#     out_vector = None


# def test_vector_clip_to_extent():
#     # Create sample clip geometry
#     clip_geom = create_sample_vector(geom_type="polygon", num_features=1)

#     out_path = os.path.join(tmpdir, "clipped_vector_extent.gpkg")

#     # Clip operation
#     clipped = vector_clip(sample_vector, clip_geom, out_path=out_path, to_extent=True, overwrite=True)

#     # Check if output vector is created
#     out_vector = ogr.Open(clipped)
#     assert out_vector is not None

#     # Check if the clip operation is performed correctly
#     # Here we assume that the clip operation reduces the feature count
#     features_after = core_vector._get_basic_metadata_vector(out_vector)["feature_count"]
#     assert features_after < core_vector._get_basic_metadata_vector(sample_vector)["feature_count"]

#     out_vector = None


# def test_vector_clip_preserve_fid():
#     # Create sample clip geometry
#     clip_geom = create_sample_vector(geom_type="polygon", num_features=1)

#     out_path = os.path.join(tmpdir, "clipped_vector_fid.gpkg")

#     # Clip operation with preserve_fid
#     clipped = vector_clip(sample_vector, clip_geom, out_path=out_path, preserve_fid=True, overwrite=True)

#     # Check if output vector is created
#     out_vector = ogr.Open(clipped)
#     assert out_vector is not None

#     # Check if the feature IDs are preserved
#     out_layer = out_vector.GetLayer()
#     for feature in out_layer:
#         assert feature.GetFID() is not None

#     out_vector = None


# def test_vector_clip_invalid_input():
#     with pytest.raises(TypeError):
#         vector_clip(123, sample_vector, out_path=os.path.join(tmpdir, "invalid_input.gpkg"), overwrite=True)


# def test_vector_clip_output_path():
#     out_path = os.path.join(tmpdir, "clipped_vector.gpkg")
#     vector_clip(sample_vector, sample_vector, out_path=out_path, overwrite=True)

#     # Check if the output file is created
#     opened = ogr.Open(out_path)
#     assert opened is not None

#     opened = None

#     try:
#         os.remove(out_path)
#     except:
#         pass


# def test_vector_clip_with_uuid_timestamp():
#     out_path = os.path.join(tmpdir, "clipped_vector_")

#     # Clip operation with add_uuid and add_timestamp
#     clipped = vector_clip(sample_vector, sample_vector, out_path=out_path, add_uuid=True, overwrite=True)

#     # Check if output vector is created and the filename contains uuid and timestamp
#     out_vector = ogr.Open(clipped)
#     assert out_vector is not None

#     out_vector = None
