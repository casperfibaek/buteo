""" Tests for core_raster.py """
# pylint: disable=missing-function-docstring, bare-except


# Standard library
import os
import sys; sys.path.append("../")

# External
from osgeo import ogr

# Internal
from buteo.vector.clip import vector_clip
from buteo.vector import core_vector

tmpdir = "./tests/tmp/"

def test_vector_clip_basic():
    # Create sample clip geometry
    source = "./tests/features/test_vector_points.gpkg"
    target = "./tests/features/test_vector_buildings.gpkg"

    out_path = os.path.join(tmpdir, "clipped_vector.gpkg")

    # Clip operation
    clipped = vector_clip(source, target, out_path=out_path, overwrite=True)

    # Check if output vector is created
    out_vector = ogr.Open(clipped)
    assert out_vector is not None

    # Check if the clip operation is performed correctly
    # Here we assume that the clip operation reduces the feature count
    features_after = core_vector._get_basic_metadata_vector(out_vector)["feature_count"]
    assert features_after < core_vector._get_basic_metadata_vector(source)["feature_count"]

    out_vector = None
