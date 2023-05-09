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
from buteo.vector.reproject import vector_reproject
from buteo.vector import core_vector

tmpdir = "./tests/tmp/"


def get_vector_projection(vector):
    proj_osr = core_vector._get_basic_metadata_vector(vector)["projection_osr"]
    epsg_code = proj_osr.GetAuthorityCode(None)

    if len(epsg_code) == 0:
        return None

    return int(epsg_code)

# Create sample vector
sample_vector = create_sample_vector(geom_type="point", num_features=10)

def test_vector_reproject_same_projection():
    # Reproject the vector to the same projection
    reprojected_vector = vector_reproject(sample_vector, 4326, copy_if_same=True)

    # Check that the reprojected vector is created
    opened = ogr.Open(reprojected_vector)
    assert opened is not None

    # Check that the projection is the same
    assert get_vector_projection(reprojected_vector) == 4326
    opened = None


def test_vector_reproject_different_projection():
    # Reproject the vector to a different projection
    reprojected_vector = vector_reproject(sample_vector, 3857)

    # Check that the reprojected vector is created
    opened = ogr.Open(reprojected_vector)
    assert opened is not None

    # Check that the projection has changed
    assert get_vector_projection(reprojected_vector) == 3857

    opened = None


def test_vector_reproject_invalid_projection():
    # Attempt to reproject the vector to an invalid projection
    with pytest.raises(ValueError):
        vector_reproject(sample_vector, "invalid_projection")


def test_vector_reproject_output_path():
    out_path = os.path.join(tmpdir, "reprojected_vector.gpkg")
    vector_reproject(sample_vector, 3857, out_path=out_path)

    # Check if the output file is created
    opened = ogr.Open(out_path)
    assert opened is not None
    opened = None

    try:
        os.remove(out_path)
    except:
        pass
