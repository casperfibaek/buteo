# pylint: skip-file
# type: ignore

# Standard library
import os

# External
from osgeo import ogr
import pytest

# Internal
from utils_tests import create_sample_vector
from buteo.vector.reproject import vector_reproject
from buteo.vector import core_vector

@pytest.fixture
def sample_vector_wgs84():
    return create_sample_vector(geom_type="point", num_features=10, epsg_code=4326)

@pytest.fixture
def sample_vector_mercator():
    return create_sample_vector(geom_type="point", num_features=10, epsg_code=3857)

def get_vector_projection(vector):
    proj_osr = core_vector._get_basic_metadata_vector(vector)["projection_osr"]
    epsg_code = proj_osr.GetAuthorityCode(None)
    if not epsg_code:
        return None
    return int(epsg_code)

def test_vector_reproject_basic(sample_vector_wgs84, sample_vector_mercator, tmp_path):
    out_path = tmp_path / "reprojected.gpkg"
    reprojected = vector_reproject(
        sample_vector_wgs84,
        sample_vector_mercator,
        out_path=str(out_path),
        overwrite=True,
    )

    assert reprojected is not None, "Reprojected vector is None"

    # Check if the reprojected vector has the same feature count as the input vector
    assert (
        ogr.Open(reprojected).GetLayer().GetFeatureCount()
        == ogr.Open(sample_vector_mercator).GetLayer().GetFeatureCount()
    ), "Reprojected vector has different feature count than input vector"

    # Check if the reprojected vector has the same geometry type as the input vector
    assert (
        ogr.Open(reprojected).GetLayer().GetGeomType()
        == ogr.Open(sample_vector_mercator).GetLayer().GetGeomType()
    ), "Reprojected vector has different geometry type than input vector"

    # Check if the reprojected vector has the same spatial reference as the input vector
    assert (
        ogr.Open(reprojected).GetLayer().GetSpatialRef().ExportToWkt()
        == ogr.Open(sample_vector_mercator).GetLayer().GetSpatialRef().ExportToWkt()
    ), "Reprojected vector has different spatial reference than input vector"

def test_vector_reproject_same_projection(sample_vector_wgs84):
    # Reproject the vector to the same projection
    reprojected_vector = vector_reproject(sample_vector_wgs84, 4326, copy_if_same=True)
    assert reprojected_vector is not None

    # Check that the projection is the same
    assert get_vector_projection(reprojected_vector) == 4326

def test_vector_reproject_different_projection(sample_vector_wgs84):
    # Reproject the vector to a different projection
    reprojected_vector = vector_reproject(sample_vector_wgs84, 3857)
    assert reprojected_vector is not None

    # Check that the projection has changed
    assert get_vector_projection(reprojected_vector) == 3857

def test_vector_reproject_invalid_projection(sample_vector_wgs84):
    # Attempt to reproject the vector to an invalid projection
    with pytest.raises(ValueError):
        vector_reproject(sample_vector_wgs84, "invalid_projection")

def test_vector_reproject_output_path(sample_vector_wgs84, tmp_path):
    out_path = tmp_path / "reprojected_vector.gpkg"
    vector_reproject(sample_vector_wgs84, 3857, out_path=str(out_path))

    # Check if the output file is created
    assert os.path.exists(out_path)
    opened = ogr.Open(str(out_path))
    assert opened is not None
    opened = None
