""" Tests for core_raster.py """
# pylint: disable=missing-function-docstring, bare-except


# Standard library
import os
import sys; sys.path.append("../")

# External
from osgeo import ogr

# Internal
from utils_tests import create_sample_vector
from buteo.vector.reproject import vector_reproject

tmpdir = "./tests/tmp/"

sample_vector_wgs84 = create_sample_vector(geom_type="point", num_features=10, epsg_code=4326)
sample_vector_mercator = create_sample_vector(geom_type="point", num_features=10, epsg_code=3857)

def test_vector_reproject_basic():
    reprojected = vector_reproject(
        sample_vector_wgs84,
        sample_vector_mercator,
        out_path="./tests/tmp/reprojected.gpkg",
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

    os.remove(reprojected)
