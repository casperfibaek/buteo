""" Tests for core_raster.py """
# pylint: disable=missing-function-docstring, bare-except


# Standard library
import os
import sys; sys.path.append("../")

# External
from osgeo import ogr

# Internal
from utils_tests import create_sample_raster
from buteo.raster.vectorize import raster_vectorize

tmpdir = "./tests/tmp/"



def test_raster_vectorize():
    raster_path = create_sample_raster()
    output_vector_path = os.path.join(tmpdir, "vectorize_01.gpkg")

    raster_vectorize(raster_path, out_path=output_vector_path, band=1)

    output_vector = ogr.Open(output_vector_path)

    assert output_vector is not None, "Output vector file should exist"
    output_vector = None

    try:
        os.remove(output_vector_path)
    except:
        pass

def test_raster_vectorize_multiple_rasters():
    raster_path1 = create_sample_raster()
    raster_path2 = create_sample_raster()
    raster_list = [raster_path1, raster_path2]

    output_vector_path1 = os.path.join(tmpdir, "vectorize_02.gpkg")
    output_vector_path2 = os.path.join(tmpdir, "vectorize_03.gpkg")

    output_vector_paths = raster_vectorize(raster_list, out_path=[output_vector_path1, output_vector_path2], band=1)

    assert isinstance(output_vector_paths, list), "Output should be a list of vector file paths"

    for output_vector_path in output_vector_paths:
        output_vector = ogr.Open(output_vector_path)
        assert output_vector is not None, "Output vector file should exist"
        output_vector = None

        try:
            os.remove(output_vector_path)
        except:
            pass
